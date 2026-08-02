"""Where the demo's files live, and what the GPU underneath it can actually do.

This exists because both facts used to be assumptions: `/workspace/soulx/...`
hardcoded in a dozen places, and "the GPU is an H200" implied by every default.
Moving the demo to another box meant editing source. Now it means setting
SOULX_ROOT and letting `plan()` pick the paths the hardware supports.

Three things live here:

  paths()            resolve weights / decoders / assets from the environment
  plan()             probe the GPUs -> attention backend, FP8 scope, 1-GPU vs SP
  build_decode_fn()  the decoder registry ("ordinary Wan VAE" vs the tiny ones)

Import is cheap: torch is imported lazily so a launcher can call `paths()` or
print the registry without paying for CUDA init.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _abs(value: str) -> str:
    return os.path.abspath(os.path.expanduser(value))


@dataclass(frozen=True)
class Paths:
    """Everything the demo reads from disk, rooted at SOULX_ROOT.

    Layout (all overridable individually):

        $SOULX_ROOT/
          weights/LiveAct/                  SoulX checkpoints + Wan2.1 VAE + t5 + clip
          weights/chinese-wav2vec2-base/    audio encoder
          decoders/taew2_1.pth              tiny decoders (lightx2v)
          assets/reference.png              default reference image
    """

    root: str
    weights: str
    decoders: str
    assets: str

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.weights, "LiveAct")

    @property
    def wav2vec_dir(self) -> str:
        return os.path.join(self.weights, "chinese-wav2vec2-base")

    @property
    def default_image(self) -> str | None:
        """`assets/reference.png` if present, else any image in assets/."""
        preferred = os.path.join(self.assets, "reference.png")
        if os.path.exists(preferred):
            return preferred
        if os.path.isdir(self.assets):
            for name in sorted(os.listdir(self.assets)):
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    return os.path.join(self.assets, name)
        return None

    def missing(self) -> list[str]:
        """Required inputs that are not on disk. Empty list means ready to run."""
        required = {
            "LiveAct checkpoints": self.ckpt_dir,
            "Wan2.1 VAE": os.path.join(self.ckpt_dir, "Wan2.1_VAE.pth"),
            "umt5 text encoder": os.path.join(
                self.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"
            ),
            "wav2vec2": self.wav2vec_dir,
        }
        return [
            f"{label}: {path}"
            for label, path in required.items()
            if not os.path.exists(path)
        ]


def paths() -> Paths:
    root = _abs(os.environ.get("SOULX_ROOT") or "/workspace/soulx")
    return Paths(
        root=root,
        weights=_abs(os.environ.get("SOULX_WEIGHTS") or os.path.join(root, "weights")),
        decoders=_abs(
            os.environ.get("SOULX_DECODERS") or os.path.join(root, "decoders")
        ),
        assets=_abs(os.environ.get("SOULX_ASSETS") or os.path.join(root, "assets")),
    )


# ---------------------------------------------------------------------------
# Hardware capabilities
# ---------------------------------------------------------------------------

# Resident footprint of the generator process measured at 720x416 with FP8 on
# the block matmuls: 76.4GB. The margin covers the allocator's fragmentation and
# the larger KV cache at bigger resolutions -- get this wrong on the low side and
# `plan()` picks a single GPU that then OOMs mid-session.
MODEL_VRAM_GB = 82.0

# FlashAttention-3 is Hopper-only. On anything else the xformers flash3 op either
# reports unavailable or silently is not built, so ask for it by architecture
# rather than by trying and catching.
_FA3_ARCH = (9, 0)

# FP8 e4m3 tensor cores: Ada (8.9), Hopper (9.0), Blackwell (10.x/12.x).
_FP8_MIN_ARCH = (8, 9)

_ARCH_NAMES = {
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (10, 3): "Blackwell",
    (12, 0): "Blackwell",
}


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    cc: tuple
    mem_gb: float

    @property
    def arch(self) -> str:
        return _ARCH_NAMES.get(self.cc, f"sm_{self.cc[0]}{self.cc[1]}")

    def __str__(self) -> str:
        return (
            f"GPU{self.index} {self.name} "
            f"({self.arch}, sm_{self.cc[0]}{self.cc[1]}, {self.mem_gb:.0f}GB)"
        )


def gpus() -> list:
    import torch

    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append(
            Gpu(
                index=i,
                name=props.name,
                cc=(props.major, props.minor),
                mem_gb=props.total_memory / (1024**3),
            )
        )
    return out


@dataclass
class Plan:
    """What to run, decided from the hardware rather than from a preset."""

    parallel: str  # "single" | "sp"
    gen_gpus: list
    aux_gpu: int
    attn: str  # "sdpa" | "fa3"
    fp8: str  # "off" | "blocks" | "all"
    notes: list = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return len(self.gen_gpus)

    def describe(self) -> str:
        lines = [
            f"  parallel : {self.parallel} (world_size={self.world_size}, "
            f"gen GPUs {self.gen_gpus}, aux GPU {self.aux_gpu})",
            f"  attention: {self.attn}",
            f"  fp8      : {self.fp8}",
        ]
        lines += [f"  note     : {n}" for n in self.notes]
        return "\n".join(lines)


def attention_backend(prefer: str | None = None) -> str:
    """Pick the self-attention backend for the streaming KV path.

    "auto" (the default) means FA3 on Hopper and SDPA everywhere else. Asking for
    fa3 explicitly on a non-Hopper card is honoured as far as it can be -- it
    warns and falls back rather than failing the launch, because a wrong
    SOULX_ATTN should not cost a 15-minute model load.
    """
    want = (prefer or os.environ.get("SOULX_ATTN") or "auto").lower()
    if want == "sdpa":
        return "sdpa"

    devices = gpus()
    cc = devices[0].cc if devices else (0, 0)

    if cc != _FA3_ARCH:
        if want == "fa3":
            arch = devices[0].arch if devices else "no CUDA device"
            print(
                f"[attn] fa3 requested but FlashAttention-3 is Hopper-only "
                f"(this is {arch}); using SDPA",
                flush=True,
            )
        return "sdpa"

    # Hopper: use xformers' bundled FA3. A separately built flash_attn_3 wheel in
    # the same process aborts at import -- both register the same torch library.
    try:
        from xformers.ops import fmha

        if not fmha.flash3.FwOp.is_available():
            raise RuntimeError("xformers flash3 unavailable on this build")
        return "fa3"
    except Exception as exc:  # noqa: BLE001
        if want == "fa3":
            print(
                f"[attn] fa3 unavailable ({type(exc).__name__}: {exc}); using SDPA",
                flush=True,
            )
        return "sdpa"


def fp8_supported() -> bool:
    devices = gpus()
    return bool(devices) and devices[0].cc >= _FP8_MIN_ARCH


def plan(
    gpus_wanted: str = "auto",
    fp8: str = "blocks",
    attn: str = "auto",
    vram_needed_gb: float = MODEL_VRAM_GB,
) -> Plan:
    """Decide 1-GPU vs sequence-parallel, and gate FP8/FA3 on the architecture.

    `gpus_wanted` is "auto" | "1" | "2" | "0,3" -- auto keeps the model on one
    card when it fits (no NCCL, and the aux LLM/TTS can co-locate) and only
    splits across cards when it does not.
    """
    devices = gpus()
    notes = []

    if not devices:
        return Plan("single", [0], 0, "sdpa", "off", ["no CUDA device visible"])

    if gpus_wanted not in ("auto", ""):
        if gpus_wanted.isdigit():
            chosen = list(range(min(int(gpus_wanted), len(devices))))
        else:
            chosen = [int(x) for x in gpus_wanted.split(",") if x.strip() != ""]
    else:
        biggest = max(d.mem_gb for d in devices)
        if biggest >= vram_needed_gb:
            chosen = [max(devices, key=lambda d: d.mem_gb).index]
            notes.append(
                f"one {biggest:.0f}GB card holds the ~{vram_needed_gb:.0f}GB model; "
                f"staying single-GPU (no NCCL)"
            )
        elif len(devices) >= 2:
            chosen = [d.index for d in devices[:2]]
            notes.append(
                f"no single card fits ~{vram_needed_gb:.0f}GB "
                f"(largest is {biggest:.0f}GB); splitting sequence-parallel"
            )
        else:
            chosen = [devices[0].index]
            notes.append(
                f"only {biggest:.0f}GB available for a ~{vram_needed_gb:.0f}GB model "
                f"-- expect OOM; drop resolution or add a GPU"
            )

    parallel = "sp" if len(chosen) > 1 else "single"

    # persona_aux (Qwen + Kokoro) wants its own card whenever one is idle. Under
    # sequence parallelism it MUST have one -- a second CUDA context inside a
    # torchrun rank destabilises NCCL. With world_size==1 there is no NCCL, so
    # sharing is merely second-best rather than dangerous.
    spare = [d.index for d in devices if d.index not in chosen]
    aux_gpu = spare[0] if spare else chosen[0]
    if not spare:
        notes.append(
            "persona_aux shares the generator's card"
            + (" -- NCCL risk under SP" if parallel == "sp" else " (~4GB)")
        )

    resolved_attn = attention_backend(attn)
    if resolved_attn == "sdpa" and attn == "auto" and devices[0].cc != _FA3_ARCH:
        notes.append(f"FA3 skipped: Hopper-only, this is {devices[0].arch}")

    resolved_fp8 = fp8
    if fp8 != "off" and not fp8_supported():
        resolved_fp8 = "off"
        notes.append(f"FP8 disabled: {devices[0].arch} has no FP8 tensor cores")

    return Plan(parallel, chosen, aux_gpu, resolved_attn, resolved_fp8, notes)


# ---------------------------------------------------------------------------
# Decoder registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderSpec:
    """A way of turning latents into frames.

    `ms` and `mae` are measured on this pipeline's real dumped latents at
    720x416, against the full Wan2.1 VAE. MAE is a WEAK signal -- it is a
    per-frame average and cannot see temporal smearing, which is how v3 scores
    respectably yet visibly smears a moving hand. Judge on moving video.
    """

    key: str
    kind: str  # "wan" | "tiny" | "lightvae" | "v3"
    label: str
    file: str | None = None
    need_scaled: bool = False
    ms: int | None = None
    mae: float | None = None
    note: str = ""


DECODERS = {
    "wan": DecoderSpec(
        key="wan",
        kind="wan",
        label="full Wan2.1 VAE",
        ms=452,
        mae=0.0,
        note="the reference; 507MB, ships with the LiveAct checkpoints",
    ),
    "taew2_1": DecoderSpec(
        key="taew2_1",
        kind="tiny",
        label="taew2_1 tiny decoder",
        file="taew2_1.pth",
        need_scaled=False,
        ms=32,
        mae=2.77,
        note="official lightx2v checkpoint, 23MB -- the default",
    ),
    "lighttaew2_1": DecoderSpec(
        key="lighttaew2_1",
        kind="tiny",
        label="lighttaew2_1 tiny decoder",
        file="lighttaew2_1.pth",
        need_scaled=True,
        ms=32,
        mae=3.76,
        note="official, 45MB",
    ),
    "lightvaew2_1": DecoderSpec(
        key="lightvaew2_1",
        kind="lightvae",
        label="lightvaew2_1 pruned VAE",
        file="lightvaew2_1.pth",
        ms=133,
        mae=3.34,
        note="75% pruned but CAUSAL Conv3D -- the only fast option that models time",
    ),
    "v3": DecoderSpec(
        key="v3",
        kind="v3",
        label="UltraFlash v3 (reconstructed)",
        file="v1.1-ultra-decoder-v3-ema_decoder.pth",
        ms=31,
        mae=6.41,
        note="RECONSTRUCTED, smears on motion -- kept for comparison only",
    ),
}

# The two lightx2v TAEs disagree on latent normalisation and the wrong flag looks
# like a broken model rather than a wrong setting (taew2_1 with need_scaled=True
# scores MAE 24.48; lighttaew2_1 without it, 21.56). Encoded in the specs above
# on purpose -- do not "fix" it to a single value.


def decoder_help() -> str:
    rows = []
    for spec in DECODERS.values():
        mae = "ref" if spec.mae == 0.0 else (f"{spec.mae:.2f}" if spec.mae else "-")
        rows.append(
            f"    {spec.key:<14} {str(spec.ms) + 'ms':>7}  MAE {mae:>5}  {spec.note}"
        )
    return "\n".join(rows)


def build_decode_fn(key: str, device, p: Paths | None = None) -> Callable | None:
    """Return a drop-in replacement for `WanVAE.decode`, or None to keep it.

    Contract every option here honours, because A/V sync depends on it: a latent
    of shape (C, T, h, w) decodes to (1, 3, T*4-3, h*8, w*8). That is what makes
    the 21-then-32 frame counts per chunk come out right whichever decoder runs.
    """
    import torch

    p = p or paths()
    if key not in DECODERS:
        raise ValueError(f"--vae must be one of {list(DECODERS)} (got {key!r})")
    spec = DECODERS[key]

    if spec.kind == "wan":
        return None  # caller keeps the Wan VAE's own decode

    ckpt = os.path.join(p.decoders, spec.file)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"decoder {key!r} needs {ckpt}\n"
            f"  fetch it with:  scope-soulx fetch --decoders\n"
            f"  or point SOULX_DECODERS at the directory holding {spec.file}"
        )

    dev = f"cuda:{device}" if isinstance(device, int) else str(device)

    if spec.kind == "tiny":
        from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny

        model = (
            WanVAE_tiny(
                vae_path=ckpt,
                dtype=torch.bfloat16,
                device=dev,
                need_scaled=spec.need_scaled,
            )
            .to(dev)
            .eval()
        )
        return lambda latent, _m=model: _m.decode(latent)

    if spec.kind == "lightvae":
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

        model = LightVAE(
            vae_path=ckpt,
            dtype=torch.bfloat16,
            device=device,
            use_lightvae=True,
            parallel=False,
        )
        return lambda latent, _m=model: _m.decode(latent)

    if spec.kind == "v3":
        # Optional: the reconstruction lives outside this tree.
        import sys

        uflash = os.environ.get("SOULX_UFLASH", "/workspace/uflash")
        for extra in (uflash, os.path.join(uflash, "UltraFlash/inference")):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        from ultra_dec_v3 import UltraDecoderV3

        model = UltraDecoderV3(ckpt).to(dev, torch.float16).eval().requires_grad_(False)
        # v3 emits [-1,1] directly and must NOT be rescaled the way the TAEs are.
        return lambda latent, _m=model: _m.decode_latents(latent.unsqueeze(0))

    raise ValueError(f"unknown decoder kind {spec.kind!r}")


# ---------------------------------------------------------------------------
# Resolution presets
# ---------------------------------------------------------------------------

# Cost tracks PIXEL COUNT, not orientation -- 336*592 measured identically to
# 592*336 -- so portrait is free and these are just the vertical framings.
RESOLUTIONS = {
    "416x720": (416, 720),  # vertical quality reference
    "368x640": (368, 640),  # vertical, ~1.4x realtime on one H200
    "336x592": (336, 592),  # vertical, fastest usable
    "720x416": (720, 416),  # landscape quality reference
    "640x368": (640, 368),
}


def parse_resolution(value: str) -> tuple:
    """Accept 416x720, 416*720, or a preset name. Returns (width, height)."""
    if value in RESOLUTIONS:
        return RESOLUTIONS[value]
    for sep in ("x", "X", "*"):
        if sep in value:
            w, h = value.split(sep, 1)
            width, height = int(w), int(h)
            break
    else:
        raise ValueError(f"resolution must look like 416x720 (got {value!r})")

    # 16 = vae_stride(8) * patch_size(2); a non-multiple silently truncates the
    # latent grid and desyncs frame_len from the KV cache.
    if width % 16 or height % 16:
        raise ValueError(f"{width}x{height}: both dimensions must be multiples of 16")
    return width, height


# ---------------------------------------------------------------------------
# CLI: `--plan` for the launcher to eval, `--doctor` for a human
# ---------------------------------------------------------------------------


def _fp8_selftest() -> str:
    """Does an FP8 matmul actually run and produce sane numbers on this card?

    Worth checking rather than assuming: this pipeline has twice shipped silent,
    plausible-looking corruption that still emitted frames.
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    if not fp8_supported():
        return f"unsupported on {gpus()[0].arch} (needs sm_89+)"
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor(1.0, device="cuda")
        out = torch._scaled_mm(
            a.to(torch.float8_e4m3fn),
            b.t().contiguous().t().to(torch.float8_e4m3fn),
            scale_a=scale,
            scale_b=scale,
            out_dtype=torch.bfloat16,
        )
        ref = a @ b
        err = (out.float() - ref.float()).abs().max().item()
        rel = err / ref.float().abs().max().item()
        if not torch.isfinite(out).all():
            return "FAILED: non-finite output"
        return f"ok (max rel err {rel:.3f} vs bf16)"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED: {type(exc).__name__}: {exc}"


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="SoulX runtime probe")
    ap.add_argument("--plan", action="store_true", help="print shell-eval'able plan")
    ap.add_argument("--doctor", action="store_true", help="human-readable diagnosis")
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--fp8", default="blocks")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--size", default="368x640")
    ap.add_argument("--vae", default="taew2_1")
    args = ap.parse_args(argv)

    p = paths()
    width, height = parse_resolution(args.size)

    # KV and attention cost scale with frame_len, so the VRAM estimate has to
    # move with resolution -- the 76GB measurement was taken at 720x416.
    ref_px = 720 * 416
    needed = MODEL_VRAM_GB - 6.0 + 6.0 * (width * height) / ref_px
    decided = plan(args.gpus, args.fp8, args.attn, vram_needed_gb=needed)

    if args.plan:
        print(f"SOULX_PARALLEL={decided.parallel}")
        print(f"SOULX_GEN_GPUS={','.join(str(i) for i in decided.gen_gpus)}")
        print(f"SOULX_WORLD_SIZE={decided.world_size}")
        print(f"SOULX_AUX_GPU={decided.aux_gpu}")
        print(f"SOULX_ATTN={decided.attn}")
        print(f"SOULX_FP8={decided.fp8}")
        print(f"SOULX_SIZE={width}x{height}")
        for note in decided.notes:
            print(f"# note: {note}")
        return 0

    print("SoulX-LiveAct runtime\n")
    print("paths")
    print(f"  root     : {p.root}")
    print(f"  weights  : {p.weights}")
    print(f"  decoders : {p.decoders}")
    print(f"  assets   : {p.assets}")
    print(f"  image    : {p.default_image or 'NONE FOUND'}")
    missing = p.missing()
    print(
        "  status   : " + ("ready" if not missing else "MISSING " + "; ".join(missing))
    )

    print("\ngpus")
    devices = gpus()
    if not devices:
        print("  none visible")
    for gpu in devices:
        print(f"  {gpu}")

    print(f"\nplan for {width}x{height} (needs ~{needed:.0f}GB)")
    print(decided.describe())

    print("\nchecks")
    print(f"  fp8 matmul   : {_fp8_selftest()}")
    try:
        import torch

        print(f"  torch        : {torch.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  torch        : MISSING ({exc})")

    spec = DECODERS.get(args.vae)
    if spec and spec.file:
        ckpt = os.path.join(p.decoders, spec.file)
        state = "ok" if os.path.exists(ckpt) else "MISSING"
        print(f"  decoder {args.vae:<10}: {state} ({ckpt})")
    else:
        print(f"  decoder {args.vae:<10}: ships with the LiveAct checkpoints")

    return 0 if not missing else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main())
