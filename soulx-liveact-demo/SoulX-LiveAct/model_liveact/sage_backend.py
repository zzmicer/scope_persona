# SageAttention as an explicitly scoped, self-verifying backend.
#
# Upstream gates SageAttention on a bare `try: from sageattention import sageattn`,
# which turns it on for EVERY attention in the model the moment the package is
# importable -- self-attention, the I2V cross-attention over a 257-token image
# context, and the audio cross-attention that drives lipsync. That is the same
# mistake `--fp8 all` made: an aggressive numeric format aimed at layers with
# near-zero FLOPs and maximum sensitivity. So sage is scoped here the way FP8 is,
# and defaults to self-attention only -- the one call site with enough work in it
# to pay for the quantization.
#
# It also VERIFIES the kernel before trusting it. On this H200 the shipped
# `_qattn_sm90` extension returned garbage (rel err ~45 against an fp32 SDPA
# reference, at every shape, dtype and layout tested) while the sm89 fp8 and
# fp16 kernels were accurate to ~1-4%. `sageattn()`'s own dispatcher picks the
# sm90 kernel on Hopper, so the whole feature read as "SageAttention produces
# noise" and got blocked at the import. A one-off check against SDPA on first
# use costs microseconds and turns that class of bad build into a printed
# warning and a working fallback instead of a corrupted stream.

import os

import torch
import torch.nn.functional as F

__all__ = ["SAGE_SELF", "SAGE_CROSS", "SAGE_AUDIO", "sage_attn", "resolved_name",
           "describe"]

# Accept anything within this relative error of an fp32 SDPA reference. Sage's
# int8 QK quantization lands at 1-4% on real shapes; bf16 SDPA itself is ~0.2%.
# A broken kernel misses by three orders of magnitude, so the threshold does not
# need to be delicate.
_MAX_REL_ERR = 0.05

# Per-architecture kernel preference, fastest first. Each entry is the name of a
# sageattention entry point plus the kwargs that kernel requires -- they are not
# interchangeable (the sm90 kernel rejects any pv_accum_dtype but "fp32+fp32").
_PREFERENCE = {
    (9, 0): [  # Hopper
        ("sageattn_qk_int8_pv_fp8_cuda_sm90", {"pv_accum_dtype": "fp32+fp32"}),
        ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16"}),
        ("sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
        ("sageattn_qk_int8_pv_fp16_triton", {}),
    ],
    (8, 9): [  # Ada
        ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16"}),
        ("sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
    ],
    (12, 0): [  # Blackwell
        ("sageattn_qk_int8_pv_fp8_cuda",
         {"qk_quant_gran": "per_warp", "pv_accum_dtype": "fp32+fp16"}),
        ("sageattn_qk_int8_pv_fp16_triton", {}),
    ],
}
_FALLBACK_PREFERENCE = [("sageattn", {})]


def _scopes():
    """Which call sites sage covers. `SOULX_SAGE_SCOPE=self+cross`, `all`, ..."""
    raw = os.environ.get("SOULX_SAGE_SCOPE", "self").strip().lower()
    if raw in ("all", "*"):
        return {"self", "cross", "audio"}
    if raw in ("", "off", "none"):
        return set()
    return {part for part in raw.replace("+", ",").split(",") if part.strip()}


_ENABLED = os.environ.get("SOULX_ATTN", "sdpa").lower() == "sage"
_SCOPE = _scopes() if _ENABLED else set()

SAGE_SELF = "self" in _SCOPE
SAGE_CROSS = "cross" in _SCOPE
SAGE_AUDIO = "audio" in _SCOPE

# device index -> (callable, kwargs) or None once we know sage is unusable there
_RESOLVED = {}


def _rel_err(out, ref):
    return (out.float() - ref).abs().mean().item() / max(ref.abs().mean().item(), 1e-12)


def _verify(fn, kwargs, device):
    """Does this kernel agree with SDPA? Returns the relative error, or None.

    The probe tensors come from a private Generator, NOT `torch.manual_seed`:
    verification happens lazily inside the first forward, and touching the
    global CUDA RNG there would shift per-chunk latent noise -- the same way
    rank0-only TTS once desynced the sequence-parallel ranks and turned half
    of every frame to static.
    """
    gen = torch.Generator(device=device).manual_seed(0)
    shape = (1, 512, 8, 128)  # small, but past every kernel's block-size threshold
    q, k, v = (torch.randn(shape, device=device, dtype=torch.bfloat16,
                           generator=gen)
               for _ in range(3))
    ref = F.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (q, k, v))).transpose(1, 2)
    try:
        out = fn(q, k, v, tensor_layout="NHD", is_causal=False, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- any failure just disqualifies it
        print(f"[sage] {fn.__name__} unusable ({type(exc).__name__}: {exc})",
              flush=True)
        return None
    if torch.isnan(out).any():
        print(f"[sage] {fn.__name__} returned NaN; skipping", flush=True)
        return None
    return _rel_err(out, ref)


def _resolve(device):
    """Pick and validate a kernel for `device`, once."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx in _RESOLVED:
        return _RESOLVED[idx]

    _RESOLVED[idx] = None  # pessimistic, so a failure below is not retried
    try:
        import sageattention
    except ImportError as exc:
        print(f"[sage] sageattention is not importable ({exc}); using SDPA",
              flush=True)
        return None

    cc = torch.cuda.get_device_capability(idx)
    for name, kwargs in _PREFERENCE.get(cc, _FALLBACK_PREFERENCE):
        fn = getattr(sageattention, name, None)
        if fn is None:
            continue
        err = _verify(fn, kwargs, device)
        if err is None:
            continue
        if err > _MAX_REL_ERR:
            print(f"[sage] {name} FAILS its accuracy check on sm{cc[0]}{cc[1]} "
                  f"(rel err {err:.3f} vs SDPA); trying the next kernel",
                  flush=True)
            continue
        print(f"[sage] {name} verified on sm{cc[0]}{cc[1]} (rel err {err:.4f}), "
              f"scope={'+'.join(sorted(_SCOPE))}", flush=True)
        _RESOLVED[idx] = (fn, kwargs)
        return _RESOLVED[idx]

    print(f"[sage] no accurate SageAttention kernel on sm{cc[0]}{cc[1]}; "
          f"falling back to SDPA", flush=True)
    return None


def sage_attn(q, k, v, tensor_layout="NHD", is_causal=False):
    """Sage attention, or None when it is off or unusable -- caller falls back.

    Returning None rather than silently doing SDPA keeps the fallback visible at
    the call site, which matters because the two paths have different layouts
    and different masking support.
    """
    resolved = _resolve(q.device)
    if resolved is None:
        return None
    fn, kwargs = resolved
    return fn(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, **kwargs)


def resolved_name(device=None):
    """Name of the verified kernel on `device`, or None if sage is unusable.

    The sequence-parallel path does not call `sage_attn` -- yunchang owns that
    attention and takes an `AttnType` instead -- so it needs to know WHICH
    kernel passed, not just that one did: `AttnType.SAGE_FP8_SM90` is hardwired
    to the sm90 kernel, the exact one that failed verification here.
    """
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    resolved = _resolve(device)
    return None if resolved is None else resolved[0].__name__


def describe():
    if not _ENABLED:
        return "sage=off"
    return f"sage=on scope={'+'.join(sorted(_SCOPE)) or 'none'}"
