"""
Wire the UltraFlash SR sidecar into interactive_demo.py.

Rank 0 sends each finished chunk latent to the sidecar and gets HR frames back,
instead of decoding locally. The SR runs out-of-process on its own GPU on
purpose: a second CUDA context inside a torchrun rank destabilizes NCCL, which
is the same reason persona_aux is already split out.

Idempotent; keeps a .bak2 on first patch. Inert unless --sr_url is passed.
"""

import shutil
import sys

PATH = "/workspace/soulx/SoulX-LiveAct/interactive_demo.py"

# ---- 1. CLI flags -----------------------------------------------------------
A_ARG = '''    parser.add_argument("--fp8_kv_cache", action="store_true", default=False)'''
P_ARG = '''    parser.add_argument("--fp8_kv_cache", action="store_true", default=False)
    parser.add_argument("--sr_url", type=str, default=None,
                        help="UltraFlash SR sidecar base URL; omit to stream at native res")
    parser.add_argument("--sr_scale", type=int, default=2)'''

# ---- 2. output size (ffmpeg needs the post-SR size) -------------------------
A_SIZE = '''        self.width, self.height = _parse_size(args.size)
        self.fps = args.fps'''
P_SIZE = '''        self.width, self.height = _parse_size(args.size)
        # SR changes only the OUTPUT size; the generator keeps running at self.width
        # x self.height, so frame_len / KV-cache sizing below must not use these.
        self.sr_url = getattr(args, "sr_url", None)
        self.sr_scale = int(getattr(args, "sr_scale", 2)) if self.sr_url else 1
        self.out_width = self.width * self.sr_scale
        self.out_height = self.height * self.sr_scale
        self.fps = args.fps'''

# ---- 3. ffmpeg raw input size ----------------------------------------------
A_FF = '''                f"{self.width}x{self.height}",'''
P_FF = '''                f"{self.out_width}x{self.out_height}",'''

# ---- 4. swap the decode for an SR round-trip -------------------------------
A_DEC = '''                    if iteration == 0:
                        vids = self.vae.decode(latent)
                    else:
                        vids = self.vae.decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent'''
P_DEC = '''                    if self.sr_url:
                        # Sidecar owns upsampler + SR DiT + tiny decoder and keeps
                        # its own streaming caches, so nothing is decoded here.
                        vids = None
                        sr_u8 = _sr_request(
                            self.sr_url, latent, cur_ctx, self.rank
                        )
                    elif iteration == 0:
                        vids = self.vae.decode(latent)
                    else:
                        vids = self.vae.decode(
                            torch.concat([pre_latent[:, -3:], latent], dim=1)
                        )[:, :, 9:]
                    pre_latent = latent'''

# ---- 5. use the SR frames on the emit path ---------------------------------
A_EMIT = '''                if self.rank == 0:
                    u8 = (
                        ((vids.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .contiguous()
                        .cpu()
                    )
                    vq.put((u8.numpy(), frames_emitted))'''
P_EMIT = '''                if self.rank == 0:
                    if sr_u8 is not None:
                        arr = sr_u8
                    else:
                        arr = (
                            ((vids.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5)
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .contiguous()
                            .cpu()
                            .numpy()
                        )
                    n_emit = arr.shape[0]
                    vq.put((arr, frames_emitted))'''

# frame accounting must follow whichever path produced the frames
A_CNT1 = '''                        n_frames = vids.shape[2]'''
P_CNT1 = '''                        n_frames = arr.shape[0]'''
A_CNT2 = '''                    frames_emitted += vids.shape[2]'''
P_CNT2 = '''                    frames_emitted += n_emit'''

# ---- 6. helper + per-session cache reset -----------------------------------
A_HELP = '''SIZE_ALIGN = 16'''
P_HELP = '''SIZE_ALIGN = 16


def _sr_request(base, latent, ctx, rank, timeout=60):
    """POST one chunk latent to the SR sidecar, return HR uint8 [T,H,W,C]."""
    import io as _io

    import numpy as _np
    import requests as _rq

    if rank != 0:
        return None
    buf = _io.BytesIO()
    torch.save(
        {
            "latent": latent.detach().to(torch.bfloat16).cpu(),
            "ctx": [c.detach().to(torch.bfloat16).cpu() for c in ctx],
        },
        buf,
    )
    r = _rq.post(f"{base}/sr", data=buf.getvalue(), timeout=timeout)
    r.raise_for_status()
    hdr, _, payload = r.content.partition(b"\\n")
    t, h, w, c = (int(x) for x in hdr.decode().split(","))
    return _np.frombuffer(payload, dtype=_np.uint8).reshape(t, h, w, c)


def _sr_reset(base):
    try:
        import requests as _rq

        _rq.post(f"{base}/reset", timeout=30)
    except Exception as e:
        print(f"[sr] reset failed: {e}", flush=True)'''


PATCHES = [
    ("cli args", A_ARG, P_ARG),
    ("out size", A_SIZE, P_SIZE),
    ("ffmpeg size", A_FF, P_FF),
    ("decode swap", A_DEC, P_DEC),
    ("emit path", A_EMIT, P_EMIT),
    ("chunk log count", A_CNT1, P_CNT1),
    ("frames_emitted", A_CNT2, P_CNT2),
    ("helpers", A_HELP, P_HELP),
]


def main():
    src = open(PATH).read()
    if "_sr_request" in src:
        print("already patched")
        return 0
    for name, anchor, _ in PATCHES:
        if anchor not in src:
            print(f"ANCHOR NOT FOUND: {name}\n---\n{anchor}\n---")
            return 1
        if src.count(anchor) != 1:
            print(f"ANCHOR NOT UNIQUE ({src.count(anchor)}x): {name}")
            return 1
    shutil.copy(PATH, PATH + ".bak2")
    for name, anchor, patch in PATCHES:
        src = src.replace(anchor, patch, 1)
    # reset sidecar caches whenever a session starts
    # NB: "pre_latent = None" also appears in the warmup path, so anchor on the
    # live loop's unique follow-on line.
    live_anchor = "        pre_latent = None\n        iteration = 0\n"
    assert src.count(live_anchor) == 1, "live-loop anchor not unique"
    src = src.replace(
        live_anchor,
        "        pre_latent = None\n"
        "        sr_u8 = None\n"
        "        if self.sr_url and self.rank == 0:\n"
        "            _sr_reset(self.sr_url)\n"
        "        iteration = 0\n",
    )
    open(PATH, "w").write(src)
    print("patched ok (backup at .bak2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
