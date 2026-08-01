"""Step 1 gate: does the v3 tiny decoder hold up at NATIVE resolution?

It was only ever run on SR-DiT output latents (104x180 -> 1440x832). The plan
for 1xH200 drops the SR stage and keeps only this decoder, so it has to decode
SoulX's own 52x90 latents -> 720x416 at Wan-VAE quality.

Reference is the full Wan2.1 VAE decode of the SAME dumped latents. Anything
near the 5.2/255 MAE the HR path scored is a pass; a colour cast or structural
break shows up as a large MAE and in the side-by-side.
"""

import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/uflash")
sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

from ultra_dec_v3 import UltraDecoderV3  # noqa: E402

DEC_CK = "/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth"
VAE_PATH = "/workspace/soulx/weights/LiveAct/Wan2.1_VAE.pth"
CHUNKS = [1, 5, 9, 13]
OUT = "/workspace/uflash/cmp/v3_native_vs_wan.jpg"


def to_u8(v):
    """(1,C,T,H,W) in [-1,1] -> (T,H,W,C) uint8."""
    return (((v.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) * 127.5)
            .clamp(0, 255).to(torch.uint8).cpu().numpy())


def timeit(fn, n=4):
    with torch.no_grad():
        for i in range(n + 1):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            out = fn()
        torch.cuda.synchronize()
    return out, (time.time() - t0) / n


def main():
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

    dec = UltraDecoderV3(DEC_CK).to("cuda:0", torch.float16).eval().requires_grad_(False)
    print("v3 decoder loaded strict=True")

    vae = LightVAE(vae_path=VAE_PATH, dtype=torch.bfloat16, device=0,
                   use_lightvae=False, parallel=False)
    vae.model.eval()
    print("full Wan2.1 VAE loaded\n")

    rows, maes = [], []
    t_wan = t_v3 = 0.0

    for ci in CHUNKS:
        d = torch.load(f"/workspace/uflash/dump/chunk_{ci:04d}.pt",
                       map_location="cpu", weights_only=False)
        lat = d["latent"].to("cuda:0", torch.bfloat16)          # (16,T,52,90)

        ref, dt_w = timeit(lambda: vae.decode(lat))
        if ref.dim() == 4:
            ref = ref.unsqueeze(0)
        out, dt_v = timeit(lambda: dec.decode_latents(lat.unsqueeze(0)))
        t_wan += dt_w
        t_v3 += dt_v

        a, b = to_u8(ref), to_u8(out)
        n = min(a.shape[0], b.shape[0])
        a, b = a[-n:], b[-n:]                                   # v3 trims lead frames
        mae = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        maes.append(mae)
        print(f"chunk {ci:04d}: {tuple(lat.shape)} -> {a.shape[1]}x{a.shape[2]}  "
              f"MAE {mae:5.2f}/255   wan {dt_w*1000:6.1f}ms   v3 {dt_v*1000:6.1f}ms")

        k = n // 2
        rows.append(np.concatenate([a[k], b[k]], axis=1))

    nw = len(CHUNKS)
    print(f"\nmean MAE {np.mean(maes):.2f}/255")
    print(f"wan VAE decode {t_wan/nw*1000:.1f} ms   v3 decode {t_v3/nw*1000:.1f} ms   "
          f"speedup {t_wan/t_v3:.1f}x  (saves {(t_wan-t_v3)/nw*1000:.0f} ms/chunk)")

    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT, quality=92)
    print(f"\nside-by-side (left=Wan VAE, right=v3) -> {OUT}")


if __name__ == "__main__":
    main()
