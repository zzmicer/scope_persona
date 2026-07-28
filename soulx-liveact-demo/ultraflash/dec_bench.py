"""
Decode is the remaining cost in the SR cascade, so price the options directly.

LR = what SoulX decodes today (52x90 latent -> 720x416).
HR = what it would decode after 2x SR (104x180 latent -> 1440x832).
The question is whether any decoder puts HR inside the leftover chunk budget
once the SR stage has taken its 0.67s.
"""

import sys
import time

import torch

sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

VAE_PATH = "/workspace/soulx/weights/LiveAct/Wan2.1_VAE.pth"


def bench(vae, c, t, h, w, n=4, tag=""):
    lat = torch.randn(c, t, h, w, device="cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        for i in range(n + 1):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            out = vae.decode(lat)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  {tag:28s} lat({c},{t},{h},{w}) -> {tuple(out.shape)}  {dt*1000:7.1f} ms")
    return dt


def main():
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

    for use_light in (False, True):
        tag = "LightVAE(pruned)" if use_light else "full Wan2.1 VAE"
        print(f"\n=== {tag} ===")
        try:
            vae = LightVAE(vae_path=VAE_PATH, dtype=torch.bfloat16, device=0,
                           use_lightvae=use_light, parallel=False)
            vae.model.eval()
        except Exception as e:
            print(f"  UNAVAILABLE: {type(e).__name__}: {e}")
            continue
        try:
            bench(vae, 16, 8, 52, 90, tag=f"{tag} LR 720x416")
            bench(vae, 16, 8, 104, 180, tag=f"{tag} HR 1440x832")
        except Exception as e:
            print(f"  decode failed: {type(e).__name__}: {e}")
        del vae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
