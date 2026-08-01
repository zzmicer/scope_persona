"""Compare every available Wan2.1 decoder against the full Wan VAE, on real
dumped SoulX latents at native 592x336-class resolution.

Candidates:
  wan          full Wan2.1 VAE                    (reference, 507MB)
  lightvae     lightvaew2_1, 75% pruned Conv3D    (32MB, causal -> temporal)
  v3           our RECONSTRUCTED UltraFlash TAEHV (25MB)
  lighttae     lighttaew2_1, "quality near official" (45MB)
  tae          taew2_1, baseline tiny             (23MB)

The point is to find out whether an OFFICIALLY RELEASED checkpoint matches the
reconstructed v3, which would let us drop a rebuilt artifact from the pipeline.

Note the output-range convention differs: lightx2v's WanVAE_tiny.decode does
`.mul_(2).sub_(1)` (net emits [0,1]); our v3 emits [-1,1] directly and must NOT
be rescaled. Both are handled by using each class's own decode path.
"""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/uflash")
sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

VAE_PATH = "/workspace/soulx/weights/LiveAct/Wan2.1_VAE.pth"
LX = "/workspace/uflash/ckpt_lx2v"
V3_CK = "/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth"
CHUNKS = [1, 5, 9, 13]


def to_u8(v):
    if v.dim() == 4:
        v = v.unsqueeze(0)
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
    from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny

    lats = []
    for ci in CHUNKS:
        d = torch.load(f"/workspace/uflash/dump/chunk_{ci:04d}.pt",
                       map_location="cpu", weights_only=False)
        lats.append(d["latent"].to("cuda:0", torch.bfloat16))
    print(f"{len(lats)} latents, shape {tuple(lats[0].shape)}\n")

    ref_vae = LightVAE(vae_path=VAE_PATH, dtype=torch.bfloat16, device=0,
                       use_lightvae=False, parallel=False)
    ref_vae.model.eval()
    refs, t_ref = [], 0.0
    for lat in lats:
        out, dt = timeit(lambda l=lat: ref_vae.decode(l))
        refs.append(to_u8(out))
        t_ref += dt
    t_ref /= len(lats)
    print(f"{'decoder':12s} {'ms':>8s} {'speedup':>8s} {'MAE/255':>9s}   note")
    print(f"{'wan (ref)':12s} {t_ref*1000:8.1f} {'1.0x':>8s} {'-':>9s}   full Wan2.1 VAE")
    del ref_vae
    torch.cuda.empty_cache()

    def score(name, dec_fn, note):
        try:
            tot, maes = 0.0, []
            for lat, ref in zip(lats, refs):
                out, dt = timeit(lambda l=lat: dec_fn(l))
                tot += dt
                a = to_u8(out)
                n = min(a.shape[0], ref.shape[0])
                maes.append(float(np.abs(a[-n:].astype(np.int16)
                                         - ref[-n:].astype(np.int16)).mean()))
            dt = tot / len(lats)
            print(f"{name:12s} {dt*1000:8.1f} {t_ref/dt:7.1f}x {np.mean(maes):9.2f}   {note}")
        except Exception as e:
            print(f"{name:12s} {'FAIL':>8s} {'':>8s} {'':>9s}   {type(e).__name__}: {str(e)[:70]}")

    # LightVAE: 75% pruned official, causal Conv3D
    try:
        lv = LightVAE(vae_path=f"{LX}/lightvaew2_1.pth", dtype=torch.bfloat16,
                      device=0, use_lightvae=True, parallel=False)
        lv.model.eval()
        score("lightvae", lambda l: lv.decode(l), "lightvaew2_1, pruned Conv3D")
        del lv
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{'lightvae':12s} {'FAIL':>8s} {'':>8s} {'':>9s}   {type(e).__name__}: {str(e)[:70]}")

    # our reconstructed v3
    from ultra_dec_v3 import UltraDecoderV3
    v3 = UltraDecoderV3(V3_CK).to("cuda:0", torch.float16).eval().requires_grad_(False)
    score("v3 (ours)", lambda l: v3.decode_latents(l.unsqueeze(0)), "RECONSTRUCTED UltraFlash")
    del v3
    torch.cuda.empty_cache()

    # official lightx2v TAEs
    for tag, ck, note in (
        ("lighttae", "lighttaew2_1.pth", "official, 'near-official quality'"),
        ("tae", "taew2_1.pth", "official baseline tiny"),
    ):
        try:
            m = WanVAE_tiny(vae_path=f"{LX}/{ck}", dtype=torch.bfloat16, device="cuda:0")
            m = m.to("cuda:0").eval()
            score(tag, lambda l, _m=m: _m.decode(l), note)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{tag:12s} {'FAIL':>8s} {'':>8s} {'':>9s}   {type(e).__name__}: {str(e)[:70]}")


if __name__ == "__main__":
    main()
