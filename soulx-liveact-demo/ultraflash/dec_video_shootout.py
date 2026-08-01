"""Decode ONE set of dumped SoulX latents through every candidate decoder, so the
only variable between the output videos is the decoder itself.

Still-frame MAE missed what matters here: the reconstructed v3 looks blurry on a
MOVING hand, which a per-frame average error simply does not capture. These clips
put the same wave through each decoder.

Reproduces the streaming windowed decode exactly -- chunk 0 decodes alone, later
chunks prepend the previous 3 latent frames and keep only the last n*4 -- so the
32-frame chunk seams appear here the same way they do live.
"""

import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/uflash")
sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

DUMP = "/workspace/uflash/dump_wave"
OUT = "/workspace/uflash/vae_clips"
LX = "/workspace/uflash/ckpt_lx2v"
WAN = "/workspace/soulx/weights/LiveAct/Wan2.1_VAE.pth"
V3 = "/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth"
PCM = "/workspace/uflash/work_ref/audio.pcm"
FPS, SR = 16, 16000


def to_u8(v):
    if v.dim() == 4:
        v = v.unsqueeze(0)
    return (((v.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) * 127.5)
            .clamp(0, 255).to(torch.uint8).cpu().numpy())


def run(tag, decode, lats):
    """decode: (C,T,h,w) -> pixels. Windowed exactly like the live path."""
    frames, prev, t0 = [], None, time.time()
    for lat in lats:
        n = lat.shape[1]
        if prev is None:
            out = decode(lat)
        else:
            out = decode(torch.cat([prev[:, -3:], lat], dim=1))
            out = out[:, :, -(n * 4):] if out.dim() == 5 else out[:, -(n * 4):]
        prev = lat
        frames.append(to_u8(out))
    dt = time.time() - t0
    arr = np.concatenate(frames, axis=0)

    d = f"{OUT}/frames_{tag}"
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    from PIL import Image
    for i, fr in enumerate(arr):
        Image.fromarray(fr).save(f"{d}/{i:06d}.jpg", quality=93)

    mp4 = f"{OUT}/wave_{tag}.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS), "-i", f"{d}/%06d.jpg"]
    if os.path.exists(PCM):
        cmd += ["-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", PCM]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if os.path.exists(PCM):
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += [mp4]
    subprocess.run(cmd, check=True)
    print(f"  {tag:12s} {arr.shape[0]:3d} frames  {arr.shape[2]}x{arr.shape[1]}  "
          f"decode {dt:5.2f}s  -> {mp4}", flush=True)
    return mp4


def main():
    from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
    from lightx2v.models.video_encoders.hf.wan.vae_tiny import WanVAE_tiny

    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in os.listdir(DUMP) if f.endswith(".pt"))
    lats = [torch.load(f"{DUMP}/{f}", map_location="cpu", weights_only=False)["latent"]
            .to("cuda:0", torch.bfloat16) for f in files]
    print(f"{len(lats)} chunks, latent {tuple(lats[0].shape)}\n")

    made = {}

    vae = LightVAE(vae_path=WAN, dtype=torch.bfloat16, device=0,
                   use_lightvae=False, parallel=False)
    vae.model.eval()
    made["wan"] = run("wan", lambda l: vae.decode(l), lats)
    del vae
    torch.cuda.empty_cache()

    try:
        lv = LightVAE(vae_path=f"{LX}/lightvaew2_1.pth", dtype=torch.bfloat16,
                      device=0, use_lightvae=True, parallel=False)
        lv.model.eval()
        made["lightvae"] = run("lightvae", lambda l: lv.decode(l), lats)
        del lv
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  lightvae FAILED: {type(e).__name__}: {e}")

    # need_scaled differs per checkpoint -- getting it wrong looks like a bad model
    for tag, ck, ns in (("taew2_1", "taew2_1.pth", False),
                        ("lighttaew2_1", "lighttaew2_1.pth", True)):
        try:
            m = WanVAE_tiny(vae_path=f"{LX}/{ck}", dtype=torch.bfloat16,
                            device="cuda:0", need_scaled=ns).to("cuda:0").eval()
            made[tag] = run(tag, lambda l, _m=m: _m.decode(l), lats)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {tag} FAILED: {type(e).__name__}: {e}")

    from ultra_dec_v3 import UltraDecoderV3
    v3 = UltraDecoderV3(V3).to("cuda:0", torch.float16).eval().requires_grad_(False)
    made["v3"] = run("v3", lambda l: v3.decode_latents(l.unsqueeze(0)), lats)
    del v3
    torch.cuda.empty_cache()

    # side-by-side: reference first, then the two serious candidates
    order = [t for t in ("wan", "taew2_1", "lightvae") if t in made]
    if len(order) == 3:
        sbs = f"{OUT}/wave_SIDEBYSIDE_wan_tae_lightvae.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for t in order:
            cmd += ["-i", made[t]]
        cmd += ["-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[v]",
                "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-shortest", sbs]
        subprocess.run(cmd, check=True)
        print(f"\n  side-by-side (L->R: {' | '.join(order)}) -> {sbs}")


if __name__ == "__main__":
    main()
