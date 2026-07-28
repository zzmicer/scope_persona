"""
UltraFlash SR sidecar for the SoulX-LiveAct demo.

Runs as its OWN PROCESS on its OWN GPU, for the same reason persona_aux does: a
second CUDA context inside a torchrun rank destabilizes NCCL. The demo's rank 0
POSTs a chunk latent, this returns the upscaled frames.

Pipeline per chunk (all in Wan2.1 latent space, no pixel round-trip):
    LR latent -> causal latent upsampler (streaming caches)
              -> sparse causal SR DiT   (streaming KV cache)
              -> v3 tiny decoder        -> HR uint8 frames

Both caches persist across chunks for the life of a session; /reset clears them
when the demo starts a new session, which is what keeps the stream temporally
coherent rather than flickering per chunk.

KNOWN ARTIFACT: the visible texture artifacts in the stream originate HERE, not
in the generator. The SR DiT was trained with AIGC-oriented degradation on
photoreal Wan output, so against flat cel-shaded anime it invents pore/hair-grade
high-frequency detail where none exists -- a repeating cross-hatch weave on
ambiguous regions (the hand under the chin, the hair/cheek boundary). Identity,
expression and colour survive intact; it is the fabricated micro-texture that
reads as wrong. Dropping --sr_url removes it. Experimental; expected to revert.
"""

import argparse
import io
import os
import sys
import threading
import time

import numpy as np
import torch
from flask import Flask, jsonify, request

sys.path.insert(0, "/workspace/uflash")
sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

from sr_probe import DIT_PARAMS, UPSAMPLER_PARAMS, load_sr_dit, load_upsampler  # noqa: E402
from ultra_dec_v3 import UltraDecoderV3  # noqa: E402

app = Flask(__name__)
ENG = None


class SREngine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES pins the physical card
        self.dtype = torch.bfloat16
        self.lock = threading.Lock()

        sr_ck = os.path.join(
            args.ckpt_dir,
            f"{'1K' if args.scale == 2 else '2K'}-Causal-Sparse-SR-Dit-fidelity-step_12800.pth",
        )
        up_ck = os.path.join(
            args.ckpt_dir,
            "1K-ultraLatentUpsampler-step_19900.pth" if args.scale == 2
            else "2K-ultraLatentUpsampler-step_28100.pth",
        )
        t0 = time.time()
        self.upsampler = load_upsampler(up_ck, args.scale, self.device, self.dtype)
        self.sr_dit = load_sr_dit(sr_ck, self.device, self.dtype)
        self.dec = (UltraDecoderV3(args.dec_ckpt)
                    .to(self.device, torch.float16).eval().requires_grad_(False))
        print(f"[sr] models loaded in {time.time() - t0:.1f}s", flush=True)

        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)
        self.reset()

    def reset(self):
        self.up_caches = None
        self.sr_cache = None
        self.pre_hr = None
        self.offset = 0
        self.n = 0
        self.first = True
        print("[sr] caches reset", flush=True)

    @torch.no_grad()
    def run(self, lat_cthw, ctx_list):
        from sr.stream_forward import _stream_forward_causal

        t_all = time.time()
        lr = lat_cthw.to(self.device, self.dtype).unsqueeze(0)

        t0 = time.time()
        hr_cond, self.up_caches = self.upsampler(
            lr, parallel=False, caches=self.up_caches,
            return_caches=True, detach_caches=True,
        )
        t_up = time.time() - t0

        s = self.args.cond_noise
        if s > 0:
            noise = torch.randn(hr_cond.shape, device=self.device,
                                dtype=hr_cond.dtype, generator=self.gen)
            cond_y = (1.0 - s) * hr_cond + s * noise
        else:
            cond_y = hr_cond
        cond_y = cond_y.to(self.dtype)

        lat = torch.randn(cond_y.shape, device=self.device, dtype=self.dtype,
                          generator=self.gen)
        tval = torch.full((1,), self.args.sr_timestep, device=self.device)
        if self.first and hasattr(self.sr_dit, "clear_cross_kv"):
            self.sr_dit.clear_cross_kv()
            self.first = False

        ctx = [c.to(self.device, self.dtype) for c in ctx_list]
        t0 = time.time()
        out, self.sr_cache = _stream_forward_causal(
            model=self.sr_dit, x_list=[lat[0]], t=tval, context=ctx,
            y_list=[cond_y[0]],
            temporal_offset=self.offset // DIT_PARAMS["patch_size"][0],
            kv_len=self.args.sr_kv_len, cache_state=self.sr_cache,
        )
        t_dit = time.time() - t0
        clean = lat - torch.stack(out, dim=0)
        self.offset += lr.shape[2]

        # The tiny decoder's mem-blocks only carry state WITHIN one call, so decode
        # each chunk with the previous 3 HR latent frames prepended and drop the
        # lead-in. This mirrors what SoulX does with the Wan VAE and, critically,
        # reproduces its exact frame counts (21 for chunk 0, then 32) -- the demo
        # drives A/V sync off frames_emitted, so a different count desyncs audio.
        t0 = time.time()
        n_lat = clean.shape[2]
        if self.pre_hr is None:
            vids = self.dec.decode_latents(clean)            # n_lat*4 - 3 frames
        else:
            ctx_in = torch.cat([self.pre_hr[:, :, -3:], clean], dim=2)
            vids = self.dec.decode_latents(ctx_in)[:, :, -(n_lat * 4):]
        self.pre_hr = clean
        u8 = (((vids.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) * 127.5)
              .clamp(0, 255).to(torch.uint8).contiguous().cpu().numpy())
        t_dec = time.time() - t0

        self.n += 1
        if self.n % 10 == 1:
            print(f"[sr] chunk {self.n}: {u8.shape[0]}f {u8.shape[2]}x{u8.shape[1]} | "
                  f"up {t_up*1000:.0f}ms dit {t_dit*1000:.0f}ms dec {t_dec*1000:.0f}ms | "
                  f"total {(time.time()-t_all)*1000:.0f}ms", flush=True)
        return u8


@app.route("/health")
def health():
    return jsonify({"ok": ENG is not None, "chunks": ENG.n if ENG else 0})


@app.route("/reset", methods=["POST"])
def reset():
    with ENG.lock:
        ENG.reset()
    return jsonify({"ok": True})


@app.route("/sr", methods=["POST"])
def sr():
    """Body: torch.save'd {latent: (C,T,H,W) bf16 cpu, ctx: [ (L,4096) bf16 cpu ]}.
    Returns raw uint8 frames + shape header, which is cheaper than re-encoding
    to PNG only for the caller to decode again."""
    buf = io.BytesIO(request.get_data())
    d = torch.load(buf, map_location="cpu")
    with ENG.lock:
        u8 = ENG.run(d["latent"], d["ctx"])
    hdr = f"{u8.shape[0]},{u8.shape[1]},{u8.shape[2]},{u8.shape[3]}".encode()
    return hdr + b"\n" + u8.tobytes(), 200, {"Content-Type": "application/octet-stream"}


def main():
    global ENG
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--ckpt_dir", default="/workspace/uflash/ckpt")
    ap.add_argument("--dec_ckpt",
                    default="/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth")
    ap.add_argument("--scale", type=int, default=2, choices=[2, 3])
    ap.add_argument("--sr_kv_len", type=int, default=3)
    ap.add_argument("--sr_timestep", type=float, default=1000.0)
    ap.add_argument("--cond_noise", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ENG = SREngine(args)
    print(f"[sr] serving on http://127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
