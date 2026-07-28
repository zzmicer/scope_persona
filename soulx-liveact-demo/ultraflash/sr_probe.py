"""
Offline probe: SoulX-LiveAct LR latents -> UltraFlash SR cascade.

Both models live in the Wan2.1 VAE latent space (16ch, stride 4/8/8), so the
handoff needs no pixel round-trip. This script mirrors what
UltraFlash's CausalCascadeStreamingPipeline._flush_cascade does, minus their
generator: latents come from a real SoulX session instead.

Decode is deliberately done with SoulX's own Wan2.1 VAE rather than UltraFlash's
tiny decoder, so a quality regression can be blamed on the SR DiT and not on a
decoder we have not validated. --decoder ultra switches to their fast decoder
once the SR output itself is trusted.
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

UF = "/workspace/uflash/UltraFlash/inference"
sys.path.insert(0, UF)

# Architecture params are copied from configs/sr_sparse_ultraforcing.py. They are
# not importable without dragging in the whole ExpConfig/training stack.
DIT_PARAMS = dict(
    model_type="t2v",
    patch_size=[1, 2, 2],
    text_len=512,
    in_dim=32,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12,
    num_layers=30,
    window_size=[-1, -1],
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    use_sparse_attn=True,
    sparse_causal=True,
    sparse_block_size=[2, 8, 8],
    sparse_top_k=None,
    sparse_top_k_ratio=1.0,
    sparse_kv_ratio=3.0,
    sparse_local_range=9,
    sparse_local_num=None,
    sparse_use_kernel=True,
    stream_chunk_size=2,
    rope_max_seq_len=1024,
    rope_theta=10000.0,
    rope_cache_multiple=1024,
    sparse_ref_spatial_tokens=1560,
)

UPSAMPLER_PARAMS = dict(
    in_channels=16,
    out_channels=16,
    mid_channels=128,
    num_blocks=8,
    activation="relu",
    residual=False,
    residual_scale=1.0,
    memory_init="replicate",
    zero_init_final=True,
    default_parallel=False,
)


def load_upsampler(ckpt, scale, device, dtype):
    from sr.latent_upsampler.ultra_latent_up_v1 import UltraLatentUpsampler

    params = dict(UPSAMPLER_PARAMS)
    params["upsample_scale"] = scale
    m = UltraLatentUpsampler(**params).to(device=device, dtype=dtype)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[upsampler] scale={scale} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("           missing e.g.:", missing[:4])
    if unexpected:
        print("           unexpected e.g.:", unexpected[:4])
    return m.eval().requires_grad_(False)


def load_sr_dit(ckpt, device, dtype):
    from sr.dit_sparse import Transformer3DModel

    m = Transformer3DModel(**DIT_PARAMS)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    for k in ("model", "generator"):
        if k in sd:
            sd = sd[k]
            break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[sr_dit] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("        missing e.g.:", missing[:6])
    if unexpected:
        print("        unexpected e.g.:", unexpected[:6])
    return m.to(device=device, dtype=dtype).eval().requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", default="/workspace/uflash/dump")
    ap.add_argument("--out_dir", default="/workspace/uflash/out")
    ap.add_argument("--ckpt_dir", default="/workspace/uflash/ckpt")
    ap.add_argument("--scale", type=int, default=2, choices=[2, 3])
    ap.add_argument("--sr_kv_len", type=int, default=3)
    ap.add_argument("--sr_timestep", type=float, default=1000.0)
    ap.add_argument("--cond_noise", type=float, default=0.2)
    ap.add_argument("--max_chunks", type=int, default=8)
    ap.add_argument("--decoder", default="wan", choices=["wan", "ultra", "v3", "none"])
    ap.add_argument("--vae_path", default="/workspace/soulx/weights/LiveAct/Wan2.1_VAE.pth")
    ap.add_argument("--ultra_ckpt",
                    default="/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth")
    ap.add_argument("--no_kernel", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    if args.no_kernel:
        DIT_PARAMS["sparse_use_kernel"] = False

    sr_ck = f"{args.ckpt_dir}/{'1K' if args.scale == 2 else '2K'}-Causal-Sparse-SR-Dit-fidelity-step_12800.pth"
    up_ck = (
        f"{args.ckpt_dir}/1K-ultraLatentUpsampler-step_19900.pth"
        if args.scale == 2
        else f"{args.ckpt_dir}/2K-ultraLatentUpsampler-step_28100.pth"
    )

    t0 = time.time()
    upsampler = load_upsampler(up_ck, args.scale, device, dtype)
    sr_dit = load_sr_dit(sr_ck, device, dtype)
    print(f"[load] {time.time() - t0:.1f}s")

    try:
        from block_sparse_attn import block_sparse_attn_func  # noqa: F401

        print("[kernel] block_sparse_attn AVAILABLE")
    except Exception:
        print("[kernel] block_sparse_attn MISSING -> dense pytorch fallback (timings pessimistic)")

    vae = None
    ultra = None
    if args.decoder == "wan":
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE

        vae = LightVAE(vae_path=args.vae_path, dtype=torch.bfloat16, device=0,
                       use_lightvae=False, parallel=False)
        vae.model.eval()
    elif args.decoder == "v3":
        sys.path.insert(0, "/workspace/uflash")
        from ultra_dec_v3 import UltraDecoderV3

        ultra = (UltraDecoderV3(args.ultra_ckpt)
                 .to(device, torch.float16).eval().requires_grad_(False))
        print(f"[v3 decoder] {args.ultra_ckpt}")
    elif args.decoder == "ultra":
        from sr.ultra_decoder import WanUltraDecoder

        ultra = WanUltraDecoder(checkpoint_path=args.ultra_ckpt,
                                dtype=torch.float16).to(device).eval()
        print(f"[ultra decoder] {args.ultra_ckpt}")

    files = sorted(glob.glob(os.path.join(args.dump_dir, "chunk_*.pt")))[: args.max_chunks]
    if not files:
        print(f"no dumps in {args.dump_dir}")
        return
    print(f"[dumps] {len(files)} chunks")

    from sr.stream_forward import _stream_forward_causal

    gen = torch.Generator(device=device).manual_seed(args.seed)
    up_caches = None
    sr_cache = None
    offset = 0
    patch_t = DIT_PARAMS["patch_size"][0]
    stats = []

    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        lr = d["latent"].to(device=device, dtype=dtype)          # (C,T,H,W)
        ctx = [c.to(device=device, dtype=dtype) for c in d["ctx"]]
        lr_b = lr.unsqueeze(0)                                    # (1,C,T,H,W)

        torch.cuda.synchronize()
        ta = time.time()
        with torch.no_grad():
            hr_cond, up_caches = upsampler(
                lr_b, parallel=False, caches=up_caches,
                return_caches=True, detach_caches=True,
            )
        torch.cuda.synchronize()
        t_up = time.time() - ta

        s = args.cond_noise
        if s > 0:
            noise = torch.randn(hr_cond.shape, device=device, dtype=hr_cond.dtype, generator=gen)
            cond_y = (1.0 - s) * hr_cond + s * noise
        else:
            cond_y = hr_cond

        cond_y = cond_y.to(dtype=dtype)
        lat = torch.randn(cond_y.shape, device=device, dtype=dtype, generator=gen)
        tval = torch.full((1,), args.sr_timestep, device=device)

        if i == 0 and hasattr(sr_dit, "clear_cross_kv"):
            sr_dit.clear_cross_kv()

        torch.cuda.synchronize()
        tb = time.time()
        with torch.no_grad():
            out, sr_cache = _stream_forward_causal(
                model=sr_dit,
                x_list=[lat[0]],
                t=tval,
                context=ctx,
                y_list=[cond_y[0]],
                temporal_offset=offset // patch_t,
                kv_len=args.sr_kv_len,
                cache_state=sr_cache,
            )
        torch.cuda.synchronize()
        t_dit = time.time() - tb

        clean = lat - torch.stack(out, dim=0)
        offset += lr.shape[1]
        if i == 1:  # stash one steady-state HR latent for decoder-norm diagnosis
            torch.save(clean.detach().cpu(), "/workspace/uflash/hr_latent.pt")

        t_dec = 0.0
        vids = None
        if vae is not None:
            torch.cuda.synchronize()
            tc = time.time()
            with torch.no_grad():
                vids = vae.decode(clean[0].to(torch.bfloat16))
            torch.cuda.synchronize()
            t_dec = time.time() - tc
        elif ultra is not None:
            torch.cuda.synchronize()
            tc = time.time()
            with torch.no_grad():
                vids = ultra.decode_latents(clean)
            torch.cuda.synchronize()
            t_dec = time.time() - tc
        if vids is not None:
            if i < 4:
                import imageio
                import numpy as np

                u8 = (((vids.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) * 127.5)
                      .clamp(0, 255).to(torch.uint8).cpu().numpy())
                imageio.imwrite(f"{args.out_dir}/sr_c{i:02d}_f00.png", u8[0])
                imageio.imwrite(f"{args.out_dir}/sr_c{i:02d}_last.png", u8[-1])

        n_lat = lr.shape[1]
        n_pix = n_lat * 4
        tot = t_up + t_dit + t_dec
        print(
            f"[chunk {i}] lat{tuple(lr.shape)} -> hr{tuple(clean.shape)} | "
            f"up {t_up*1000:.0f}ms  dit {t_dit*1000:.0f}ms  dec {t_dec*1000:.0f}ms | "
            f"total {tot:.2f}s for {n_pix} frames -> {n_pix/tot:.1f} pix fps",
            flush=True,
        )
        stats.append(dict(chunk=i, t_up=t_up, t_dit=t_dit, t_dec=t_dec,
                          n_lat=n_lat, n_pix=n_pix,
                          lr_shape=list(lr.shape), hr_shape=list(clean.shape)))

    # chunk 0 is cold (allocator warmup, cache init) -> report steady state too
    warm = stats[1:] or stats
    m_up = sum(s["t_up"] for s in warm) / len(warm)
    m_dit = sum(s["t_dit"] for s in warm) / len(warm)
    m_dec = sum(s["t_dec"] for s in warm) / len(warm)
    m_pix = sum(s["n_pix"] for s in warm) / len(warm)
    print("\n=== steady state (excl. chunk 0) ===")
    print(f"upsample {m_up*1000:7.1f} ms")
    print(f"sr_dit   {m_dit*1000:7.1f} ms")
    print(f"decode   {m_dec*1000:7.1f} ms")
    print(f"TOTAL    {(m_up+m_dit+m_dec)*1000:7.1f} ms per {m_pix:.0f}-frame chunk")
    print(f"peak vram {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    with open(f"{args.out_dir}/stats.json", "w") as fh:
        json.dump(dict(args=vars(args), stats=stats), fh, indent=1)


if __name__ == "__main__":
    main()
