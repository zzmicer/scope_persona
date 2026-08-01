"""Generalize --fast_decode from the reconstructed v3 to any tiny decoder,
defaulting to taew2_1.

Measured on real dumped latents (MAE vs the full Wan2.1 VAE):
    taew2_1       32ms  MAE 2.77   official, 23MB   <- new default
    lighttaew2_1  32ms  MAE 3.76   official, 45MB
    v3 (ours)     31ms  MAE 6.41   RECONSTRUCTED    <- was the default
taew2_1 wins on quality at the same speed AND is an officially released
checkpoint, so the rebuilt-v3 liability disappears.

All three emit n_lat*4-3 frames for n_lat latent frames, matching the Wan VAE, so
the existing windowed decode and the 21-then-32 frame counts A/V sync relies on
stay correct whichever is selected.

GOTCHA baked in below: the two lightx2v TAEs use OPPOSITE latent normalization.
taew2_1 needs need_scaled=False (True -> MAE 24.48); lighttaew2_1 needs
need_scaled=True (False -> MAE 21.56). Wrong flag looks like a broken model.
"""

import shutil
import sys

TARGETS = [
    "/workspace/soulx/SoulX-LiveAct/pruna_bench.py",
    "/workspace/soulx/SoulX-LiveAct/interactive_demo.py",
]

OLD = '''        if getattr(args, "fast_decode", False):
            import sys as _sys

            _sys.path.insert(0, "/workspace/uflash")
            _sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")
            from ultra_dec_v3 import UltraDecoderV3 as _V3

            _dec = (
                _V3(args.fast_decode_ckpt)
                .to(f"cuda:{self.device}", torch.float16)
                .eval()
                .requires_grad_(False)
            )
            self.fast_dec = _dec

            def _fast_decode(latent, _d=_dec):
                # (C,T,h,w) -> (1,3,T*4-3,h*8,w*8), same contract as WanVAE.decode
                return _d.decode_latents(latent.unsqueeze(0))

            self.vae.decode = _fast_decode
            if self.rank == 0:
                print("[fastdec] v3 tiny decoder active", flush=True)'''

NEW = '''        if getattr(args, "fast_decode", False):
            import sys as _sys

            _sys.path.insert(0, "/workspace/uflash")
            _sys.path.insert(0, "/workspace/uflash/UltraFlash/inference")

            _which = getattr(args, "tiny_decoder", "taew2_1")
            _lx = "/workspace/uflash/ckpt_lx2v"
            _dev = f"cuda:{self.device}"

            if _which == "v3":
                # Reconstructed from the checkpoint's shape signature; kept only
                # for comparison. MAE 6.41 vs taew2_1's 2.77.
                from ultra_dec_v3 import UltraDecoderV3 as _V3

                _d = (_V3(args.fast_decode_ckpt).to(_dev, torch.float16)
                      .eval().requires_grad_(False))

                def _fast_decode(latent, _m=_d):
                    return _m.decode_latents(latent.unsqueeze(0))

            else:
                # need_scaled is per-checkpoint, NOT a preference -- see module docs.
                _NEEDS_SCALE = {"taew2_1": False, "lighttaew2_1": True}
                if _which not in _NEEDS_SCALE:
                    raise ValueError(
                        f"--tiny_decoder must be one of "
                        f"{['v3'] + list(_NEEDS_SCALE)} (got {_which})"
                    )
                from lightx2v.models.video_encoders.hf.wan.vae_tiny import (
                    WanVAE_tiny as _Tiny,
                )

                _d = _Tiny(
                    vae_path=f"{_lx}/{_which}.pth",
                    dtype=torch.bfloat16,
                    device=_dev,
                    need_scaled=_NEEDS_SCALE[_which],
                ).to(_dev).eval()

                def _fast_decode(latent, _m=_d):
                    # (C,T,h,w) -> (1,3,T*4-3,h*8,w*8), as WanVAE.decode returns
                    return _m.decode(latent)

            self.fast_dec = _d
            self.vae.decode = _fast_decode
            if self.rank == 0:
                print(f"[fastdec] tiny decoder active: {_which}", flush=True)'''

OLD_ARG = '''    parser.add_argument("--fast_decode_ckpt", type=str,'''

NEW_ARG = '''    parser.add_argument("--tiny_decoder", type=str, default="taew2_1",
                        choices=["taew2_1", "lighttaew2_1", "v3"],
                        help="which tiny decoder --fast_decode uses "
                             "(taew2_1: official, MAE 2.77; v3: reconstructed, 6.41)")
    parser.add_argument("--fast_decode_ckpt", type=str,'''


def main():
    rc = 0
    for path in TARGETS:
        try:
            src = open(path).read()
        except FileNotFoundError:
            print(f"skip (missing): {path}")
            continue

        if "--tiny_decoder" in src:
            print(f"already patched: {path}")
            continue
        if src.count(OLD) != 1 or src.count(OLD_ARG) != 1:
            print(f"ANCHOR MISMATCH ({src.count(OLD)} body, {src.count(OLD_ARG)} arg): {path}")
            rc = 2
            continue

        shutil.copy2(path, path + ".bak_tinydec")
        open(path, "w").write(src.replace(OLD, NEW).replace(OLD_ARG, NEW_ARG))
        print(f"patched {path} (backup {path}.bak_tinydec)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
