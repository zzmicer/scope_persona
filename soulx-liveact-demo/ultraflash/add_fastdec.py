"""Patch interactive_demo.py with --fast_decode: v3 tiny decoder in place of the
Wan2.1 VAE decode, inline (no sidecar, no HTTP).

Measured on this pod at native 720x416: 31.0ms vs 452.3ms, MAE 6.41/255 against
the Wan VAE on real dumped SoulX latents. That is 421ms/chunk off the critical
path, which is what makes a single-GPU deployment reachable.

The swap is a one-liner because the frame arithmetic already matches: the v3 net
has frames_to_trim=3, so n_lat latent frames -> n_lat*4-3 frames, exactly what
the Wan VAE emits. Both existing call sites (warmup and the chunk loop) keep
their windowing -- prepend 3 latent frames, drop the 9-frame lead-in -- and keep
SoulX's 21-then-32 frame counts, which A/V sync is driven from.

Reversible: writes interactive_demo.py.bak4. Without --fast_decode nothing changes.
"""

import re
import shutil
import sys

SRC = "/workspace/soulx/SoulX-LiveAct/interactive_demo.py"
BAK = SRC + ".bak4"

ARGS_ANCHOR = """    parser.add_argument("--sr_url", type=str, default=None,
                        help="UltraFlash SR sidecar base URL; omit to stream at native res")"""

ARGS_NEW = """    parser.add_argument("--fast_decode", action="store_true",
                        help="decode with the v3 tiny decoder instead of the Wan2.1 VAE "
                             "(~31ms vs ~452ms at 720x416, MAE 6.4/255)")
    parser.add_argument("--fast_decode_ckpt", type=str,
                        default="/workspace/uflash/ckpt/v1.1-ultra-decoder-v3-ema_decoder.pth")
""" + ARGS_ANCHOR

INIT_ANCHOR = """        self.vae.model.eval()
        if not args.no_compile:
            self.vae.decode = torch.compile(self.vae.decode)"""

INIT_NEW = INIT_ANCHOR + """

        # Swap in the tiny decoder AFTER any torch.compile of the real one.
        # self.vae.encode is untouched -- the reference image still goes through
        # the full Wan VAE, only the per-chunk decode is replaced.
        if getattr(args, "fast_decode", False):
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
                print("[fastdec] v3 tiny decoder active", flush=True)"""


def main():
    src = open(SRC).read()

    if "fast_decode" in src:
        print("already patched; nothing to do")
        return 1

    for anchor in (ARGS_ANCHOR, INIT_ANCHOR):
        if src.count(anchor) != 1:
            print(f"anchor not unique ({src.count(anchor)}x), aborting:\n{anchor[:70]}...")
            return 2

    shutil.copy2(SRC, BAK)
    src = src.replace(ARGS_ANCHOR, ARGS_NEW).replace(INIT_ANCHOR, INIT_NEW)
    open(SRC, "w").write(src)

    print(f"patched {SRC} (backup {BAK})")
    print("  + --fast_decode / --fast_decode_ckpt")
    print("  + v3 decoder swapped onto self.vae.decode after the compile block")
    return 0


if __name__ == "__main__":
    sys.exit(main())
