"""Add FlashAttention-3 as a selectable backend for SoulX's streaming self-attention.

Today the single-GPU path runs PyTorch SDPA: SageAttention is import-blocked (its
sm90 kernels produce noise/NaN here) and the flash_attention calls were commented
out. This wires FA3 in behind SOULX_ATTN=fa3.

USE XFORMERS' BUNDLED FA3, NOT a separately built flash_attn_3 wheel.
xformers 0.0.32 already ships `xformers/flash_attn_3/_C.so` (fa3F@2.8.0.post2).
Loading a second flash_attn_3 extension alongside it aborts the process at import:
both register the same torch library namespace and c10::Dispatcher::registerLibrary
fails. That is almost certainly what failed the earlier local-wheel "gate" -- the
kernel was fine, the environment had two of them.

Gate results on this pod, real SoulX shapes (40 heads x 128 dim, bf16):
    max|delta| vs SDPA  0.0002-0.0005          -> numerically equivalent
    self-attn per call  592x336  0.823 -> 0.358 ms  (2.30x)
                        368x640  0.970 -> 0.435 ms  (2.23x)
                        720x416  1.519 -> 0.730 ms  (2.08x)
    over 40 blocks x 3 steps that is ~56 / ~64 / ~95 ms off a chunk.

SCOPE: self-attention only -- that is where the streaming KV window makes tensors
big. Cross-attention runs over a ~29-token text context; not worth the risk.

NOTE self.window_size is (-1, -1) (unbounded) and self.attn_mask is None in this
config, so dropping them on the FA3 path is safe HERE. A config that sets a real
window MUST pass it through.
"""

import shutil
import sys

SRC = "/workspace/soulx/SoulX-LiveAct/model_liveact/model_memory.py"
BAK = SRC + ".bak_fa3"

OLD_IMPORT = '''try:
    from sageattention import sageattn

    USE_SAGEATTN = True
    logging.info("Using sageattn")
except:
    USE_SAGEATTN = False'''

NEW_IMPORT = '''try:
    from sageattention import sageattn

    USE_SAGEATTN = True
    logging.info("Using sageattn")
except:
    USE_SAGEATTN = False

# SOULX_ATTN=fa3 opts into FlashAttention-3 for the streaming self-attention.
# Default stays sdpa: that is the path every other measurement was taken on.
# Deliberately xformers' bundled FA3 -- a second flash_attn_3 extension in the
# same process aborts on duplicate torch-library registration.
USE_FA3 = False
_fmha = None
_fa3_op = None
if os.environ.get("SOULX_ATTN", "sdpa").lower() == "fa3":
    try:
        from xformers.ops import fmha as _fmha_mod

        _op = _fmha_mod.flash3.FwOp
        if not _op.is_available():
            raise RuntimeError("xformers flash3 reports unavailable on this build")
        _fmha, _fa3_op = _fmha_mod, _op
        USE_FA3 = True
        print(f"[attn] FlashAttention-3 enabled for self-attention ({_op.NAME})",
              flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[attn] SOULX_ATTN=fa3 unavailable ({type(_e).__name__}: {_e}); "
              f"falling back to SDPA", flush=True)


def _fa3_attn(q, k, v):
    """(B, N, H, D) in and out -- the NHD layout this call site already uses."""
    return _fmha.memory_efficient_attention_forward(q, k, v, op=_fa3_op)'''

OLD_ATTN = '''        if USE_SAGEATTN:
            x = sageattn(
                roped_query,
                roped_key[:, :end_idx, ...],
                v_cache[:, :end_idx, ...],
                tensor_layout="NHD",
                is_causal=False,
            ).type_as(x)
        else:
            x = sdpa_attention(
                q=roped_query,
                k=roped_key[:, :end_idx, ...],
                v=v_cache[:, :end_idx, ...],
                k_lens=seq_lens,
                window_size=self.window_size,
                attn_mask=self.attn_mask,
            ).type_as(x)'''

NEW_ATTN = '''        if USE_FA3:
            x = _fa3_attn(
                roped_query,
                roped_key[:, :end_idx, ...],
                v_cache[:, :end_idx, ...],
            ).type_as(x)
        elif USE_SAGEATTN:
            x = sageattn(
                roped_query,
                roped_key[:, :end_idx, ...],
                v_cache[:, :end_idx, ...],
                tensor_layout="NHD",
                is_causal=False,
            ).type_as(x)
        else:
            x = sdpa_attention(
                q=roped_query,
                k=roped_key[:, :end_idx, ...],
                v=v_cache[:, :end_idx, ...],
                k_lens=seq_lens,
                window_size=self.window_size,
                attn_mask=self.attn_mask,
            ).type_as(x)'''


def main():
    import os

    # Start from the pristine file if a previous attempt patched this.
    if os.path.exists(BAK):
        shutil.copy2(BAK, SRC)
        print(f"restored {SRC} from {BAK}")

    src = open(SRC).read()
    for anchor in (OLD_IMPORT, OLD_ATTN):
        if src.count(anchor) != 1:
            print(f"anchor not unique ({src.count(anchor)}x):\n{anchor[:60]}...")
            return 2
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
    open(SRC, "w").write(src.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_ATTN, NEW_ATTN))
    print(f"patched {SRC} (backup {BAK})")
    print("  enable with SOULX_ATTN=fa3; default sdpa unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
