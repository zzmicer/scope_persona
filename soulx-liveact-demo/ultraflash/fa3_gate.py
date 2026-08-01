"""Execution gate for the locally built FA3 wheel, on SoulX's actual attention shapes.

Import alone proves nothing -- the prior attempt is recorded as failing an
"import/execution gate", and the import failure we reproduced turned out to be a
polluted --target install (a second torch), not the kernel. So: run it, check the
numbers against SDPA, and time it.

Shapes come from the live config: dim 5120 / 40 heads -> head_dim 128, bf16.
Query is one chunk of latent frames; KV is the whole streaming window.
  592x336 -> frame_len (336/16)*(592/16) = 21*37 = 777 tokens/latent frame
  368x640 -> (640/16)*(368/16)          = 40*23 = 920
KV window is frame_len * sum(blksz_lst=[6,8]) = frame_len * 14.
"""

import time

import torch
import torch.nn.functional as F

HEADS, HDIM = 40, 128
CASES = [
    ("592x336", 777, 777 * 14),
    ("368x640", 920, 920 * 14),
    ("720x416", 1170, 1170 * 14),
]


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.time() - t0) / n


def main():
    import flash_attn_interface as fa3

    print(f"{'case':10s} {'q':>6s} {'kv':>7s} {'sdpa ms':>9s} {'fa3 ms':>8s} "
          f"{'speedup':>8s} {'max|Δ|':>9s}  status")
    for tag, qlen, kvlen in CASES:
        q = torch.randn(1, qlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, kvlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, kvlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)

        # SDPA wants (B, H, S, D)
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        ref, t_sdpa = bench(lambda: F.scaled_dot_product_attention(qs, ks, vs))
        ref = ref.transpose(1, 2)

        try:
            out, t_fa3 = bench(lambda: fa3.flash_attn_func(q, k, v)[0]
                               if isinstance(fa3.flash_attn_func(q, k, v), tuple)
                               else fa3.flash_attn_func(q, k, v))
            d = (out.float() - ref.float()).abs().max().item()
            ok = "OK" if d < 0.05 else f"MISMATCH (>{0.05})"
            print(f"{tag:10s} {qlen:6d} {kvlen:7d} {t_sdpa*1000:9.3f} {t_fa3*1000:8.3f} "
                  f"{t_sdpa/t_fa3:7.2f}x {d:9.4f}  {ok}")
        except Exception as e:
            print(f"{tag:10s} {qlen:6d} {kvlen:7d} {t_sdpa*1000:9.3f} {'FAIL':>8s} "
                  f"{'':>8s} {'':>9s}  {type(e).__name__}: {str(e)[:60]}")

    # whole-model impact: 40 blocks x 3 denoising steps of self-attention
    print("\nper-chunk self-attention total (40 blocks x 3 steps):")
    for tag, qlen, kvlen in CASES:
        q = torch.randn(1, qlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, kvlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, kvlen, HEADS, HDIM, device="cuda", dtype=torch.bfloat16)
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        _, t_sdpa = bench(lambda: F.scaled_dot_product_attention(qs, ks, vs))
        try:
            r = fa3.flash_attn_func(q, k, v)
            _, t_fa3 = bench(lambda: fa3.flash_attn_func(q, k, v))
            n = 40 * 3
            print(f"  {tag:10s} sdpa {t_sdpa*n*1000:7.1f} ms   fa3 {t_fa3*n*1000:7.1f} ms   "
                  f"saves {(t_sdpa-t_fa3)*n*1000:6.1f} ms/chunk")
        except Exception as e:
            print(f"  {tag:10s} fa3 unavailable: {type(e).__name__}")


if __name__ == "__main__":
    main()
