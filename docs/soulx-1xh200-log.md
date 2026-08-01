# SoulX-LiveAct: 4 GPUs → 1, session log (2026-08-01)

Goal: the demo needed 4×H200 and that is too much for a demo. Target one H200.

Result: **3.64s → 1.96s per chunk at 720×416 (1.86× faster), and realtime reached
on a single H200**, from three independent wins that cost no image quality —
an official tiny decoder, block-level `torch.compile`, and FlashAttention-3.

Everything below was measured on the pod. A chunk is 32 frames = 2.0s of video at
16fps, so the budget is **2.0s/chunk**, and comfortably under: a wall-clock pacer
keeps ~1.6s of lead and any overrun surfaces as a micro-freeze.

## Final configuration

`soulx-liveact-demo/deploy_demo.sh` — one H200, `persona_aux` co-located
(`world_size==1` → no NCCL, so the old process-split rule does not apply).

| component | setting |
|---|---|
| decoder | `taew2_1` tiny decoder (official lightx2v, 23MB) |
| compile | `--compile_blocks`, `max-autotune-no-cudagraphs` |
| attention | FlashAttention-3 via xformers (`SOULX_ATTN=fa3`) |
| quantization | FP8 W8A8, `--fp8_scope blocks` |

## Latency, 1×H200

| size | px | wanVAE | +taew2_1 | +compile | +FA3 |
|---|---|---|---|---|---|
| 720×416 | 299,520 | 3.64 (0.55×) | 3.01 (0.66×) | 2.50 (0.80×) | **1.97 (1.02×)** |
| 416×720 | 299,520 | — | — | — | **1.96 (1.02×)** |
| 640×368 | 235,520 | — | 2.18 (0.92×) | — | — |
| 368×640 | 235,520 | — | — | 1.80 (1.11×) | **1.45 (1.38×)** |
| 592×336 | 198,912 | 2.19 (0.91×) | 1.78 (1.12×) | 1.42 (1.41×) | — |
| 560×320 | 179,200 | — | 1.57 (1.27×) | — | — |

Two invariants confirmed repeatedly:

- **Cost tracks pixel count, not orientation.** 336×592 and 592×336 both measured
  1.73s; 416×720 and 720×416 both ~1.96s. Portrait is free.
- **Time scales ~linearly with pixels**, slightly faster than linear (attention is
  superlinear in tokens). Break-even for realtime was ~215k px before FA3.

## What was done, in order

1. **Dropped SR, kept its decoder.** The 4-GPU SR build was *faster* than plain
   SoulX (1.62s at 1440×832 vs 2.00s at 720×416) — not because of the
   super-resolution, but because the sidecar's tiny decoder had displaced SoulX's
   own 452ms Wan VAE decode. So the move was to discard the 5.3GB SR DiT and keep
   the 25MB decoder. Verified at native resolution: 31ms vs 452ms, MAE 6.41/255.
2. **`--fast_decode` inlined** (`ultraflash/add_fastdec.py`). A one-line swap of
   `self.vae.decode`, because the tiny decoder's `frames_to_trim=3` yields
   `n_lat*4-3` frames — identical to the Wan VAE — so the existing windowed decode
   and the 21-then-32 frame counts A/V sync depends on needed no changes.
3. **Single-GPU launcher** + resolution sweep → realtime reached at ≤215k px.
4. **Block compile** (`pruna_bench.py --compile_blocks`): 17–21% everywhere
   measured. Compiling 40 blocks individually is more robust to graph breaks than
   compiling the stateful streaming `WanModel` as one graph.
5. **Decoder shootout vs `lightx2v/Autoencoders`** → replaced the reconstructed v3
   with **taew2_1**: better quality (MAE 2.77 vs 6.41) at the same speed, official
   checkpoint, loads through already-installed code.
6. **FlashAttention-3 enabled** (`ultraflash/add_fa3.py`): 19–21% end-to-end.

## Decoders measured (MAE vs the full Wan2.1 VAE, real dumped latents)

| decoder | ms | MAE/255 | size | notes |
|---|---|---|---|---|
| wan (reference) | 452 | — | 507MB | |
| lightvaew2_1 | 133 | 3.34 | 32MB | 75% pruned, **causal Conv3D** — models time |
| **taew2_1** | 32 | **2.77** | 23MB | in use |
| lighttaew2_1 | 32 | 3.76 | 45MB | |
| v3 | 31 | 6.41 | 25MB | reconstructed; visibly blurry on motion — rejected |

**MAE is a weak signal.** v3 scored a respectable 6.41 and I called it "visually
equivalent"; on a *moving hand* it is visibly blurry, which a per-frame average
cannot see. Judge decoders on moving video —
`ultraflash/dec_video_shootout.py` decodes one set of dumped latents through every
candidate so the decoder is the only variable.

**Gotcha:** the two lightx2v TAEs use OPPOSITE latent normalization. `taew2_1`
needs `need_scaled=False` (True → MAE 24.48); `lighttaew2_1` needs `True`
(False → MAE 21.56). The wrong flag looks like a broken model, not a wrong flag.

## FlashAttention-3: what actually blocked it

Previously recorded as blocked on the Torch 2.8 stack, needing a Torch upgrade
that would risk the working vLLM FP8 path. That was wrong, and the local wheel
built for it was unnecessary.

**xformers 0.0.32 already ships FA3** (`xformers/flash_attn_3/_C.so`,
`fa3F@2.8.0.post2`, `is_available() == True`). Loading a *second* flash_attn_3
extension alongside it aborts the process — both register the same torch library
namespace and `c10::Dispatcher::registerLibrary` fails. That is almost certainly
what failed the original "import/execution gate": the environment had two of them,
the kernel was fine.

Using xformers' own op needs **no Torch upgrade, no wheel, no ABI risk**:

```python
from xformers.ops import fmha
fmha.memory_efficient_attention_forward(q, k, v, op=fmha.flash3.FwOp)
```

Numerically equivalent to SDPA (max|Δ| 2e-4 – 5e-4) on real shapes.

**A microbenchmark lesson.** Timed against bare
`F.scaled_dot_product_attention`, FA3 looked worth ~64ms/chunk (~3.6%), and I
briefly concluded attention was a minor lever. End-to-end it was **350ms (19.4%)**
— because the model calls SoulX's `sdpa_attention` *wrapper* (padding-mask logic,
`k_lens`, reshapes), and FA3 replaces all of it. Synthetic measurements
under-counted the win by ~5×.

Scope: self-attention only. Cross-attention runs over a ~29-token text context.
`window_size` is `(-1,-1)` and `attn_mask` is None in this config, so dropping
them on the FA3 path is safe **here** — a config with a real window must pass
them through.

## Open

- **Slightly more jerky output** — see `TODO.md`; leading hypothesis is the 1.02×
  pacing margin, not the decoder or FA3. Disambiguation plan recorded there.
- **Lipsync under compile is unverified.** ~20% win, visually clean, but nobody has
  confirmed A/V alignment survives compilation. Gate before compile is default.
- **Denoising steps 3 → 2** is now the biggest remaining lever — a full third of
  the chunk, vs attention's ~19%. Needs step distillation; quality risk is real.
- `lightvaew2_1` (causal Conv3D) untested on moving video; the fallback if a
  Conv2D TAE turns out to cause inter-frame jitter.

## Environment traps (each cost a run today)

- **`set -u` is unsafe** after sourcing `env.sh`: conda's `activate.d` references
  unbound `NVCC_PREPEND_FLAGS`, and `PYTHONPATH` is often unset.
- **`SIZE` is clobbered** by conda's binutils activation
  (`SIZE=x86_64-conda-linux-gnu-size`), so `${SIZE:-...}` silently keeps conda's
  value. Use `STREAM_SIZE`.
- **conda itself is gone** after container recreation (it lived on the root
  overlay); the env on the network volume survives, so `env.sh` now sets `PATH`
  and sources `activate.d` directly.
- **The pod's GPUs are shared with other containers** — PIDs outside our namespace
  have held GPUs 0/2/3, one pegged at 100%. Benchmark only on a verified-idle GPU.
- `torch.compile(dynamic=False)` + `max-autotune` means **every resolution
  re-tunes** (~8 min at 592×336, ~15 min at 720×416). Same pixel count in a
  different orientation is cheaper (chunk 0: 4.4s vs 11.0s) — the caches at
  `/workspace/.inductor_cache` (2.6GB) partially hit.
