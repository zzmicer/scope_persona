# SoulX-LiveAct — model optimization brief

Handoff for an agent tasked with pruning/optimizing the models behind the
interactive persona demo. Everything below is measured on the pod unless marked
as an estimate.

## The one thing to internalize first

**The constraint is LATENCY, not memory.** On the target box (1×H200, 143GB) the
demo occupies 76.4GB. There is 67GB spare. Optimizations that trade compute for
memory are worthless here; optimizations that cut wall-clock per chunk are the
entire job.

This inverts the usual "model optimization" instinct. Do not spend effort on
weight quantization for footprint, offloading, or paging unless the target moves
to a consumer card (see §6).

## Budget

The demo generates a **32-frame chunk covering 2.0s of video at 16fps**, so the
chunk must be produced in **under 2.0s** — and comfortably under, because an
explicit wall-clock pacer keeps ~1.6s of lead and any chunk over budget shows up
as a micro-freeze. Target ~1.8s, not 2.0s.

| Config | s/chunk | Realtime |
|---|---|---|
| 1×H200, 720×416, FP8-blocks, Wan VAE decode | 3.61 | 0.55× |
| 1×H200, 720×416, FP8-blocks, **v3 fast decode** | **3.01** | 0.66× |
| 2×H200, 720×416, FP8-blocks | 2.00 | 1.00× |
| 2×H200, 320×576, FP8-blocks | 1.16 | 1.72× |

Generation time scales **~linearly with pixel count** (0.62× pixels → 0.58× time),
which is the cheapest dial but costs image quality directly.

## Where the 3.01s goes

Decode is **solved** — do not re-optimize it. The Wan2.1 VAE decode (452ms) was
replaced by the reconstructed UltraFlash v3 tiny decoder: **31ms, MAE 6.41/255**
against the Wan VAE on real dumped latents. That is ~421ms already banked.

That leaves **~2.97s in the DiT**, which is the whole target:

- **18B params** (Wan2.1 14B backbone + ~4B audio path), 35.4GB bf16
- **40 blocks**, each `{self_attn, cross_attn, ffn}`
- **3 denoising steps** — `timesteps = [1000, 937.5, 833.33, 0]`
- **8 latent frames/chunk** (`blksz_lst = [6, 8]`; chunk 0 is 6)
- At 720×416: `frame_len` = (416/16)×(720/16) = **1170 tokens/latent frame**,
  KV ≈ **16.4k tokens** (`frame_len × 14`)
- Already quantized: **FP8 W8A8 scoped to the 480 block matmuls**
  (`--fp8_scope blocks`)

So the cost is roughly `40 blocks × 3 steps × (attn over 16.4k KV + FFN)`.

## Ranked targets

### 1. Attention backend — best value/effort, no retraining

**The single-GPU path is running PyTorch SDPA.** In
`model_liveact/model_memory.py`, `USE_SAGEATTN` is False (SageAttention 2.2.0 is
installed but produces noise/NaN on this pod's sm90 and is deliberately blocked
via a PYTHONPATH shim), and the fallback is `sdpa_attention` — with the
`flash_attention` calls sitting **commented out** right above at lines 381/384.

- flash-attn **2.8.3 is installed and unused on this path**
- flash-attn **3 is not built** (`flash_attn_interface` → ModuleNotFoundError);
  FA3 is the Hopper-optimized one and H200 is Hopper
- The multi-GPU path is different — `model_memory_sp.py` uses
  `xFuserLongContextAttention(attn_type=AttnType.FA)`, i.e. FA2

Tasks: benchmark SDPA vs FA2 vs FA3 on the **actual shapes** (q: 1170 tokens,
kv: 16.4k, 40 heads); build FA3 for sm90; and re-open the SageAttention NaN —
Sage's FP8 attention would be a large win if the root cause turns out to be
configuration rather than the kernel.

### 2. Denoising steps 3 → 2 — the biggest single lever

A third of the DiT cost. **3.01s → ~2.0s on its own**, without touching
resolution. Needs step distillation or a schedule search over the existing
distilled model. Precedent: the sibling longlive2 pipeline runs a 2-step
schedule (`[1000, 681]`) that works empirically. Highest payoff, highest quality
risk — validate hard (§7).

### 3. Depth pruning of the 40 blocks

Layer-importance analysis → drop 4–8 blocks → short heal finetune. Expect
10–20% latency. Precedent that this family tolerates pruning: LightVAE is
literally a pruned `PrunableWanVAE`.

### 4. torch.compile

Disabled everywhere (`--no_compile` in every run script) and **nobody knows
why** — it may have been startup cost rather than instability. The code already
has the call sites (`torch.compile(self.wan, ...)` and on `vae.decode`). Free
10–30% if it holds. Cheapest experiment on this list; do it first.

### 5. Step-selective computation

There is already precedent in the codebase: the audio path is computed only on
steps 1 and 2 (`skip_audio=False if i in [1,2] else True`). Worth auditing what
else is recomputed identically across the 3 steps — cross-attention over an
unchanged text context is the obvious candidate. Classic caching methods
(TeaCache, first-block-cache) have limited headroom at only 3 steps.

### 6. Memory-side work — ONLY if targeting a consumer GPU

Skip for the 1×H200 target. Relevant only if the goal becomes a 32GB card:

- **umt5-xxl text encoder: 11GB resident, near-zero duty cycle.** Only runs when
  the composed prompt changes. Better than offloading: the persona's action
  vocabulary is **bounded**, so prompt embeddings can be precomputed offline and
  umt5 never loaded at runtime.
- **CLIP (xlm-roberta-large-vit-huge-14): 4.5GB, used once per session** for the
  reference image.
- **The Wan2.1 VAE decoder half is now dead weight** — only `vae.encode` is still
  called. Free the decoder submodule.

That is ~16GB of the 76GB for almost no duty cycle.

### 7. Quantization beyond current FP8 — low priority on H200

Already at FP8 W8A8 on the block matmuls. NVFP4 is **Blackwell-only**, so it does
nothing on Hopper; INT4 weight-only would cut footprint but not latency (no fast
W4A8 path on sm90). Promote this **only** if the target moves to Blackwell — in
which case there is hard-won NVFP4/fouroversix knowledge in
`docs/longlive2-runpod-bringup.md` Session 2.

## Validation — non-negotiable

This pipeline has produced silent, plausible-looking corruption twice. Both times
the model still ran and still emitted frames.

1. **SageAttention sm90 kernels → noise/NaN** on this pod. Blocked via shim.
2. **FP8 applied too broadly → noise.** `enable_fp8_gemm` was called without its
   `module_filter`, wrapping *every* `nn.Linear` including `time_projection` (the
   AdaLN modulation scaling every block's residual), `head`, `img_emb` and
   `audio_proj` — no FLOPs, maximum quantization sensitivity. **Any new
   quantization must be scoped to the block matmuls only.**

Required gates for any change:

- **No NaN/noise**, obviously, but check the *subtle* case — identity drift of the
  character across chunks, not just per-frame sanity.
- **Lipsync alignment** survives (the audio path is 4B of the 18B).
- **Frame-count contract**: chunk 0 emits 21 frames, subsequent chunks 32
  (`n_lat*4-3`). A/V sync is driven off `frames_emitted`/`audio_samples_emitted`,
  so a changed count desyncs audio.
- **Measure with the real harness** — `interactive_demo.py` prints per-chunk
  `32f in X.XXs`. Synthetic microbenchmarks have already misled here once.

Useful reference material already on the pod:

- `/workspace/uflash/dump/chunk_*.pt` — real dumped SoulX latents + T5 context
- `/workspace/uflash/dec_native_cmp.py` — the MAE-vs-reference comparison pattern
- `/workspace/soulx/run_single_fast.sh` — 1-GPU harness, `STREAM_SIZE` etc.

## Explicitly out of scope

- **VAE decode** — solved at 31ms.
- **Memory reduction on H200** — 76 of 143GB.
- **persona_aux** (Qwen2.5-1.5B + Kokoro, ~4GB) — negligible, off the chunk path.
- **wav2vec2-base** (1.5GB) — small per-chunk cost.

## Environment notes

- Pod GPUs are **shared with other containers** — PIDs outside our namespace hold
  GPUs 0/2/3, one pegged at 100%. Benchmark on a verified-idle GPU or numbers lie.
- `env.sh` no longer needs conda (the installer lived on the wiped root overlay);
  it sets PATH + sources `activate.d` directly. `env.sh.conda-bak` is the original.
- The conda env `/workspace/soulx/env` and all weights are on the **network
  volume** and survive container recreation.
