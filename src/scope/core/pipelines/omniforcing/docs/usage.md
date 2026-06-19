# OmniForcing (LTX-2 causal audio-video) — usage & bring-up

OmniForcing (<https://github.com/OmniForcing/OmniForcing>) distills the bidirectional
audio-visual **LTX-2** diffusion model (19B = 14B video + 5B audio) into a *streaming
autoregressive* (causal-forcing) generator: block-causal attention + KV-cache + a short
distilled denoising schedule, producing **synchronized audio + video** from a text
prompt with a **Gemma-3-12B** text encoder. It is the LTX-2-family analogue of the
`longlive`/`longlive2` causal pipelines, with an added audio track.

Weights: `Exploration/omniforcing-ltx2-5s-causal` (distilled causal generator only).

## Status (VERIFIED on H100, 2026-06-19)

This pipeline **renders end-to-end on an H100 80GB** (secure cloud). The scope-side
integration (config/schema, HF artifacts, registry, CPU contract tests, dependency
wiring) is complete and import-safe on any host, and `runtime.OmniForcingRuntime` is
fully implemented + verified: it loads the LTX-2 base + distilled OmniForcing
generator + Gemma text encoder + video/audio VAEs + vocoder and drives
`CausalAVInferencePipeline` to produce **synchronized audio + video**.

Verified result (prompt "Realistic. Rain falls on a quiet street at night."):

- Build (model load): ~148s. Generation: **121 frames (5s @ 24fps) in 7.8s ≈ 15.6 fps**.
- Peak GPU memory **77.5GB** — fits 80GB but tight (the Gemma text encoder is ~24GB
  resident; offload-after-encode is the obvious headroom win, mirroring longlive2).
- Output: coherent photorealistic video `[121,512,768,3]` + synced stereo audio
  `[2,120240]` (5.01s vs video 5.04s). Muxed to MP4 with AAC.

The WebRTC **audio output track** is still a separate follow-up (see "Audio output
gap" below) — verification used offline MP4 mux.

## Hardware

~60GB of weights in bf16 (19B LTX-2 base + 12B Gemma + VAEs/vocoder). Targets
**H100/H200 80GB-class** GPUs — it will NOT fit a 32GB consumer Blackwell card.

## Install (GPU host only)

The LTX-2 model code is **not vendored** (LTX-2 Community License). It installs from
the OmniForcing monorepo as source packages via the `omniforcing` extra:

```bash
uv sync --extra omniforcing
```

This maps `ltx-core`, `ltx-causal`, `ltx-pipelines`, `ltx-distillation` to the
`OmniForcing/OmniForcing` git subdirectories (see `[tool.uv.sources]` in
`pyproject.toml`) and adds `soundfile` / `sentencepiece`. `ltx-distillation` pulls in
training extras (wandb, lmdb) — acceptable for the pod.

**torchaudio gotcha (fixed):** `ltx-core` imports `torchaudio` (audio VAE ops). Left
unpinned, resolution pulls `torchaudio==2.11` from PyPI, whose C++ extension
(`_torchaudio.abi3.so`) is ABI-incompatible with our `torch 2.9.1+cu128` and fails to
load (`OSError: Could not load this library`). The extra now pins
`torchaudio==2.9.1` with a `pytorch-cu128` source so it stays matched to torch.

## Weights

`download_models --pipeline omniforcing` fetches:

- **`Exploration/omniforcing-ltx2-5s-causal`** — the 5 generator shards + index (~36GB).
- **`Lightricks/LTX-2`** (**PUBLIC** — no `HF_TOKEN` needed, verified 2026-06-19). The
  CONFIRMED minimal subset is:
  - `ltx-2-19b-dev.safetensors` (~41GB) — the consolidated base. It carries the
    transformer, **both VAEs, the vocoder and the connectors**; `create_vae_wrappers`
    reads the vocoder from this file, so the per-component `vae/` `audio_vae/`
    `vocoder/` `connectors/` `scheduler/` dirs are NOT downloaded.
  - `text_encoder/` (the `model-*` shard set only) + `tokenizer/` — the Gemma-3 text
    encoder (~24GB; stored fp32 on disk, loaded as bf16). The duplicate
    `diffusion_pytorch_model-*` set in `text_encoder/` (~24GB) is skipped.

  The text-encoder loader (`ModelLedger` → `module_ops_from_gemma_root`) takes a
  `gemma_root_path` and does a recursive `rglob` for `tokenizer.model`,
  `preprocessor_config.json` and `model*.safetensors`. So the gemma root must be the
  **LTX-2 repo root** (which contains both `text_encoder/` and `tokenizer/`), NOT
  `text_encoder/` — `runtime._resolve_paths["gemma"]` points at the base dir.

Layout on disk: `<DAYDREAM_SCOPE_MODELS_DIR>/LTX-2/...` and
`<...>/omniforcing-ltx2-5s-causal/...` (see `runtime._resolve_paths`).

## Defaults (from `omniforcing_causal_inference.py`)

| Param | Default |
|-------|---------|
| frames | 121 (~5s @ 24fps, N·8+1) |
| resolution | 512×768 (multiples of 32) |
| fps | 24 |
| denoising schedule | `[1000, 909, 725, 421, 0]` (trailing 0 = clean boundary) |
| block size | 3 (first block 4) |
| audio sample rate | 24000 Hz (vocoder may BWE to 48k) |

## Pod bring-up checklist (DONE — 2026-06-19, H100 80GB)

1. ✅ **Loader entry points confirmed** against the installed packages — all exist as
   scaffolded: `ltx_causal.transformer.causal_model.{CausalLTXModel,CausalLTXModelConfig}`,
   `ltx_causal.wrapper.CausalLTX2DiffusionWrapper`,
   `ltx_distillation.inference.causal_pipeline.CausalAVInferencePipeline`,
   `ltx_distillation.models.{text_encoder_wrapper.create_text_encoder_wrapper,
   vae_wrapper.create_vae_wrappers}`, `ltx_core.loader.registry.StateDictRegistry`.
   `runtime.py` mirrors the construction sequence in
   `scripts/omniforcing_causal_inference.py` (load base → strip `audio_sink_tokens` →
   load distilled generator weights; build text encoder + VAEs; convert the timestep
   schedule to flow-matching sigmas via `LTX2Scheduler`; drive the pipeline).
2. ✅ **Minimal weight subset confirmed** — see Weights above. `LTX2_BASE_ARTIFACT.files`
   trimmed to the consolidated safetensors + `text_encoder/model-*` + `tokenizer/`.
3. ✅ **`OmniForcingRuntime.__init__` + `generate_chunk` implemented** — loads everything
   and runs a full 5s AV generation, decoding video (video VAE) + audio (audio VAE →
   vocoder) into the scope AV dict.
4. ✅ **Render-verified** — coherent 5s clip with synchronized stereo audio, muxed to MP4.

### Remaining (follow-ups)

- **Text-encoder offload** for headroom (peak 77.5GB on 80GB): move Gemma to CPU after
  prompt encode (it's used once per generation), mirroring the longlive2 plan.
- **Incremental VAE decode** (the remaining streaming optimization). The streaming
  path below is implemented; the open item is making per-block decode cheap (left-
  context overlap) instead of re-decoding the growing take each block.
- Confirm `uv sync --extra omniforcing` (with the new torchaudio pin) resolves cleanly
  from scratch — the pod was bootstrapped via `uv pip install` over the prebuilt image.

## Continuous streaming (block-by-block) — DRAFTED, pod-verify pending

The default path is now **continuous streaming**: instead of regenerating a fresh 5s
clip every call (which loops), `generate_chunk` advances **one causal block** of a
single take against a persistent KV cache. Successive calls continue the same shot,
so the output is a coherent, non-looping stream — the basis for "long video".

How it works (`runtime.OmniForcingRuntime._generate_streaming`), mirroring the upstream
`CausalAVInferencePipeline.generate` body but one block per call:

- `init_cache=True` (re)starts the take: allocate `gen.init_av_kv_caches(...)` once,
  re-seed, encode the prompt, reset the running `current_video/audio_start_frame`.
- Each call denoises one block (`num_frame_per_block`, first block `_first`) over the
  distilled `denoising_sigmas`, runs the context-noise cache refresh, advances the
  frame counters, and decodes **only the newly-revealed** frames/samples.
- A mid-stream **prompt change** re-encodes the conditioning but KEEPS the cache, so
  the character/scene stays continuous while the new prompt steers from there.

Config knobs (`schema.py`): `streaming` (default True; False = old one-shot loop) and
`stream_max_seconds` (KV-cache budget; default 12s, max 20s).

Limits to verify on the pod:

- The KV cache is a **fixed buffer with no sliding window** (no `local_attn_size`), so
  `stream_max_seconds` bounds both max continuous length and VRAM (~1.5GB/s of video
  cache). On overflow the stream **re-anchors** (a visible seam) and logs a warning.
- RoPE stays valid to ~20s (`pe_max_pos[0]` is in seconds), but the released checkpoint
  was **distilled at 5s**, so coherence/identity may drift past ~5–6s — empirical.
- **Incremental decode** is correctness-first today: `_decode_new` re-decodes the
  growing latent each block (the LTX video VAE is causal, so decoding a block in
  isolation mis-aligns frame counts) and slices off new frames. This re-decode cost
  grows with take length — the optimization is a left-context-overlap decode. This is
  the #1 thing to tune/verify on the H100.

## Audio output gap (Phase 4)

This fork's WebRTC stack is **video-only** (`server/webrtc.py` / `server/tracks.py` wire
only `VideoProcessingTrack`). `OmniForcingPipeline.__call__` already returns the
audio-capable dict shape used by `daydreamlive/scope-ltx-2`'s `LTX2Pipeline`
(`video`, `video_timestamps`, `audio`, `audio_sample_rate`, `audio_timestamps`,
`frame_rate`). Streaming the audio live requires adding an `AudioProcessingTrack` and an
audio track on the peer connection (shared 90kHz media clock for A/V sync). Until then,
validate offline by muxing the returned audio + video into an MP4.

## Persona relevance

OmniForcing's synchronized-audio causal streaming is a strong fit for the future
"speaking persona with lip-sync" direction (TODO Phase 5): a real-time character that
both speaks (audio) and moves (video) from a single conditioned generator.
