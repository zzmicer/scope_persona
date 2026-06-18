# OmniForcing (LTX-2 causal audio-video) — usage & bring-up

OmniForcing (<https://github.com/OmniForcing/OmniForcing>) distills the bidirectional
audio-visual **LTX-2** diffusion model (19B = 14B video + 5B audio) into a *streaming
autoregressive* (causal-forcing) generator: block-causal attention + KV-cache + a short
distilled denoising schedule, producing **synchronized audio + video** from a text
prompt with a **Gemma-3-12B** text encoder. It is the LTX-2-family analogue of the
`longlive`/`longlive2` causal pipelines, with an added audio track.

Weights: `Exploration/omniforcing-ltx2-5s-causal` (distilled causal generator only).

## Status (scaffold)

This pipeline is **scaffolded**: the scope-side integration (config/schema, HF
artifacts, registry entry, CPU contract tests, dependency wiring) is complete and
import-safe on any host. The **GPU runtime** (`runtime.py` → `OmniForcingRuntime`) is
finalized on the pod — it raises `NotImplementedError` until the LTX stack is installed
and the loader/AR-loop is wired against the installed package. The WebRTC **audio
output track** is a separate follow-up (see "Audio output gap" below).

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

## Weights

`download_models --pipeline omniforcing` fetches:

- **`Exploration/omniforcing-ltx2-5s-causal`** — the 5 generator shards + index.
- **`Lightricks/LTX-2`** (gated; needs `HF_TOKEN` with access) — the consolidated
  `ltx-2-19b-dev.safetensors` plus the diffusers-layout `vae/` (video), `audio_vae/`,
  `vocoder/`, `connectors/`, and `text_encoder/` + `tokenizer/` (the Gemma-3-12B
  encoder). One repo covers base + both VAEs + vocoder + text encoder, so no separate
  gated `google/gemma-*` download is required.

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

## Pod bring-up checklist (the remaining work)

1. **Confirm loader entry points** against the installed packages:
   `ltx_causal.transformer.causal_model.CausalLTXModel`,
   `ltx_distillation.inference.causal_pipeline.CausalAVInferencePipeline`,
   `ltx_distillation.models.text_encoder_wrapper.create_text_encoder_wrapper`. Mirror
   the construction sequence in `scripts/omniforcing_causal_inference.py` (load base →
   strip audio tokens → load distilled generator weights; build VAEs/vocoder; build the
   pipeline with `add_noise_fn` + `denoising_sigmas` derived from the timestep schedule).
2. **Confirm the minimal weight subset** the loader actually reads (single
   `ltx-2-19b-dev.safetensors` vs. the per-component diffusers dirs; bundled
   `text_encoder/` vs. a standalone Gemma path). Trim `LTX2_BASE_ARTIFACT.files` to match.
3. **Implement `OmniForcingRuntime.__init__` + `generate_chunk`** to run one AR block and
   decode video (video VAE) + audio (audio VAE → vocoder), returning the AV dict.
4. **Render-verify**: a coherent 5s clip with synchronized audio. Write an MP4 with muxed
   audio first (as the upstream script does), then move to live streaming.

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
