# Daydream Scope — AI Persona TODO

## In Progress

- [~] Define product direction and update CLAUDE.md with persona vision
- [~] Research character consistency techniques (persistent latent conditioning, reference image anchoring, IP-Adapter)

## OmniForcing (LTX-2 causal AV) Integration

See plan: `.claude/plans/dreamy-frolicking-snail.md` and pipeline docs:
`src/scope/core/pipelines/omniforcing/docs/usage.md`. New `omniforcing` pipeline:
streaming autoregressive **audio+video** distilled from bidirectional LTX-2 (19B),
Gemma-3-12B text encoder. ~60GB bf16 → H100/H200 only. Weights:
`Exploration/omniforcing-ltx2-5s-causal` + gated `Lightricks/LTX-2` base.

- [x] Phase 0: branch `omniforcing` + deps (`omniforcing` extra → ltx-core/causal/
  distillation git-installed; soundfile/sentencepiece) + HF artifact ids/files verified.
- [x] Phase 3: scope-side scaffold — `omniforcing/{schema,pipeline,runtime,model.yaml,
  __init__}.py`, registry entry, import-safe offline; `runtime.is_available()` gates the
  LTX stack. 7/7 CPU contract tests pass (`tests/test_omniforcing_contract.py`).
- [x] Phase 1+2: LTX-2 stack + causal layer — NOT vendored (LTX-2 Community License);
  installed via the `omniforcing` extra on the pod. `runtime.OmniForcingRuntime` is
  IMPLEMENTED + verified (CausalLTXModel + VAEs + vocoder + Gemma +
  CausalAVInferencePipeline). torchaudio pinned to 2.9.1 (PyPI 2.11 breaks the ABI).
- [x] Pod bring-up (H100 80GB, 2026-06-19): loader entry points confirmed (all as
  scaffolded); minimal weight subset confirmed + `LTX2_BASE_ARTIFACT` trimmed
  (consolidated safetensors carries VAEs+vocoder; gemma_root = LTX-2 root, not
  text_encoder/); `generate_chunk` implemented; **render-verified a coherent 5s clip
  with synced stereo audio** ("rainy night street", 121 frames in 7.8s ≈ 15.6 fps,
  peak 77.5GB). MP4 muxed offline. See `docs/.../omniforcing/docs/usage.md`.
- [ ] Phase 4: audio output plumbing — fork's WebRTC is video-only; add
  `AudioProcessingTrack` + audio track in `server/webrtc.py` (90kHz A/V sync). Pipeline
  already returns the AV dict; offline MP4-mux is the interim validation path.
- [~] Continuous streaming `generate_chunk` (block-by-block + persistent KV cache) —
  on `omniforcing` branch. First pod test (2026-06-20, H100 NVL 94GB): LOOP FIXED (231
  continuous frames, no OOM at 12s budget), BUT ~2fps + quality bad from frame 1.
  Root-caused: `_decode_new` re-decoded the whole growing clip each block (2fps), and the
  LTX-2 video VAE is NON-causal (`causal_decoder` False) so decoding growing prefixes
  gave per-block boundary artifacts. FIXED offline: windowed decode (left context +
  right look-ahead, emit interior frames only → bounded/constant cost + bilateral
  context), skip audio decode while streaming (WebRTC is video-only), per-block timing
  logs, knobs `decode_context_latents`/`decode_lookahead_latents`/`stream_audio`.
  11/11 CPU contract tests (incl. windowed-decode bookkeeping). NOT yet re-tested on pod.
  Pod TODO: re-test quality + fps; tune ctx/look to the VAE's temporal receptive field;
  then measure >5s drift (ckpt distilled at 5s).
- [ ] OmniForcing follow-ups: text-encoder offload (Gemma ~24GB resident → CPU after
  encode for headroom); confirm clean `uv sync --extra omniforcing` from scratch (pod
  was bootstrapped via `uv pip install`).

## LongLive 2.0 (NVFP4) Integration

See plan: `.claude/plans/longlive2-nvfp4-integration.md`. New `longlive2` pipeline on Wan2.2-TI2V-5B
(text + image), NVFP4 W4A4 on RTX 5090. Native kernel phases run on the 5090, not macOS.

- [~] Phase 0: env/dep spike on Blackwell (RTX PRO 6000, sm_120) — kv_dequant ext BUILDS; fouroversix conflict ROOT-CAUSED (needs transformers>=5.0.0 for `WeightConverter`; entrypoint now upgrades it in NVFP4-only branch); transformer-engine dev headers now auto-installed by entrypoint. TE build + render still to validate on pod. See `docs/longlive2-runpod-bringup.md`.
- [x] Phase 1: scaffold `pipelines/wan2_2/` component layer — causal 5B model loads (825/825 keys match), VAE 2.2 (48ch) decodes
- [x] Phase 2: `pipelines/longlive2/` BF16 correctness gate PASSED — renders coherent video on GPU after fixing model.yaml load, latent channels (16→48), and denoising schedule
- [x] DMD timesteps: confirmed real 5B 4-step from upstream NVlabs/LongLive → `[1000, 946, 854, 681]` (was wrong assumed `[1000,750,500,250]`); set in model.yaml. 2-step s2 still unverified (no upstream sampling_steps:2 config; subsample = `[1000,681]`).
- [~] Phase 3: NVFP4 path — NVFP4 RENDERS CORRECTLY on Blackwell (sm_120). Fixed: pipeline NVFP4 wiring (shim + _inject_nvfp4_config), 2-step noise (enforce distilled schedule), from_blocked import drift. Key: needs fouroversix **1.1.0 from LongLive repo + cutlass** (PyPI 1.0.5 has incompatible 6-field metadata vs checkpoint's 4-field); 1.1.0 build downgrades cudnn→9.10 (VAE OOM) so restore cudnn. transformers v5 confirmed safe. REMAINING for 45fps (currently ~6.7fps): VAE decode is the bottleneck (~10fps full Wan2.2 VAE) → integrate **LightVAE** (mg_lightvae); also dispatch cutlass FP4 backend in generator (~20fps). See `docs/longlive2-runpod-bringup.md` Session 2.
- [x] Phase 3b: post-review hardening (code review of longlive2 branch) — FIXED: (1) `LONGLIVE2_NVFP4_TE` env opt-in was disconnected (nothing read `model_quant_use_transformer_engine`, so the TE backend was unreachable) → now a real schema field + the manager honors the env var, which `_inject_nvfp4_config`/`setup_nvfp4_pipeline` use to pick the SETUP-time weights; (2) `setup_nvfp4_pipeline` called `pipeline.to()` on a `SimpleNamespace` (crashed the entire TE / non-prequantized branch) → casts per-component; (3) VAE decode hardcoded `.to("cuda")`, breaking the advertised CPU/macOS fallback → decodes on the VAE's own device; (4) added `tests/test_longlive2_contract.py` (CPU-runnable: 5B config, RoPE split, schedule subsample, backend file selection, model.yaml/VAE consistency, NVFP4 probe import-safety). NOTE: the manager still loads `model_te.pt` (the BF16 base) at WRAPPER CONSTRUCTION for NVFP4 — this is correct-by-design (clean seed for the model + the BF16 fallback; quantized `model_4o6.pt` is loaded later by setup), not a bug; documented inline.
- [x] Phase 3b VERIFIED on RTX 5090 (32GB, SECURE) 2026-06-18 — fresh pod, branch overlaid, **13/13 contract tests pass on GPU** (incl. the 2 that skip on macOS). FourOverSix NVFP4 **renders a coherent panda** end-to-end (real frames, no NaN, schedule forced to `[1000,681]`) at **25→56 fps** @256×448. Memory profile after build: generator(NVFP4)=**0.18GB**, text_encoder(umt5-xxl)=**11.36GB**, vae=1.41GB, total alloc 15.7GB → CONFIRMS the `model_te.pt` BF16 base IS freed after quantization (no double-load *leak*; the redundant read is time-only). OOM at ≥448×768 on 32GB is driven by the **11GB resident text encoder** + KV cache + Wan2.2 VAE decode, NOT the generator.
- [ ] Phase 3c follow-ups (NOT done): (a) **Text encoder is resident at ~11GB in bf16** — the real blocker for the schema's "~45fps on RTX 5090" claim at full 704×1280; offload to CPU after prompt-encode OR load the fp8 weights as fp8 to fit 32GB at full res. (b) TE path is selectable + no longer crashes but is STILL render-unverified (needs `transformer-engine` wheel built on Blackwell — not installed in this run, so the line-262 fix is code-verified only). (c) 2-step S2 schedule `[1000,681]` still a heuristic subsample, unconfirmed upstream. (d) generator checkpoint still read at construction AND in setup (correct file, redundant time only — `defer_weight_load` flag would remove it).
- [~] Phase 4: Scope integration — WebRTC/TURN FIXED (Cloudflare direct keys, validated); prompt-switch/I2V/VRAM tuning remain
- [ ] Phase 5 (optional): VACE/LoRA parity
- [~] CI/Dockerfile: runtime image was shipping without nvcc/CUDA headers — FIXED in Dockerfile (CUDA_HOME + PATH + build-time `nvcc` assertion) + entrypoint (auto-installs dev headers). Verify on next CI build + pod boot.

## Phase 1: Foundation — Action Schema & Interpreter

- [ ] Design structured action/expression schema (Pydantic models for `{ action, expression, dialogue, intensity }`)
- [ ] Define action vocabulary — enumerate supported actions (wave, sit, stand, nod, turn, walk, etc.) and expressions (smile, frown, laugh, confused, neutral, etc.)
- [ ] Build action interpreter module (`scope.core.persona.action_interpreter`) — LLM call that maps free-text user input to structured action directives
- [ ] Write tests for action interpreter (edge cases: ambiguous commands, multiple actions, unknown actions)
- [ ] Decide on LLM provider/model for action interpretation (local vs API, latency budget)

## Phase 2: Conversation Layer

- [ ] Build conversation manager (`scope.core.persona.conversation`) — maintains chat history, persona system prompt, personality definition
- [ ] Design persona personality config schema (name, backstory, voice style, behavioral constraints)
- [ ] Create chat API endpoint (`server/chat.py`) — WebSocket or SSE for real-time back-and-forth
- [ ] Wire conversation manager → action interpreter → pipeline manager flow
- [ ] Add character state tracker (`scope.core.persona.state`) — tracks current pose, expression, activity for coherent transitions

## Phase 3: Persona Video Pipeline

- [ ] Scaffold persona pipeline directory (`core/pipelines/persona/`)
- [ ] Implement persona pipeline `__call__()` accepting structured action directives (not raw text)
- [ ] Integrate character consistency module — evaluate approaches: IP-Adapter, reference image conditioning, LoRA identity, latent anchoring
- [ ] Implement smooth action transitions (don't jump-cut between poses; interpolate or blend)
- [ ] Benchmark frame latency — must stay real-time (<100ms per frame target)
- [ ] Test character identity drift over long sessions (>5 min continuous generation)

## Phase 4: Frontend Chat UI

- [ ] Design chat interface component (message history + text input)
- [ ] Integrate chat UI alongside or replacing the timeline/prompt editor
- [ ] Show character status indicator (current action/expression)
- [ ] Display real-time video stream next to chat panel
- [ ] Add action shortcut buttons (quick-fire common actions like wave, smile, nod)

## Performance: SpargeAttn (Sparse SA3) for Autoregressive DiT

- [ ] Add SpargeAttn support — sparse block-masking on top of SageAttention3 FP4 for inference speedup on Blackwell GPUs
  - New wrapper module (`wan2_1/modules/sparge.py`) following `sage.py` pattern
  - Extend `attention()` routing with `use_sparge`/`sparge_topk` params
  - Hybrid precision: first 2 + last 2 layers use standard SA3, middle layers use SpargeAttn+SA3
  - Timestep-conditional: only active for t < 800 (configurable)
  - Config fields on LongLiveConfig (`sparge_attention`, `sparge_topk`, `sparge_timestep_threshold`)
  - Dependencies: `sparge-attn` (thu-ml/SpargeAttn), compile at runtime like sageattn3
  - Extend Modal test to validate sparge kernel compilation
  - See plan: `.claude/plans/recursive-sauteeing-castle.md`

## Phase 5: Polish & Future

- [ ] Audio/TTS integration — persona speaks responses, lip-sync with generated video
- [ ] Emotion inference from conversation context (auto-set expression without explicit user command)
- [ ] Multi-character support (stretch goal)
- [ ] Persona gallery — prebuilt characters users can select
- [ ] Export/record conversation sessions as video files

## Completed

- [x] Initial project setup (LongLive pipeline working with WebRTC streaming)
- [x] Update CLAUDE.md with AI persona product direction and architectural plan
- [x] Create TODO.md for session continuity

## Decisions & Notes

- **Separation of concerns is critical**: Conversational AI layer (LLM, chat, action interpretation) must be fully decoupled from video generation layer (pipeline). They communicate via the structured action schema only.
- **Character consistency is the hardest unsolved problem** — research this early, prototype multiple approaches before committing.
- **Latency budget**: Action interpreter LLM call + video generation must fit within ~200ms total for real-time feel. Consider pre-generating idle animations to fill gaps.
- **Existing LongLive pipeline stays intact** — persona pipeline is a new pipeline registered alongside it, not a replacement.
