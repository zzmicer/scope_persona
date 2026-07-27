# Daydream Scope — AI Persona TODO

## In Progress

- [~] Define product direction and update CLAUDE.md with persona vision
- [x] Write product vision doc (`VISION.md`) — target product modeled on Vidu S1
  (arXiv:2607.03118) / Vidu Stream: voice/chat-controlled persona, infinite
  drift-free streaming, 540p ≥24fps, single-image characters, joint AV lipsync
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

## LingBot-World-V2 (interactive world model) — branch `lingbot-world`

14B causal-fast world model (Wan2.2-based, CC BY-NC-SA) from
github.com/Robbyant/lingbot-world-v2: image → interactive world, camera controlled by
per-latent-frame pose deltas (Plücker), events/scene by prompt swaps, rolling KV cache
(`local_attn_size=18` + sink 6) → unbounded horizon. Runs on 1×H200 (~70GB, no offload,
~4.5s per 4-latent-frame chunk ≈ 3.5 video-fps at 480×832).

- [x] Pod bring-up (4×H200, 2026-07-12): deps + prebuilt flash-attn 2.7.4.post1
  (cu12torch2.4 cp311) + weights (~93GB) on /workspace; stock `generate.py` verified
  single-GPU (161 frames in 45s; text prompt morphed lakeside→Great Wall = text-driven
  world change confirmed).
- [x] `src/scope/core/pipelines/lingbot_world/`: `actions.py` (text → camera
  trajectories), `session.py` (turn-based generation over persistent KV cache +
  windowed VAE decode), `demo.py` (interactive/scripted CLI), README.
- [x] Interactive demo verified on pod: event prompts animate the character (she
  waves, identity preserved); camera motion navigates the world. Event prompts must be
  composed with the base world prompt and reverted after one turn (else the world
  morphs into the event's subject — learned from smoke test).
- [x] Wire into scope pipeline registry + WebRTC streaming (2026-07-12):
  `lingbot-world` registered (schema.py/pipeline.py), ctrl_input (WASD/arrows/
  Q/E/mouse) → per-chunk camera pose deltas, prompt updates → events,
  first_frame_image seeds the world, horizon exhaustion re-seeds from last
  frame. Deployed on the H200 pod (/workspace/scope, Cloudflare TURN).
  Browser/WebRTC verified: 1,043-frame interactive stream at 256×448 (~1.2s per
  16-frame chunk); parameter updates now coalesce latest keys + accumulated mouse
  motion instead of dropping new controls during generation (100-update burst,
  zero drops). Follow-ups: [x] browser-verified end-to-end stream; [x] add scipy/easydict/
  ftfy as a `lingbot` extra in pyproject (uv-pip-installed on pod for now);
  [x] reproduce `beauty.mp4` interaction with a dedicated Event Proposals panel
  (1 hair, 2 chin-in-hands, 3 candle, F butterfly, G snow) and persistent
  base+event prompt composition; [ ] frame pacing vs ~4s/chunk generation (buffer
  tuning); [ ] chat UI / agent-generated event proposals.
- [ ] LLM-based action interpreter (free text → motion/event tuples) per CLAUDE.md
  persona architecture (current parser is keyword-based).
- [ ] Explore upstream `wasd_action`/`ijkl_action` channels (present in examples but
  unused by released code) and the causal-pretrain 1.3B when released (speed).

## SoulX-LiveAct (interactive talking/acting persona) — pod /workspace/soulx

18B (Wan2.1 14B + 4B audio) realtime streaming human animation from
github.com/Soul-AILab/SoulX-LiveAct: reference image + audio → lip-synced video at
20fps on 2×H200; motion/emotion controlled by per-chunk prompt swaps (edit_prompt).
Goal: chat with the chano39 anime character; she speaks (kokoro TTS) and performs
moves from prompts in realtime.

- [x] Pod bring-up (4×H200, 2026-07-16): weights (51GB LiveAct + wav2vec2) on
  /workspace/soulx/weights; conda env on /workspace/soulx/env (root disk full);
  torch 2.8 cu128 + vllm 0.11 + SageAttention v2.2 + LightVAE + flash-attn 2.8.3
  wheel (nvcc 12.8 via conda; xformers needs XFORMERS_IGNORE_FLASH_VERSION_CHECK=1).
- [x] E2E VERIFIED (2026-07-17): interactive demo live on GPUs 2,3 port 8090 —
  chat (Qwen2.5 → {say, action}), kokoro TTS lipsync, realtime action prompts,
  clean 720×416 stream at ~14.9fps gen (target 20).
  CRITICAL fixes: (1) SageAttention sm90 kernels produce noise/NaN on this
  pod — run SDPA/flash-attn instead (sage import blocked via PYTHONPATH shim,
  model_memory_sp.py AttnType.SAGE_FP8_SM90→FA); FP8 GEMM (vllm) also noise —
  `--no_fp8_gemm`; (2) rank0-only kokoro/LLM sampling desyncs CUDA RNG across
  SP ranks → half-frame corruption; latents now use a dedicated torch.Generator.
- [x] Realtime no-player streaming (2026-07-17): HLS replaced by /ws WebSocket
  (JPEG frames + PCM16, ts-prefixed) → canvas + WebAudio client, ~0.4s latency;
  fps 20→16 so ~15fps generation keeps up. Session-OOM leak fixed
  (per-forward xFuserLongContextAttention → singleton; mem flat 76GB, 150+ chunks);
  sessions auto-restart on error (≤3).
- [x] Vidu-goal reconciliation (2026-07-20): SoulX is now the primary open/local
  product baseline, not an isolated demo. Official Vidu S1 currently exposes a
  product/API/agent skill but no model code or weights; keep Vidu as the hosted
  reference/evaluation oracle and keep the persona contracts provider-neutral.
  Latest API path is Aliyun RTC media + a server-proxied control WebSocket and
  currently hard-caps calls at 600s (fresh session/credentials required).
- [ ] Integrate SoulX as a Scope persona service/provider: preserve its persistent
  distributed session and A/V clock; expose lifecycle, structured directives,
  status, and media through Scope without coupling `scope.core.persona` to Flask
  or the SoulX runtime.
- [~] Add Vidu-style voice interaction: FIRST PROTOTYPE DEPLOYED 2026-07-20 —
  browser SpeechRecognition mic → the existing chat/action interpreter, live
  captions, plus retained Chat/Say/Do modes and shortcuts. Follow-up: replace
  browser-dependent one-shot recognition with provider-neutral streaming ASR.
- [x] Persistent action states (Wan-Streamer v0.3 "world + event stream") —
  branch `soulx-persistent-state`. POD-VERIFIED 2026-07-21 on 2xH200. Was: any
  action held ~4 chunks then hard-reverted to a pose-locked idle prompt (she'd sit
  then stand back up). Now the T5 context is composed as world (persistent,
  pose-neutral identity/scene) + sustained state (sticky held posture) + transient
  transition (motion held `action_hold` chunks then dropped). Posture changes
  update the sustained state so she STAYS in the new pose; gestures (wave/nod)
  don't. Qwen emits `{say, action, pose}` (pose = resulting held state or null);
  `/action` takes `{pose}`/`{persist}`; empty body clears to neutral idle; UI adds
  Sit/Stand/Turn buttons. Log verify (per-chunk `state=/trans=`): SIT -> state
  sticks 30+ chunks after trans expired; WAVE -> state stays sitting (gesture
  didn't disturb it); IDLE -> cleared; /chat "sit down" -> Qwen returned distinct
  action + pose, pose persisted. Stable 14.9fps, 72.3GB flat, no errors, ranks in
  sync. VISUAL CHECK DONE 2026-07-27 (frames pulled off /ws) — SPLIT RESULT:
  transient transitions DO render (an "arms high above her head" directive visibly
  swings an arm into frame), but the sustained POSTURE does not: after
  `pose="She is sitting up..."` stuck in state for 30+ chunks she is still lying
  in the reference-image pose. Identity itself is stable (no drift). So the state
  machine is correct and gestures work; gross re-posing is overpowered by the
  reference-image conditioning, which is applied at constant strength every chunk
  and pulls each chunk back to the reference pose (the VACE identity-vs-motion
  tension already noted in memory). Next: decay/re-anchor the reference strength,
  or drive posture through something other than the T5 prompt.
- [x] Vertical/portrait video (2026-07-27, pod-verified 416*720 @ 15.0fps, 72.3GB
  — identical to landscape, since SP shards the FRAME axis and only
  frame_len=(H/16)*(W/16)=1170 reaches the model; `480*832` would be 1.33x tokens).
  `--size` now validated by `_parse_size` (multiples of 16) instead of failing deep
  in the transformer; chat.html canvas placeholder templated (it already resized
  from the ws `meta` frame). TWO GOTCHAS: (1) the reference image is centre-cropped
  to the stream aspect, so the stock 16:9 chano39 loses her face in portrait — use
  the pre-framed `/workspace/soulx_setup/chano39-portrait.png`; (2) `conda activate`
  exports binutils `SIZE=<host>-size` and silently clobbered a `SIZE=` override, so
  run_interactive.sh overrides are now `STREAM_SIZE`/`STREAM_FPS`/`STREAM_PORT`/
  `STREAM_IMAGE`. Launch: `STREAM_SIZE='416*720' STREAM_IMAGE=...portrait.png
  ./run_interactive.sh`.
- [~] Add camera perception: local camera preview + permission UX deployed;
  follow-up is vision/emotion context for the conversation manager, never direct
  coupling to the SoulX generator.
- [~] Add character creation flow: deployed name, personality, reference-image
  upload/validation, and four Kokoro voice choices; chano39 remains the default
  fixture. Follow-ups: template gallery and content-safety validation.
- [x] Vidu-style live-call shell deployed on the H200 pod (2026-07-20): full-screen
  generated character, create/join screen, presence state, captions, mic/camera/
  end/chat controls, action drawer, mobile layout. Verified persona config,
  action, and chat→Qwen→Kokoro→SoulX paths live at stable ~14.9fps. Qwen/Kokoro
  isolated in `persona_aux.py` on physical GPU 0; SoulX/NCCL sees only GPUs 2,3.
  Warm chat+TTS request verified in 1.85s without stopping generation.
- [ ] Establish Vidu-aligned baseline metrics: action latency (next chunk), A/V
  sync, FPS, identity/quality curves over >=30 minutes, and automatic-recovery
  behavior. Live check 2026-07-20: active, no error, 132,339 chunks.
- [ ] Optional hosted-reference adapter: evaluate Vidu API behind the same
  session/provider boundary (Aliyun RTC + proxied control WS); handle NOT_READY
  retries, heartbeat, billing state, stale events, and the 600s renewal boundary.
- [x] Perf: FP8 W8A8 WORKS (2026-07-27) — the old "FP8 produces noise" finding was
  wrong about the cause. `enable_fp8_gemm` was called without its `module_filter`,
  so EVERY nn.Linear got wrapped, including `time_projection` (the AdaLN
  modulation scaling every block's residual), `head`, `img_emb` (identity),
  `audio_proj` (lipsync) — no FLOPs, maximum quantization sensitivity. New
  `--fp8_scope blocks` wraps only the 480 `blocks.N.{self_attn,cross_attn,ffn}`
  matmuls: 416*720 went 15.0→16.1 fps and 72.3→56.2GB with clean output (no
  noise/NaN, identity preserved). `STREAM_FP8=off|blocks|all` in run_interactive.sh
  (default off). SageAttention still untested/blocked.
- [ ] Perf: the stream is capped at `--fps` by an explicit wall-clock pacer
  (`interactive_demo.py` "pace to wall clock", keeps ~1.6s lead) — faster
  generation does NOT raise stream fps, it buys SLACK in the 2.0s chunk budget.
  That slack is what stops micro-freezes. Measured per-chunk generation:
  bf16 416*720 = 2.13s (OVER the 2.0s budget → continuous starvation),
  FP8 416*720 = 1.99s (+0.5%, still stutters on any long chunk),
  FP8 320*576 = 1.16s (+42%, comfortable, visibly softer image).
  NOTE `--fps` must divide 16000 (asserted): ladder is 8/10/16/20/25, no step
  between 16 and 10. Open: measure 352*608 (836 tokens, untested — the sweep
  timed out); consider 320*576 @ fps 20 (1.6s budget, +27% slack, smoother
  motion); raise client `JITTER` (0.35s) to absorb the 2s burst cadence; and
  torch.compile is still untried.
- [x] Dedicated LingBot web studio: focused start-image → world flow, explicit
  model/WebRTC/warm-up states, full-size video stage, Beauty event controls,
  custom natural-language actions, and WASD/mouse camera input. Replaces the
  generic multi-pipeline editor at `/`; advanced editor remains at `/scope`.
  Production build + deployed HTTP assets verified; live H200 E2E verified
  (WebRTC connected, data channel open, seed accepted, event delivered, 13+16
  frames generated without runtime errors).

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
