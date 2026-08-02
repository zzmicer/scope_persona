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
  model_memory_sp.py AttnType.SAGE_FP8_SM90→FA). [Both claims superseded
  2026-08-02: the noise was ONLY the sm90 kernel, not sage as a whole, and the
  shim is gone — see the SageAttention entry below. The FP8 GEMM claim was
  already superseded on 2026-07-27.] FP8 GEMM (vllm) also noise —
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
- [ ] Replace the SoulX demo's HLS transport with WebRTC + RTP-timestamp pacing.
  Today `interactive_demo.py` muxes to 1s HLS segments (`:840`) plus a side-channel
  WS feed of JPEG frames and PCM (`ws_pack`, `:262`), and paces by sleeping against
  wall clock with a hand-tuned 1.6s lead (`:1179`). Generation is already realtime
  (1.96s/chunk on 1xH200), so the 3-6s a user waits after "wave at me" is almost
  entirely transport, not the model. Reuse Scope's stack: `server/tracks.py`
  `VideoProcessingTrack` + `next_timestamp()` (`:79`), which stamps every frame with
  a 90kHz `pts` (`:156`) and derives `frame_ptime` from the measured pipeline FPS
  (`:133`) instead of a constant. Wins: the browser jitter-buffers on the stamps
  rather than stuttering on late frames; integer `timestamp +=` can't drift the way
  chained `sleep()` does; and A/V sync becomes expressible at all — sleep-based
  pacing has no way to say "this frame goes with that audio sample", which is why
  the demo currently needs ffmpeg to mux. That last point is the real reason to do
  this before lipsync audio moves off HLS.
  BLOCKED ON: Scope's WebRTC is video-only (no `AudioStreamTrack`/`AudioFrame`
  anywhere in `src/scope/server/` or the frontend hooks) — same `AudioProcessingTrack`
  gap already listed as OmniForcing Phase 4. Build it once, both pipelines use it;
  SoulX additionally needs the audio timeline (`drain_to`/`slice_abs`, `:963`) to
  keep feeding wav2vec sample-aligned. Keep the Flask/HLS path as the reference
  until WebRTC A/V sync is pod-verified.
- [ ] Bounded-queue + measured-FPS discipline in the SoulX demo. Independent of the
  WebRTC item above — needs no audio work, can land on the current HLS path.
  (a) Queue sizes: `vq`/`aq` are `maxsize=8` (`interactive_demo.py:900`), i.e. 8
  whole chunks ~= 16s of buffer, a constant unrelated to the 32-frame chunk; the
  `ws_broadcast` clients separately carry 256-slot queues with their own hand-rolled
  drop-oldest (`:255`). Two buffers, two unrelated sizes, two drop policies. Derive
  from chunk size the way `pipeline_processor.py:525` does
  (`num_frames * OUTPUT_QUEUE_MAX_SIZE_FACTOR`, resized when chunk size changes) and
  use one drop policy in one place.
  (b) FPS is assumed, not measured: the pacer trusts the `--fps 16` constant
  (`:1183`), so a chunk that takes 2.4s instead of 1.96s goes unnoticed and the
  hand-tuned 1.6s lead silently absorbs the error until it can't — then a micro-freeze.
  (This matches the FP8 finding already recorded in `docs/`: the speedup bought SLACK
  inside the 2.0s budget rather than fps, and that slack is what stopped the freezes.)
  Port `pipeline_processor.py:559-586`: timestamp each produced frame, keep the last
  30 INTER-FRAME DELTAS (deltas, not absolute times — a pause then leaves a gap
  without dragging the average down), FPS = 1/avg delta, clamped. Feed that measured
  value to the pacer instead of the constant, as `tracks.py:133` does.
  Payoff: the consumer adapts when the model slows instead of needing headroom to
  survive being wrong about the rate — i.e. removes the reason the 1.6s lead exists.
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
  UPDATE 2026-08-02 — the "sustained posture does not render" half is too broad.
  Recorded proof for an ARMS state (`/workspace/soulx/clips/state_persist.mp4`,
  50s off the websocket, 368x640): 12s baseline arms down matching the reference,
  action at 12.6s, transition expires at ~25.9s, and she holds arms-above-head
  for the remaining 24s with identity intact. So sustained state renders AND
  persists for upper-body/arm poses; what fails is whole-body re-posing
  (sit/stand/lie), where the reference conditioning wins. Arm wardrobe does drift
  over the hold (bare arms -> sleeves -> black gloves by 45s) while face, horns,
  tattoo and outfit stay fixed. Repro: `/workspace/soulx/state_persist.sh`.
- [x] The Do box can set a sustained state (2026-08-02). The "typed actions don't
  persist" complaint was never the hold duration: the composer sent `{text}` only,
  so the `pose` slot was idle on every chunk and every typed action was a gesture —
  only the Sit/Stand/Turn preset buttons ever passed a pose. `chat.html` now splits
  the Do input on `|` (`she sits down | she is sitting, relaxed`) into
  `{text, pose}`; no pipe keeps the old gesture behaviour, and the placeholder
  shows the form when Do is selected.
- [x] Sticky state visible without grepping chunk logs (2026-08-02). `/status`
  gained `pose: {state, transition, transition_left}`, mirrored out of the chunk
  loop at the compose site (so an expired transition shows up too) and cleared on
  session stop; the UI status line reads `Live · N chunks · <state>`.
- [x] Kohya LoRA support (2026-08-02, verified 1xH200). `SoulX-LiveAct/lora.py`
  merges `lora_unet_*` into the DiT (merge, never adapter — an adapter side-path
  would add un-quantized, un-compiled matmuls); merge point sits between the bf16
  cast and `enable_fp8_gemm`, order forced load -> merge -> fp8 -> cuda -> compile.
  `--lora NAME --lora-strength F`, plus `POST /lora {"strength":F}` to retune live
  (~25s). Inference cost is ZERO: 1.70s/chunk, 18.8fps, 69.9GB with and without.
  Both NSFW-API LoRAs key-match 100%.
- [ ] Reference-image test for wardrobe control. LoRA strength is exhausted as a
  lever: SoulX is I2V and the reference-image path (`img_emb`, `cross_attn.k_img/
  v_img`) has no LoRA counterpart in either T2V LoRA, so the reference image owns
  wardrobe. e15 @1.0 desaturates; 0.5-0.7 is the usable band. Next variable to
  change is `/change`, not strength.
- [ ] Controlled LoRA sweep with a restart per arm. `lora_ab.py --restart-each` is
  deployed but its run died. Every sweep so far ran in one continuous session, so
  pose/framing drift is NOT attributable to the LoRA — only "coherent vs degraded"
  is a safe claim from the existing data.
- [ ] Make `scope-soulx stop` port-scoped. It is currently a blanket pkill, so it
  cannot be used to retire one run while another is live.
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
- [~] Add conversational appearance changes: `/change <instruction>` edits the
  current reference image through a server-side external image provider, then
  restarts the causal SoulX session with the new reference while keeping the
  loaded model and persona configuration. Code/dependency/credential are staged
  on the pod; real fal wedding-dress edit visually verified; 9/9 unit tests.
  Final `/chat` -> session-restart verification awaits the next demo launch —
  current 8092/8094 LoRA runs both predate the patched source and remain active.
  Default editor switched from FLUX Kontext to Nano Banana
  (`fal-ai/nano-banana/edit`) on 2026-08-02.
- [ ] Authenticate the public SoulX control endpoints before treating paid
  appearance editing as a production service; the prototype currently bounds
  exposure with a single-flight call, cooldown, and per-process edit cap.
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
  (default off).
- [x] Perf: SageAttention v2.2.0 TESTED AND REJECTED on Hopper (2026-08-02).
  Two independent blockers, both measured on an idle H200 on the 4xH200 pod:
  (1) `sageattn_qk_int8_pv_fp8_cuda_sm90` — the kernel `sageattn()` auto-picks on
  sm90, and the source of the old "sage renders noise" verdict — is numerically
  broken here: rel err ~45 vs an fp32 SDPA reference at every shape/dtype/layout,
  NaN with `smooth_k=False,smooth_v=False`. Rebuilt v2.2.0 from source (`eb615cf`,
  correct `arch=compute_90a,code=sm_90a`, nvcc 12.8) and it reproduced BIT-FOR-BIT
  (45.31310), so it is a real v2.2.0 bug on this stack, not a bad build.
  (2) The sage kernels that ARE accurate are the Ampere/Ada ones and are slower
  than SDPA on Hopper. At the 368x640 self-attn shape (q=2760 kv=5520 H=40 D=128):
  FA3 1.16ms, SDPA 2.19ms, flash-attn2 2.27ms, sage int8/fp16 2.28ms, sage
  int8/fp8 2.92ms. FA3 (already `--attn auto` on Hopper) is 2.5x the best correct
  sage kernel. Even a FIXED sm90 kernel is worth little: 40 blocks x 3 steps x
  1.16ms = ~139ms of a ~1960ms chunk, i.e. self-attn is ~7% of the budget.
  SageAttentionFusion (QKV op fusion) inherits the same kernels — not worth trying.
  Kept as an opt-in backend because the verdict is per-architecture: new
  `model_liveact/sage_backend.py` makes sage explicit (`--attn sage`) and scoped
  (`--sage-scope self|self+cross|all`, default self, same reasoning as `--fp8`),
  and VERIFIES the chosen kernel against SDPA on first use before trusting it.
  The `nosage` PYTHONPATH shim is deleted — it only existed because the upstream
  `try: import sageattention` gate had no real off switch.
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
- [~] Perf: fit the demo on ONE H200 (2026-08-01). 4 GPUs is too much for a demo.
  Drop the SR stage but KEEP its tiny decoder — the SR build was faster than plain
  SoulX (1.62s vs 2.00s at 4x the pixels) only because the sidecar's decoder
  replaced SoulX's own 452ms Wan VAE decode. Measured baselines on this pod:
  1xH200 720×416 = 3.61s/chunk (0.55x realtime, 76.3GB); 2xH200 = 2.00s; 2xH200
  320×576 = 1.16s. Time scales ~linearly with pixel count (0.62x px → 0.58x time).
  - [x] Step 1 GATE: v3 tiny decoder verified at NATIVE res on real dumped SoulX
    latents (52×90) — MAE 6.41/255 vs the full Wan VAE, no colour cast, no
    structural break; 31.0ms vs 452.3ms = 14.6x, **saves 421ms/chunk**.
    `/workspace/uflash/dec_native_cmp.py`, image `cmp/v3_native_vs_wan.jpg`.
  - [x] Step 2: `--fast_decode` inlined into interactive_demo.py (no sidecar, no
    HTTP). One-line swap of `self.vae.decode` — v3's `frames_to_trim=3` gives
    `n_lat*4-3` frames, IDENTICAL to the Wan VAE, so the existing windowed decode
    and the 21-then-32 frame counts A/V sync depends on are unchanged.
    Patch `/workspace/uflash/add_fastdec.py`, backup `interactive_demo.py.bak4`.
  - [x] Step 3: `run_single_fast.sh` — no torchrun/NCCL, so persona_aux CO-LOCATES
    on the generator GPU (the "second CUDA context destabilizes NCCL" rule does
    not apply at world_size==1). t5 deliberately stays on GPU: 11GB fits in 143GB
    and `--t5_cpu` would stall the chunk loop on every action transition.
    VERIFIED: 720×416 = 3.01s/chunk flat over 50 chunks, 76.4GB.
    CORRECTION (2026-08-01): fast_decode saves 0.60s, more than the 421ms the
    isolated decode benchmark predicted. I first blamed a contended baseline —
    that was WRONG. A clean `FAST_DECODE=0` run on an idle GPU measured 3.64s,
    confirming the original 3.61s baseline. The swap really is worth ~0.6s at
    720×416 and ~0.41s at 592×336, i.e. MORE than the decode call itself costs;
    the surplus is unexplained (allocator/memory-pressure effects suspected).
  - [x] Step 4: resolution sweep on 1×H200 (2026-08-01) — **REALTIME REACHED**.
    Break-even is ~215k px; time falls slightly FASTER than pixel count.
    | size    | px     | s/chunk | realtime | mem    |
    | 720×416 | 299520 | 3.01    | 0.66x    | 76.4GB |
    | 640×368 | 235520 | 2.18    | 0.92x    | 67.8GB |
    | 592×336 | 198912 | 1.78    | 1.12x    | 62.8GB |  <- best quality w/ slack
    | 320×576 | 184320 | 1.61    | 1.24x    | 60.8GB |  <- portrait, most slack
    | 560×320 | 179200 | 1.57    | 1.27x    | 60.2GB |
    Logs `/workspace/soulx/sweep/*.log`, driver `sweep_res.sh`.
  - [x] Step 5: Pruna-style `torch.compile` of the 40-block `ModuleList`, tested
    at 592×336 on the same idle H200: 1.76s control → 1.42–1.43s steady state
    (**1.24x / ~19% latency reduction**), stable through 100 chunks. Warmup is
    expensive (469s) and first emitted chunk triggers another compile (6.59s),
    but the persistent service amortizes it. Visual gate passed: clean frame,
    identity preserved, and a wave directive rendered. Lipsync gate still needed
    before enabling in the production launcher.
  - [x] 20s action demos recorded at all 5 resolutions (2026-08-01), identical
    scripted content (4 actions + 2 TTS lines) fired on FRAME INDEX so the clips
    are directly comparable. `/workspace/soulx/demos/persona_*.mp4`; recorder
    `/workspace/uflash/record_demo.py` drives /action + /say over HTTP while
    draining /ws (mtype 0 = JPEG, mtype 1 = PCM16); driver `make_demos.sh`.
    Recorded realtime: 720×416 0.70x, 640×368 0.92x, 592×336 1.03x,
    560×320 1.06x, 320×576 0.98x. The last three are **PACER-CAPPED at ~1.0x**,
    NOT slower than the 1.12-1.27x benchmark — the wall-clock pacer caps output
    at `--fps`, so surplus speed shows up as slack, never as higher fps. This
    also means a recorded clip can never demonstrate >1.0x; read headroom off
    the sweep, not off a recording.
    320×576 portrait does NOT permanently crop the face as feared: only the
    opening frames are off-centre, then the character recomposes centred.
  - [x] Wan-VAE A/B clips (2026-08-01): `persona_{720x416,592x336}_wanvae.mp4`,
    FAST_DECODE=0. **Reverting to the Wan VAE costs live streaming**: 592×336
    goes 1.78s→2.19s (1.12x→0.91x, below realtime); 720×416 goes 3.01s→3.64s.
    CAVEAT: these clips compare CONFIGURATIONS, not decoders — each run is an
    independent generation, so the poses diverge and no decoder conclusion can
    be drawn from them. The decoder A/B is still only the still-frame test
    (same latents, MAE 6.41/255). A true TEMPORAL A/B is cheap and not yet done:
    decode the 14 consecutive chunks in `/workspace/uflash/dump/` through both
    decoders (no GPU generation needed) to check inter-frame flicker and seams
    at the 32-frame chunk boundaries where the windowed decode stitches.
  - [x] Decoder shootout vs `lightx2v/Autoencoders` (2026-08-01) — **DROP the
    reconstructed v3**. Measured on real dumped 52×90 latents, MAE vs full Wan VAE:
    | decoder                 | ms    | vs wan | MAE/255 | size  |
    | wan (reference)         | 452.1 | 1.0x   | -       | 507MB |
    | lightvaew2_1 (pruned 3D)| 132.9 | 3.4x   | 3.34    | 32MB  |
    | **taew2_1**             | 31.9  | 14.2x  | **2.77**| 23MB  |
    | lighttaew2_1            | 32.0  | 14.1x  | 3.76    | 45MB  |
    | v3 (ours, RECONSTRUCTED)| 31.4  | 14.4x  | 6.41    | 25MB  |
    **taew2_1 beats our v3 by 2.3x on error at identical speed**, is officially
    released, is the smallest, and loads through the ALREADY-INSTALLED
    `lightx2v...wan.vae_tiny.WanVAE_tiny` — so the reconstructed-checkpoint
    liability (rebuilt layer list, `block`→`up` rename, suppressed `v*2-1`) goes
    away entirely. v3 IS a TAEHV-family sibling of these (`hf/tae.py` has the same
    MemBlock/TGrow/`frames_to_trim=2**sum(...)-1`), just from UltraFlash.
    GOTCHA: the two TAEs use OPPOSITE latent normalization —
    `taew2_1` needs `need_scaled=False` (True → MAE 24.48) while
    `lighttaew2_1` needs `need_scaled=True` (False → MAE 21.56). Getting this
    backwards looks like a bad model, not a bad flag.
    `lightvaew2_1` is the fidelity option: causal Conv3D (so genuinely temporal,
    unlike the Conv2D TAEs) at 3.4x, worth testing where chunk-boundary/flicker
    behaviour matters. Ckpts in `/workspace/uflash/ckpt_lx2v/`,
    harness `/workspace/uflash/dec_shootout.py`.
  - [x] **SOLVED (2026-08-02): the jerkiness was the FRONTEND render loop, and
    all three hypotheses below were wrong.** User escalated to "really jagged,
    like she had tics"; measured the emitted websocket stream instead of
    guessing again (`/workspace/soulx/jitter.py`, 240 frames):
    - media timestamps are PERFECT: 62.5ms, p50 = p95 = max, zero non-monotonic.
    - arrivals are BURSTY: p50 1.6ms, max 1961ms, 8 stalls >150ms in 240 frames
      = exactly one per 32-frame chunk. The server emits a whole 2s chunk in
      ~50ms and then sends nothing for ~1.95s. It always did.
    - `chat.html`'s renderLoop has two paths, and the audio-clock one is only
      armed when `tsOffset` is set — which happens ONLY inside the `type === 1`
      PCM branch, i.e. only after the user enables sound. Without it the
      fallback ran `drawImage(frames[last]); frames.length = 0` — drawing the
      newest frame and DISCARDING the other 31. One frame per chunk: a 0.5fps
      slideshow of a 2s jump. That is the "tics".
    So: sound ON was always smooth, sound OFF was always a slideshow, and this
    is why the frame-driven recorded clips never reproduced it (noted below as a
    caveat and never followed up). Audio DOES flow when idle (~1 PCM msg/chunk,
    measured), so enabling sound is the instant workaround.
    FIXED: the fallback now paces on a local `performance.now()` clock seeded
    from the first frame's ts, draining on frame timestamps exactly like the
    audio path, with a symmetric re-seed (>1s desync either way, or >90 queued)
    for starvation, a backgrounded tab, and session restarts that reset the ts
    timeline to zero. Also `TEMPLATES_AUTO_RELOAD` — a template edit was costing
    a full weight-load + compile warmup to see.
    Residual, now bounded: the chunk-grid test puts phase 31 at 1.67x the mean
    frame delta (matching the 1.92x offline seam), but the 8 biggest jumps were
    NOT chunk-locked — they were real motion. The seam is real and small; it was
    never what "jerky" meant.
  - [-] SUPERSEDED by the entry above — kept because the reasoning was wrong in
    an instructive way: all three suspects were in the pipeline, and the bug was
    in the 40 lines of JS that put the pixels on screen. **USER-REPORTED: output
    is slightly more jerky ("дерганый") with the tiny VAE and/or FA3**
    (2026-08-01). Three candidate causes, in my order of suspicion:
    1. **PACING, not either change.** The config being watched when this was
       reported is 416×720 at **1.02x realtime = 0.04s slack per 2.0s chunk**,
       measured IDLE. Chat/TTS/prompt re-encoding add work the benchmark omits,
       and the wall-clock pacer starving is exactly what produces micro-freezes.
       Every earlier config viewed had 0.2-0.6s of slack.
    2. **The decoder.** taew2_1 (like v3 and lighttaew2_1) is a **Conv2D** TAE —
       it decodes frames independently and models NO time, so inter-frame jitter
       is architecturally possible and per-frame MAE cannot see it. This is the
       temporal question flagged and never closed.
    3. **FA3** — least likely: numerically equivalent to SDPA (max|delta| 5e-4),
       and it made things FASTER, which should mean smoother, not jerkier.
    CHEAP DISAMBIGUATION, do (a) first — it is one env var and no recompile
    beyond the usual:
    (a) same decoder + FA3 at **368*640 (1.45s, 1.38x, 0.55s slack)**. If the
        jerkiness disappears → cause 1, and the fix is to stop shipping 1.02x
        configs, not to change the decoder.
    (b) if it persists: `SOULX_ATTN=sdpa` at the same size → isolates FA3.
    (c) if it still persists: `FAST_DECODE=0` (Wan VAE) → isolates the decoder;
        if that fixes it, switch to **lightvaew2_1** (causal Conv3D, 133ms,
        MAE 3.34) which is the only candidate that actually models time.
    NOTE the recorded clips are frame-driven captures, so they will NOT reproduce
    pacing jerkiness — cause 1 can only be judged on the live stream.
  - [~] **RUNNING (2026-08-02): 368×640 + `lightvaew2_1` on 1×H200** — the config
    that attacks causes 1 and 2 together. `scope-soulx run --res 368x640 --vae
    lightvaew2_1 --gpus "3," --port 8090`, log
    `/workspace/soulx/logs/lightvae_368x640.log`. Measured over chunks 10-50:
    **1.55s/chunk = 20.6 fps = 1.29x realtime, 0.45s slack**, 67.7GB, flat.
    Cheaper than expected: the causal Conv3D decoder costs ~0.10s/chunk more than
    taew2_1 (368×640 + taew2_1 measured 1.45s) — 3x the slack of the 416×720
    config the jerkiness was reported on, with the only decoder that models time.
    Frames verified clean (identity, no cross-hatch, no cast). NEEDS the user's
    eyes on the LIVE stream — it is a pacing question, not a still-frame one.
    Because it moves both dials at once it cannot attribute the cause; if it is
    smooth and attribution matters, A/B `--vae taew2_1` at the same 368×640.
    LAUNCH GOTCHA: `--gpus N` means a COUNT, so an explicit card index needs a
    trailing comma (`--gpus "3,"` = card 3; `--gpus 3` = cards 0,1,2).
  - [x] **Decoder cost IN SITU at 368×640, 1×H200** (2026-08-02, fa3 + fp8 blocks
    + compile blocks, chunks 30-80 idle, 67.7GB flat in all three):
    | --vae         | s/chunk | realtime | slack | vs taew2_1 |
    | taew2_1       | 1.45    | 1.38x    | 0.55s | -          |
    | lightvaew2_1  | 1.55    | 1.29x    | 0.45s | +0.10s     |
    | **wan**       | 1.79    | **1.12x**| 0.21s | +0.34s     |
    **The full Wan2.1 VAE now streams above realtime** — the `reference` preset's
    "NOT realtime" comment is stale at this resolution (it was written when
    592×336 + wan measured 0.91x on the pre-FA3/pre-compile stack). The 452ms
    bench decode costs only 340ms of wall time here: decode overlaps other work,
    so the bench overstates every decoder's in-situ cost. Whole ladder from
    reference quality to 1.38x fits inside one flag with no re-plumbing.
    Warmup was 545s for wan vs 204s for lightvae (the VAE decode path compiles
    too; the block cache on /workspace only covers the DiT).
    Grabbed frames suggest the wan decode has more contrast and less of a
    magenta cast than lightvae — but they are different frames from different
    sessions, so that is an impression, NOT the controlled comparison. The
    controlled one is `dec_shootout.py` on dumped latents (MAE 3.34/255), and a
    global cast that visible would score worse; re-run it on the 368×640 latents
    before believing either reading. RESOLVED below: windowed vs streaming decode
    differ by 0.5/255, far too little to be that cast — it was session-to-session
    variation in the generation, not the decoder. Impression was wrong.
  - [x] **Streaming causal decode for `lightvaew2_1`** (2026-08-02) — we had a
    CAUSAL decoder and were calling it acausally. The Wan2.1 VAE is a causal 3D
    VAE (`CausalConv3d` + per-conv `feat_cache`) and `lightvaew2_1` is the same
    class at `pruning_rate=0.75`, but `decode()` calls `clear_decode_cache()` on
    entry AND exit, so every chunk decoded cold and `interactive_demo.py`
    re-approximated the lost history by hand: `decode(cat(pre_latent[:,-3:],
    latent))[:, :, 9:]` — 11 latent frames in, 41 out, 9 thrown away.
    lightx2v already ships the right call on the wrapper we hold:
    `cached_decode_withflag(zs, is_first, is_last)`, which clears only on
    `is_first`. Now `soulx_runtime.StreamingDecode` + `self.stream_dec` at both
    decode sites (session loop and warmup), with `reset()` on session start so a
    session never inherits the warmup's or a dead session's cache.
    OFFLINE, 14 real consecutive dumped chunks, order-swapped to rule out warmup:
    | arm      | steady ms | seam /255 | interior /255 | seam/interior |
    | windowed | 190       | 3.06      | 1.28          | 2.40x         |
    | stream   | 148       | 2.33      | 1.22          | **1.92x**     |
    Frame counts IDENTICAL (21 then 32) — a warm cache emits the full `T*4`
    because only a cold first latent frame costs the 3 warm-up frames, so A/V
    sync is untouched. Streaming vs windowed MAE 0.5/255 (the decoder's own
    error vs Wan is 3.34), i.e. no drift over 14 chunks.
    LIVE at 368×640: **1.55s → 1.52s/chunk (1.29x → 1.32x realtime)**, 68.2GB,
    flat over 80+ chunks, `32f` per chunk in the log confirming the contract
    holds in the running system. The live saving is smaller than the offline
    42ms because this latent is 0.79x the dumped one's area (80×46 vs 52×90) and
    decode partly overlaps other work — 33ms predicted, 30ms measured.
    NOTE the seam is REDUCED, not removed: 1.92x the interior delta remains, and
    the decoder can no longer be the cause of it. The residual belongs to the
    generator's own chunk boundary (KV cache / latent side). That is where the
    remaining "jerky" budget is, if any survives the pacing fix.
    Both arms run on ONE binary: `SOULX_STREAM_DECODE=0` restores the windowed
    path, so this stays A/B-able without a redeploy.
  - [x] **Streaming decode extended to `--vae wan`, and that config SHIPS**
    (2026-08-02). Wan gains more than lightvae did, because the 27% overlap tax
    was levied on a bigger decode: offline 641→497ms, seam 3.35→2.66/255
    (2.33x→1.95x), inter-arm MAE 0.95/255, frame counts still 21-then-32.
    **Live: 1.79 → 1.69s/chunk (1.12x → 1.18x realtime), 69.9GB flat.**
    USER VERDICT on the live stream: **best quality so far** — 368×640 + `wan`
    streaming is now the configuration to beat, and the deployed default.
    Two wiring differences from lightvae, both load-bearing: (1) `wan` has no
    registry entry to build, it decodes through the SAME VAE object the demo
    builds for `encode`, so the wrapper goes around that instance and the
    `torch.compile` target moves from `decode` to `cached_decode_withflag` —
    compiling the old entry point would have silently compiled nothing;
    (2) gated OFF under sequence parallelism, because `decode` splits 1D there
    while `cached_decode_withflag` only has a 2D-grid path, so on SP it would
    not be the same computation. Moot at world_size=1, would have bitten the
    next 2×H200 run. The TAEs would each need their own cross-call cache.
    Full current-state spec: `docs/soulx-current-spec.md`.
  - [~] Pruna optimization spike: block compilation is a confirmed win above.
    FA3 is BLOCKED on the current Torch 2.8 stack: Pruna publishes CUDA 12.8
    kernels for Torch 2.10/2.11 or stable-ABI 2.9+, and a locally built minimal
    Torch-2.8/SM90 BF16 hdim-128 extension failed its import/execution gate. Do
    not upgrade Torch in-place because that risks vLLM FP8 and the working ABI.
    Revisit FA3 in an isolated Torch>=2.9 environment; Pruna's open-source FORA
    remains Flux-only and Wan Taylor/auto caching remains Pro-only.
  - [-] 2×5090 REJECTED: per-rank weights are ~50GB (DiT 35 + umt5 11 + CLIP 4.5)
    vs 32GB VRAM, needing t5-offload + CLIP evict + FP8 *storage* (today's FP8 is
    compute-time only). And the payoff is poor: ~210 vs ~495 TFLOPS bf16, 1.79 vs
    4.8 TB/s, and NO NVLink — the 2xH200 scaling was near-linear (1.81x) only
    because SP collectives ride NVLink; over PCIe expect 1.3-1.5x. Net ≈ 1xH200
    after weeks of sm_120 dep work. 1xH200/1xH100 is the better demo box.
  - NOTE: this pod's GPUs are SHARED with other containers (PIDs not in our
    namespace held GPUs 0/2/3, one pegged at 100%). Only GPU1 was free.
- [x] Portability refactor (2026-08-02) — the demo ran only on the box it was
  built on. Removed: `pruna_bench.py` (a 1567-line fork of `interactive_demo.py`
  that differed by 30 lines of block-compile, now merged as `--compile blocks`);
  the `ultraflash/add_*.py` source-rewriting patchers (fast_decode, tiny_decoder,
  FA3, SR — all folded into the tracked source, so there is nothing to re-apply
  on a new pod); and six divergent `run_*.sh`/`deploy_demo.sh`/`setup_*.sh`
  launchers, replaced by one `scope-soulx` (`run`/`doctor`/`fetch`/`bench`/
  `stop`/`logs`). New `SoulX-LiveAct/soulx_runtime.py` owns the three things that
  were hardcoded: paths (`SOULX_ROOT`), the decoder registry, and a planner that
  picks single-GPU vs sequence-parallel from actual VRAM and gates FP8 (sm_89+)
  and FA3 (sm_90 ONLY) on compute capability. Flags now: `--res` / `--vae` /
  `--gpus` / `--fp8` / `--attn` / `--compile`.
- [ ] **Verify on 2× RTX PRO 6000 Blackwell (sm_120, 96GB)** — nothing below has
  run on Blackwell; this is the gate on the whole refactor. In order:
  1. `scope-soulx doctor` — expect: single-GPU plan (96GB > ~82GB), aux on GPU1,
     `attn=sdpa` (FA3 is Hopper-only), `fp8=blocks`, and the `torch._scaled_mm`
     self-test passing.
  2. `scope-soulx run --preset fast` (368×640 + taew2_1), then `--preset quality`
     (416×720). Read s/chunk from chunk 20+, not chunk 0 (compile warmup).
  3. A/B the two dials the refactor exposes: `--vae wan` vs `taew2_1`, and the
     two vertical resolutions. `scope-soulx bench --sizes 416x720,368x640`.
  4. **Watch for silent FP8 corruption**: vLLM's `Fp8LinearOp` may dispatch a
     cutlass kernel that is untested on sm_120. If the stream renders noise,
     `--fp8 off` isolates it in one flag. This pipeline has shipped
     plausible-looking corruption twice; check identity drift across chunks, not
     just per-frame sanity.
  5. Expect the H200 s/chunk table NOT to transfer: no FA3, different SM
     count/clocks. Re-measure before quoting any realtime multiple.
  Note this does NOT contradict the 2×5090 rejection below — that was 32GB cards
  needing t5-offload + FP8 *storage*. At 96GB the model fits one card whole.
- [ ] NVFP4 is now live as an option (sm_120 has it; Hopper did not) and
  `SoulX-LiveAct/fp4_gemm.py` already carries an unused `FP4Linear`. Same scoping
  rule as FP8: block matmuls only. See `docs/soulx-optimization-brief.md` §7 and
  the fouroversix notes in `docs/longlive2-runpod-bringup.md` Session 2.
- [ ] Docker image is written but UNBUILT (`docker/Dockerfile`, cu128 + torch 2.8,
  no conda, SageAttention deliberately omitted). First build will surface pin
  conflicts — most likely `vllm==0.11.0` vs the torch pin, and LightX2V's
  `setup_vae.py`.
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
