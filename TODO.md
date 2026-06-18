# Daydream Scope — AI Persona TODO

## In Progress

- [~] Define product direction and update CLAUDE.md with persona vision
- [~] Research character consistency techniques (persistent latent conditioning, reference image anchoring, IP-Adapter)

## LongLive 2.0 (NVFP4) Integration

See plan: `.claude/plans/longlive2-nvfp4-integration.md`. New `longlive2` pipeline on Wan2.2-TI2V-5B
(text + image), NVFP4 W4A4 on RTX 5090. Native kernel phases run on the 5090, not macOS.

- [~] Phase 0: env/dep spike on Blackwell (RTX PRO 6000, sm_120) — kv_dequant ext BUILDS; fouroversix conflict ROOT-CAUSED (needs transformers>=5.0.0 for `WeightConverter`; entrypoint now upgrades it in NVFP4-only branch); transformer-engine dev headers now auto-installed by entrypoint. TE build + render still to validate on pod. See `docs/longlive2-runpod-bringup.md`.
- [x] Phase 1: scaffold `pipelines/wan2_2/` component layer — causal 5B model loads (825/825 keys match), VAE 2.2 (48ch) decodes
- [x] Phase 2: `pipelines/longlive2/` BF16 correctness gate PASSED — renders coherent video on GPU after fixing model.yaml load, latent channels (16→48), and denoising schedule
- [x] DMD timesteps: confirmed real 5B 4-step from upstream NVlabs/LongLive → `[1000, 946, 854, 681]` (was wrong assumed `[1000,750,500,250]`); set in model.yaml. 2-step s2 still unverified (no upstream sampling_steps:2 config; subsample = `[1000,681]`).
- [ ] Phase 3: NVFP4 path — local prep DONE (transformers pin, dev-header automation). REMAINING (pod): finish TE build, validate nvfp4-s2 render + measure fps, confirm umt5 tolerates transformers v5.
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
