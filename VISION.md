# Product Vision: Real-Time Interactive AI Persona

> **North star:** Daydream Scope becomes a real-time, voice-and-chat-controlled
> digital character system — you upload a picture, pick a voice, and talk to a
> living, streaming video persona that answers you, follows your directions
> ("wave", "stand up", "look sad"), and never drifts or collapses, for as long
> as the session runs.
>
> The concrete reference for what "done" looks like is **Vidu S1 / Vidu Stream**
> (ShengShu Technology & Tsinghua, arXiv:2607.03118, July 2026,
> https://www.vidu.com/vidu-stream). This document specifies the target product
> in Vidu S1's terms and maps it onto this codebase.

---

## 1. The Reference: What Vidu S1 Is

Vidu S1 is a real-time interactive video generation model for voice-controlled
digital characters. It is the proof that the product we want is feasible today:

- **Speech is a direct, explicit control signal.** The user speaks ("wave your
  hand", "make a heart with both hands", "sit down") and the character performs
  the action *while the stream continues*. Control happens **at any moment
  during generation**, not up-front before generation begins.
- **Infinite-length streaming without drift.** The video runs indefinitely with
  no blurring, identity drift, or visual collapse — their teaser shows a stable
  90+ minute session with the same character.
- **Real-time on consumer GPUs.** 540p (960×540) at up to **42 FPS on an
  RTX 5090** with a 3-step distilled generator — above the 30 FPS live-playback
  threshold, on hardware users can own.
- **Custom characters from a single image.** Real people, anime, pets. Plus
  voice selection: professional presets or cloning the user's own voice.
- **Joint audio+video generation with lip sync.** The model generates a fused
  video-audio state per frame, so the character speaks with synchronized lips
  (best-in-class Sync-D 7.847 on HDTF), preserves identity (CSIM 0.9192), and
  won 100% of controllability preference tests vs. HeyGen and LemonSlice.

The Vidu Stream product wraps this in: character creation (photo upload + voice
pick/clone), a live session with camera/mic permissions (two-way perception —
the character can see and hear you), short-term persona memory, and an API. The
hosted API transports user/bot media over Aliyun RTC and uses a separate
authenticated WebSocket for session control; it is not a simple video HTTP API.

## 2. The Product Experience (what we ship)

The end state of Daydream Scope, as a user journey:

### 2.1 Create a character
1. Upload one reference image (photo, anime frame, pet picture) — or pick a
   template character.
2. Pick a voice: preset library, or clone from a short recording (future).
3. Optionally set a personality / system prompt (the conversational layer).

### 2.2 Start a live session
4. The character appears immediately in a WebRTC video stream, idle but alive —
   breathing, blinking, small natural motion. Target: first frame within a few
   seconds, then continuous ≥24 FPS (stretch: 30–42 FPS) at 540p.
5. The user talks (mic) or types (chat panel). Two-way perception: the system
   hears the user; camera input is a future extension.

### 2.3 Interact — the core loop
6. **Directing:** "raise your left hand", "give a thumbs up", "turn around",
   "look thoughtful." The instruction takes visible effect within a small,
   bounded latency (target: next generation chunk, well under ~2 s), while the
   stream never stalls or cuts.
7. **Conversing:** the user chats; an LLM decides what the character *says* and
   *does* — the reply is rendered as synchronized speech + lip movement +
   matching expression/gesture in the same continuous stream.
8. **Persistence:** minutes or hours in, the character still has the same face,
   outfit, and environment. No re-rolls, no resets, no drift.

### 2.4 Everything runs as it does today
- WebRTC streaming from the FastAPI backend, React frontend, Electron desktop
  app. A single-GPU local deployment is a first-class target, same as Vidu S1's
  "low-cost consumer GPU" positioning.

## 3. Target Specifications

| Dimension | Vidu S1 (reference) | Our target |
|---|---|---|
| Resolution | 540p (960×540) | 540p first; 480×832 acceptable interim |
| Frame rate | 25 FPS product floor, 42 FPS peak (RTX 5090) | ≥24 FPS sustained; stretch 30–42 |
| Session length | Model: infinite; hosted API currently caps a call at 600 s | Unbounded local runtime; ≥30 min stable model milestone |
| Control | Voice instructions, any moment, mid-stream | Text chat + action commands first, voice next |
| Latency to visible action | Next chunk (real-time) | Instruction reflected within 1 generation chunk (< ~2 s) |
| Identity | CSIM 0.9192, single reference image | No perceivable identity drift across a session |
| Audio | Joint AV generation, lip sync (Sync-D 7.847) | Character speech with lip sync (phase 2) |
| Hardware | Consumer GPU (RTX 5090 class) | Single 32 GB-class GPU local; datacenter GPU interim |
| Denoising steps | 3 (DMD + PCM distilled) | Few-step (2–4) distilled generator |

## 4. Capability Pillars (and how Vidu S1 builds each)

These are the five properties the final product must have. For each, the paper
describes a concrete mechanism we should treat as the default blueprint.

### 4.1 Live mid-stream control
User instructions are conditioning for *future* frames over a persistent
generation state — not a new generation. Vidu S1 conditions each autoregressive
step on all conditions available up to that frame (`c^{≤i}`: speech, text,
reference image), so a new spoken instruction simply changes the conditioning
sequence from that point on. This is the same family of mechanism we already
verified in **LingBot-World** (prompt swaps over a persistent KV cache) and
**SoulX-LiveAct** (per-chunk prompt swaps).

### 4.2 Infinite streaming without drift
Sliding-window decoding with three attention components:
1. a **persistent reference context** — latent tokens of the user's reference
   image + the first generated state, fixed for the whole session (the "sink
   frame", analogous to LLM sink tokens) providing stable global conditioning;
2. **cached history** within the sliding window;
3. the current state being denoised.

Plus two stabilizers: **RoPE repositioning** (cache K/V pre-RoPE, re-apply at
current relative positions so positions never leave the trained range) and
**TwinCache** — keep *both* a noisy cache (preserves coarse temporal dynamics,
suppresses accumulated high-frequency artifacts; used at intermediate denoising
steps) and a clean cache (restores fine detail and identity; used at the final
step). Decoupling temporal propagation from appearance refinement is their
answer to the consistency-vs-motion tension we've hit with VACE.

### 4.3 Persistent character identity
Identity comes from the always-attended reference context (4.2.1), not from
per-chunk re-injection at constant strength. This is the strongest available
answer to our open VACE design question: anchor identity in the attention
context once, and let motion be governed by history + conditioning.

### 4.4 Real-time on consumer hardware
Three-stage training to get quality *and* speed:
1. **Bidirectional teacher** on full video-audio sequences (quality prior);
2. **Causal teacher** — causal attention mask, hybrid Teacher Forcing +
   Diffusion Forcing (Bernoulli-sampled per sample) so it tolerates its own
   imperfect prefixes at inference;
3. **DMD distillation with PCM regularization** down to ~3 steps (PCM's
   perceptual-distance consistency term prevents the mode collapse / drift
   that DMD alone causes).

Serving stack (their TurboDiffusion/TurboServe route): SageAttention /
SpargeAttention / Sparse-Linear Attention, per-block W8A8 quantized GEMMs,
Triton/CUDA kernel fusion, CUDA Graph replay, Ulysses context parallelism for
multi-GPU. Our LongLive-2 NVFP4/FourOverSix work is the same playbook.

### 4.5 Speaking persona (joint audio+video)
Vidu S1 generates a *joint* per-frame state `x = [video; audio]` under a unified
conditioning interface (speech, text, reference images), which is what makes
lip sync native rather than bolted on. Our **OmniForcing** (LTX-2 causal AV)
bring-up is exactly this class of model and is the audio-capable candidate in
this repo.

## 5. Mapping to This Codebase

What we already have vs. what the vision still needs:

**Have (assets to build on):**
- WebRTC real-time streaming server, pipeline registry, lazy `PipelineManager`,
  React/Electron frontend (`src/scope/server/`, `src/scope/core/pipelines/`).
- Autoregressive streaming video pipelines: LongLive, LongLive-2 (NVFP4
  few-step, quantized — our 4.4 track), OmniForcing (joint AV — our 4.5 track),
  LingBot-World (mid-stream prompt-swap control over persistent KV — our 4.1
  track).
- A working **SoulX-LiveAct persona baseline** (`soulx-liveact-demo/`): a
  single-image anime character, continuous 720x416 generation at ~15 FPS on
  2xH200, mid-stream action/expression prompt swaps, Qwen conversation/action
  planning, Kokoro speech, lip sync, and a timestamped low-latency WebSocket
  A/V client. This is the closest current implementation of the Vidu S1 product
  loop and should be productized before starting another persona scaffold.
- VACE reference conditioning (identity track, to be superseded/informed by the
  sink-frame + TwinCache approach in 4.2/4.3).

**Need (gaps, roughly in order):**
1. **Productize the SoulX baseline behind Scope interfaces**: preserve its
   distributed worker/session model and working timestamped A/V transport while
   moving conversation state and structured directives behind stable provider
   boundaries. Do not force the 2-GPU runtime into the existing synchronous
   `Pipeline.__call__()` abstraction if that would break continuous sessions.
2. **Vidu-style interaction UX**: microphone input and streaming ASR first, then
   reference-image upload, template/voice selection, and a one-click live call.
   Text chat remains available for accessibility and debugging. Camera-based
   user/emotion perception follows after the audio path is stable.
3. **Structured conversational layer** (`core/persona/`): conversation manager
   + LLM action interpreter emitting `{ action, expression, dialogue }`, with a
   small validated action vocabulary and explicit current-state transitions.
4. **Scope-native A/V transport**: either generalize the proven SoulX
   timestamped WebSocket protocol or add WebRTC audio. Keep one shared media
   clock and audio-master playback behavior.
5. **Measure the north-star gaps**: instruction-to-motion latency, identity and
   quality curves over >=30 minutes, lip sync, and recovery behavior. The
   currently observed 15 FPS on 2xH200 is a prototype result, not the product
   real-time target.
6. **Close model/runtime gaps**: safe `torch.compile` first; only revisit
   SageAttention/FP8 after isolated numerical validation. In parallel, evaluate
   sink/reference context, RoPE repositioning, and TwinCache-style dual caches
   for long-horizon stability and a distilled/smaller model for one-GPU use.

### Hosted Vidu S1 reference

As of 2026-07-20, the official Vidu S1 repository publishes the paper, playable
product, API documentation/quickstart, and a `vidu-s1-api` agent skill, but no
model implementation or weights. Treat its API as an optional hosted reference
backend and evaluation oracle. SoulX-LiveAct remains the reproducible open/local
implementation path; do not make the core persona schema depend on either
provider. The current API requires an Aliyun RTC client plus a server-side
control-WebSocket proxy and hard-caps each API call at 600 seconds, so seamless
session renewal would be required to compare longer user experiences.

## 6. Success Criteria (how we know we've built it)

Modeled on Vidu-StreamBench (their in-house benchmark: 500 samples of
{action instruction, reference first frame, audio clip}):

1. **Controllability:** a battery of action instructions ("wave", "thumbs up",
   "sit down", "look sad") each visibly and correctly executed mid-stream,
   judged by A/B preference. Vidu S1 hit 100% preference vs. HeyGen/LemonSlice
   here — this is the axis that defines the product.
2. **Identity:** CSIM-style face/subject similarity vs. the reference image
   stays flat over a ≥30 min session (reference bar: 0.92).
3. **Sync (phase 2):** Sync-D-competitive lip sync when the character speaks.
4. **Quality:** DOVER-style perceptual score stable over time — no degradation
   curve.
5. **Real-time:** sustained ≥24 FPS at 540p on a single consumer-class GPU,
   with instruction-to-visible-action latency under ~2 s.
6. **Experience test:** a naive user can upload a photo, start a session, and
   hold a directed conversation with the character for 10 minutes without
   restarts, prompts, or timeline editing.

## 7. Non-Goals

- Offline/one-shot clip generation as a product focus (the existing timeline
  editor may remain as an advanced tool, but it is not the persona product).
- Multi-character scenes, camera-path cinematography, or general world
  simulation (LingBot-World remains a research track feeding 4.1, not the
  product).
- Training a foundation model from scratch — we adapt/distill existing open
  autoregressive video (and AV) models, following the Vidu S1 recipe at the
  post-training stage (causal adaptation → few-step distillation).
- Photoreal deepfake tooling of specific real people without consent;
  content-safety filtering of reference uploads is in scope.

## 8. References

- **Vidu S1: A Real-Time Interactive Video Generation Model** — Zhang et al.,
  ShengShu Technology & Tsinghua University, arXiv:2607.03118 (July 2026).
- **Vidu S1 repository** — https://github.com/shengshu-ai/Vidu-S1 (product/API
  links and agent skill; no released model code or weights as of 2026-07-20).
- **Vidu S1 API integration guide** —
  https://github.com/shengshu-ai/vidu-s1-api (Aliyun RTC media, control
  WebSocket, current 600-second call cap).
- **Vidu Stream product** — https://www.vidu.com/vidu-stream (540p/25–42 FPS
  live characters, voice commands, photo-based character creation, voice
  cloning, API).
- Internal: `CLAUDE.md` (persona architecture direction), `TODO.md` (task
  state), `docs/vace.md`, `docs/longlive2-runpod-bringup.md`, LingBot-World and
  OmniForcing pipeline docs under `src/scope/core/pipelines/`.
