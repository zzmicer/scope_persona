# SoulX-LiveAct Interactive Persona Demo

Chat with an anime character rendered in real time: she speaks (kokoro TTS,
lip-synced) and performs motions from natural-language prompts. Built on
[Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct)
(18B = Wan2.1 14B + 4B audio module), verified E2E on 2×H200 (2026-07-17).

This folder is a snapshot of the working deployment at
`/workspace/soulx` on the RunPod 4×H200 pod.

The current UI is a Vidu-style live-call prototype: character creation (image,
name, personality, Kokoro voice), one-click join/start, full-screen generated
video, live captions, browser speech recognition, quick actions, chat, and an
optional local camera preview. Camera pixels are not yet interpreted by the
persona; the preview deliberately labels that limitation.

## Layout

- `SoulX-LiveAct/` — upstream repo **with our modifications**:
  - `interactive_demo.py` (ours) — continuous live session server. Flask +
    flask-sock. Routes: `/chat` (Qwen2.5-1.5B → `{say, action}`), `/say`
    (kokoro TTS → rolling 16 kHz audio buffer; silence = idle), `/action`
    (T5 prompt-context swap at the next chunk boundary, held `--action_hold`
    chunks, then reverts to idle), `/status`, `/ws` (realtime feed),
    plus a parallel HLS pipeline (ffmpeg) for recording.
  - `persona_aux.py` (ours) — isolated Qwen/Kokoro Flask service on physical
    GPU 0. The main distributed process calls it over loopback HTTP so auxiliary
    inference cannot consume generator VRAM or change its CUDA/NCCL context.
  - `templates/chat.html` (ours) — no video player: canvas + WebAudio client
    over WebSocket (~0.4 s latency). Chat panel, quick action buttons,
    Chat/Say/Do modes.
  - `model_liveact/model_memory_sp.py` (patched, 2 fixes):
    1. `AttnType.SAGE_FP8_SM90 → AttnType.FA` — SageAttention sm90 kernels
       produce noise/NaN on this stack (torch 2.8 cu128, H200).
    2. Per-forward `xFuserLongContextAttention(...)` instantiation replaced by
       a singleton (`_get_lca()`) — the per-call version leaked ~240 MB/chunk
       and OOM-killed sessions after ~10 min.
- `nosage/sageattention.py` — import shim; putting `nosage/` first on
  `PYTHONPATH` makes `import sageattention` fail so every fallback path uses
  SDPA/flash-attn instead of the broken sage kernels.
- `setup_download.sh` — HF weight download (51 GB LiveAct + wav2vec2) to
  `/workspace/soulx/weights`.
- `setup_env.sh` — conda env build on `/workspace/soulx/env` (root disk is
  full on the pod; everything must live on /workspace). Installs torch 2.8
  cu128 stack, vllm 0.11, SageAttention v2.2 (built but unused — see above),
  LightX2V VAE, nvcc 12.8 via conda (system nvcc is 12.4).
- `post_env.sh` — TTS/LLM extras: `misaki[en]`, espeakng-loader, prefetches
  hexgrad/Kokoro-82M + Qwen/Qwen2.5-1.5B-Instruct.
- `env.sh` — env activation sourced by launchers. Notable:
  `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1` (xformers caps flash-attn at 2.8.2,
  we run 2.8.3), caches redirected to /workspace.
- `run_interactive.sh` — the demo launcher (SoulX GPUs 2,3 + persona GPU 0 ·
  port 8090 · 720×416 · fps 16 · `--no_fp8_gemm --no_compile --autostart`).
- `run_stock.sh` — upstream GUI demo launcher (for comparison).
- `chano39-Anime-Original-anime-9101906.png` — the reference character image
  (pod path: `/workspace/soulx_setup/`).

## Run (on the pod)

```bash
bash run_interactive.sh          # server on 0.0.0.0:8090, session autostarts
```

From the Mac (port 8090 is NOT exposed via RunPod's HTTP proxy):

```bash
ssh -N -L 8090:localhost:8090 -p 13539 -i ~/.ssh/id_runpod_new root@213.181.104.236
# open http://localhost:8090 ; click "🔊 Enable audio" once
```

## Hard-won constraints (do not "simplify" these away)

- **fps must divide 16000** (audio sample rate): 16, 20, 25. We run 16 so the
  ~15 fps generation speed keeps up with playback (at 20 the stream drifts).
- **No FP8 GEMM, no SageAttention** on this stack — both render noise. Flash
  attention 2.8.3 + SDPA are the working combination. `torch.compile` is off
  (`--no_compile`); plain-mode compile is the untested next perf step.
- **Latent RNG must be rank-synchronized**: earlier rank0-only Kokoro/LLM work
  consumed the global CUDA RNG; per-chunk latents therefore
  come from a dedicated seeded `torch.Generator` (identical on both ranks) —
  otherwise the two sequence-parallel halves diverge and half the frame turns
  to static.
- **Keep Qwen/Kokoro outside the distributed process**: `persona_aux.py` sees
  only physical GPU 0; torchrun sees only GPUs 2,3. Co-locating auxiliary models
  with SoulX rank 0 exhausted the headroom needed for T5 action encoding, while
  exposing a third CUDA device inside rank 0 destabilized NCCL. The loopback
  service avoids both failures and delivered a 1.85s warm chat+TTS response.
- Sessions auto-restart on error (≤3 consecutive attempts) in
  `control_loop_rank0`.
- **Idle heartbeat is load-bearing**: rank 0 broadcasts a `None` heartbeat every
  15 s while no session runs; without it rank 1's gloo recv times out after
  30 min idle and the whole server dies (`control_pg` also gets a 48 h timeout).

## WebSocket feed protocol (`/ws`)

First message: JSON meta `{fps, sr, w, h}`. Then binary messages:
`1B type + f64le timestamp + payload`, type 0 = JPEG frame (quality 82),
type 1 = PCM16 mono @16 kHz. Timestamps share one media clock; the client
uses audio as master and maps it to the canvas clock (self-healing offset).
Slow clients drop oldest messages server-side (queue max 256).

## Product role: the open Vidu S1 baseline

The Daydream Scope north star is the Vidu S1/Vidu Stream experience: create a
character from one image, start a live call, speak or type instructions at any
time, and receive a stable, lip-synced response stream with optional camera-based
perception. As of 2026-07-20, Vidu publishes a playable product and API but no
model code or weights. Its API uses Aliyun RTC for bidirectional media, a
separate control WebSocket, and currently caps a call at 600 seconds. This SoulX
demo is therefore the primary reproducible open/local baseline, not a disposable
side experiment.

What this baseline already proves:

- one reference image can drive a continuous talking/acting character;
- dialogue, action, TTS, lip sync, and low-latency A/V playback can run as one
  live session;
- prompt changes can take effect at chunk boundaries without restarting the
  generator;
- the distributed session survives idle periods and sustained generation once
  the attention-object leak and control-channel timeout are fixed.

The remaining Vidu-goal gaps, in priority order:

1. Integrate the runtime as a provider behind Scope's persona/session contracts;
   keep the generic conversation/action schema independent of SoulX and Vidu.
2. Add browser microphone input plus streaming ASR, feeding the same interpreter
   as text chat. The deployed browser SpeechRecognition prototype proves the UX;
   it still needs a provider-neutral streaming ASR backend. Text and Chat/Say/Do
   controls remain useful fallbacks.
3. Add a character setup flow (reference upload, template, voice selection)
   instead of fixing the session to the chano39 image and one voice. Image upload
   and Kokoro voice selection are deployed; template gallery and content-safety
   validation remain.
4. Add camera-to-conversation perception after the voice path works; a local
   preview/permission flow is deployed, but no camera pixels leave the browser.
   Visual context must not couple directly to the SoulX model runtime.
5. Run a >=30-minute benchmark recording action latency, lip sync, FPS, memory,
   identity similarity, and perceptual-quality drift.
6. Try plain `torch.compile` as the next performance lever. SageAttention and
   FP8 remain disabled until isolated numerical tests prove them safe on this
   stack.

Do not replace the working WebSocket transport merely for architectural purity.
It already has a shared media clock and audio-master playback. A Scope-native
WebRTC A/V path is worthwhile only if it preserves those properties and the
current latency.
