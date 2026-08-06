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

## Optional 2x latent upscaling (UltraFlash) — EXPERIMENTAL, to be reverted

Passing `--sr_url` to `interactive_demo.py` streams 1440x832 instead of 720x416
by routing each chunk latent through
[UltraFlash](https://github.com/xin1u/UltraFlash)'s SR cascade before decode.
Both models live in the Wan2.1 VAE latent space (16ch, stride 4/8/8), so the
handoff needs no pixel round-trip. `scope-soulx` does not expose it: the sidecar
needs a 4-GPU layout and the feature is expected to be reverted.

Measured on the 4xH200 pod: **1.67s per 32-frame chunk (19.2fps, 1.20x
realtime)** vs 1.99s (1.00x) without SR — it is *faster* despite 4x the pixels,
because the sidecar's tiny decoder (101ms) replaces SoulX's own Wan VAE decode
(451ms) and runs on a different GPU.

**The visible artifacts in the stream come from the super-resolution stage, not
from the generator.** The SR DiT was trained with AIGC-oriented degradation on
photoreal Wan output, so on flat cel-shaded anime it hallucinates pore/hair-grade
high-frequency texture where there is no real structure to recover — most
obviously a repeating cross-hatch weave on ambiguous regions (the hand under the
chin, the hair/cheek boundary). Identity, expression and colour are preserved;
it is the invented micro-texture that reads as wrong. Turning SR off (drop
`--sr_url`) removes them entirely.

This whole feature is experimental and expected to be reverted.

- The layout it was measured on was **4 GPUs**: GPU0 aux · GPU1 SR sidecar ·
  GPU2,3 SoulX. The sidecar could share a generator GPU (~20GB) at the cost of
  contending with it.
- `ultraflash/sr_service.py` — SR sidecar (upsampler + sparse SR DiT + decoder).
  Separate process on purpose: a 2nd CUDA context inside a torchrun rank
  destabilizes NCCL, same reason `persona_aux` is split out.
- `ultraflash/ultra_dec_v3.py` — the released `ultra-decoder-v3` weights do NOT
  load into UltraFlash's own decoder classes; this reconstructs the matching
  module list from the checkpoint's key/shape signature. 1755ms -> 101ms.
- `ultraflash/sr_probe.py`, `dec_bench.py`, `dec_shootout.py` — offline
  harnesses used to measure the above before touching the live demo. These
  still hardcode the pod's `/workspace/uflash` dump layout; they are analysis
  scaffolding, not part of the deploy path.

## Layout

- `SoulX-LiveAct/` — upstream repo **with our modifications**:
  - `interactive_demo.py` (ours) — continuous live session server. Flask +
    flask-sock. The T5 context follows Wan-Streamer v0.3's **world + event
    stream**: `world` (persistent, pose-neutral identity/scene) + sustained
    `state` (sticky held posture) + a transient `transition` (motion held
    `--action_hold` chunks then dropped). Posture changes update the sustained
    state so the character STAYS in the new pose (sit → keeps sitting) instead
    of snapping to idle; gestures don't persist. Routes: `/chat` (Qwen2.5-1.5B →
    `{say, action, pose}`; `pose` = resulting held state or null; `/change ...`
    edits the current reference image and restarts the causal session), `/say`
    (kokoro TTS → rolling 16 kHz audio buffer; silence = idle), `/action`
    (`{text, pose?, persist?}`; a `pose`/`persist` sticks, empty body clears to
    neutral idle), `/status`, `/ws` (realtime feed), plus a parallel HLS
    pipeline (ffmpeg) for recording.
  - `persona_aux.py` (ours) — isolated Qwen/Kokoro Flask service on physical
    GPU 0. The main distributed process calls it over loopback HTTP so auxiliary
    inference cannot consume generator VRAM or change its CUDA/NCCL context.
  - `templates/chat.html` (ours) — no video player: canvas + WebAudio client
    over WebSocket (~0.4 s latency). Chat panel, quick action buttons,
    Chat/Say/Do modes.
  - `model_liveact/model_memory_sp.py` (patched, 2 fixes):
    1. `AttnType.SAGE_FP8_SM90 → AttnType.FA` unless a verified sage kernel
       backs it (`_lca_attn_type()`); on sm90 it never does — see
       `sage_backend.py`.
    2. Per-forward `xFuserLongContextAttention(...)` instantiation replaced by
       a singleton (`_get_lca()`) — the per-call version leaked ~240 MB/chunk
       and OOM-killed sessions after ~10 min.
  - `model_liveact/sage_backend.py` (ours) — SageAttention as a scoped,
    self-verifying backend. Upstream turned sage on for every attention the
    moment the package was importable; it is now explicit (`--attn sage`),
    scoped like FP8 (`--sage-scope self|self+cross|all`, default `self`), and
    the chosen kernel is checked against SDPA on first use so a bad build
    degrades to a printed warning instead of a stream full of noise.
  - `soulx_runtime.py` (ours) — the one place that knows where files live and
    what the GPU can do: `SOULX_ROOT` path resolution, the decoder registry,
    and the planner that decides single-GPU vs sequence-parallel and gates
    FP8/FA3 on compute capability.
- `scope-soulx` — the launcher (`run` / `doctor` / `fetch` / `bench` / `stop` /
  `logs`). Replaces the six divergent `run_*.sh` scripts.
- `docker/` — `Dockerfile` + `compose.yaml`. CUDA 12.8 / torch 2.8 cu128 so the
  image runs on Hopper, Ada and Blackwell unchanged. Weights stay on a mounted
  volume; SageAttention is deliberately not installed (it is a dead end on
  Hopper — see below — and adds ~4 min to every build).
- `chano39-Anime-Original-anime-9101906.png` — the reference character image,
  copied into `$SOULX_ROOT/assets` on first container start.

## Run

```bash
scope-soulx doctor                          # GPUs, paths, and the chosen plan
scope-soulx fetch                           # weights + tiny decoders, once
scope-soulx run --res 416x720 --vae taew2_1 # server on 0.0.0.0:8090
```

### Transport: WebRTC or WebSocket

`--webrtc` serves the stream over WebRTC (H.264/Opus) **alongside** the default
MJPEG-over-WebSocket path, off the same emit worker, so one session serves both
and `?transport=ws` always gets you back to the older path on a network where
WebRTC cannot find a relay.

```bash
scope-soulx run --res 368x640 --vae taew2_1 --webrtc
# http://host:8090              -> WebRTC
# http://host:8090/?transport=ws -> WebSocket
```

Measured on the same 30s window, same stream, 368x640:

| | WebSocket (MJPEG+PCM16) | WebRTC (H.264+Opus) |
|---|---|---|
| video bitrate | 4.47 Mbps | **0.97 Mbps** |
| mean frame | 32.2 KB | 7.4 KB |
| audio | 0.27 Mbps | ~0.001 Mbps idle (Opus DTX) |
| inter-frame p50 / p99 | 11.3ms / **1603ms** | 62.6ms / **73.8ms** |
| stdev | 262ms | **2.7ms** |

The p99 is the point. The generator produces 2s of video at once, so the
WebSocket path ships 32 frames in a 0.4s burst and then goes quiet for 1.6s,
against a fixed 0.35s client jitter buffer — 0.35s of margin against 1.6s gaps
is what a micro-freeze is made of. WebRTC's RTP timestamps carry the
generator's media clock, so the receiver's jitter buffer paces off the real
timeline instead of arrival times.

**TURN is not optional off a RunPod pod.** RunPod maps TCP ports only, so no
browser can reach aiortc directly. Set `SOULX_TURN_CF_ID` / `SOULX_TURN_CF_TOKEN`
(Cloudflare's Turn Token ID and API token — *not* the friendly key name, which
returns 404) in `/root/.config/soulx/turn.env` and the server mints short-lived
ICE credentials per page load, refreshed an hour before expiry. A static
`SOULX_TURN_URL`/`_USER`/`_PASS` also works.

Use a relay that offers **UDP**. Running coturn on the pod itself only works over
TCP (nothing else is reachable), which puts every frame through one TCP
connection where a single loss stalls the stream head-of-line: measured as
latency spikes and ~12% of frames shed before the peer dropped entirely.
Cloudflare's relay is reached outbound over UDP by *both* peers, so the pod's
lack of UDP ingress stops mattering. `aiortc` gets the same relay
(`_aiortc_ice()`) — without it the server's only candidate is the container's
private address, which no public relay can route back to.

`webrtc_bridge.py` is the standalone A/B harness those numbers came from: it
subscribes to `/ws` like any other client, so it can measure a transport against
a running demo without touching it. Superseded by the in-process path, which
skips its JPEG decode/re-encode hop.

### How long an action lasts

A `transition` (a gesture) is held for `--action_hold` chunks and then dropped,
after which the sustained `state` prompt takes back over. **One chunk is 32
frames = 2s at 16fps**, so the default 8 is roughly 16 seconds.

```bash
scope-soulx run --action_hold 12                     # ~24s, at launch
curl -X POST :8090/action_hold -d '{"chunks":12}'    # live, no restart
curl -X POST :8090/action -d '{"text":"she stretches slowly","hold":20}'
```

`/action_hold` sets the running default and takes effect on the next action;
an action already in flight keeps the TTL it was given. The `hold` field on
`/action` overrides it for that gesture only — a slow stretch and a quick wave
want different lifetimes, so the caller can say rather than everything sharing
one number. Both are capped at 120 chunks (~4min); past that you want a
sustained `pose`/`persist` state, not a transition, since the sustained state
is what the KV cache carries forward.

### Appearance changes

Set `FAL_KEY` only in the server environment, then type a chat command such as
`/change your outfit to a wedding dress`. The server uploads the current
reference to `fal-ai/nano-banana/edit`, downloads and validates the edited
image, and queues it through the same ordered restart path as a manual image
upload. The loaded SoulX model stays resident; only the causal character
session restarts. Repeated changes use the latest edited image as their source.

`SOULX_APPEARANCE_MODEL` can override the fal endpoint. Do not put `FAL_KEY` in
the browser, source tree, launcher arguments, or a committed env file. The bare
pod launcher reads a mode-600 `/root/.config/soulx/secrets.env`; Docker
deployments should use their platform's secret or env-file support.

Because these are paid calls, one edit runs at a time. The process defaults to
a 20-second cooldown and 20 provider attempts; override with
`SOULX_APPEARANCE_COOLDOWN_SECONDS` and `SOULX_APPEARANCE_MAX_EDITS`.

Or in Docker, which is the supported deploy path:

```bash
docker build -t soulx-demo -f docker/Dockerfile .
docker run --gpus all -p 8090:8090 -v /workspace/soulx:/workspace/soulx \
  soulx-demo run --res 416x720 --vae taew2_1 --foreground
```

### LoRA

A kohya-format LoRA (`lora_unet_*` keys, as produced by sd-scripts/musubi) can
be merged into the DiT at load time:

```bash
scope-soulx run --res 368x640 --vae wan \
  --lora nsfw_wan_14b_revealing_boobs.safetensors --lora-strength 0.6
# a bare NAME resolves in $SOULX_ROOT/lora; a path is used as-is
```

It is **merged, not run as an adapter**, so the FP8Linear count, the compiled
graphs, and the VRAM profile are identical to a no-LoRA run and s/chunk stays
comparable. See `lora.py` for why, and for the load → merge → fp8 → compile
ordering that this depends on.

Strength is changeable on a live server, which matters because a warmup is
~10 minutes and a sweep would otherwise cost one warmup per point:

```bash
curl -X POST :8092/lora -H 'content-type: application/json' -d '{"strength":0.4}'
python lora_sweep.py --port 8092 --strengths 0,0.25,0.5,0.75,1.0 --tag rb
```

The change is applied at the next chunk boundary (never mid-forward: it
rewrites the bf16 master weights and forces an fp8 requantize). Loading a LoRA
switches FP8 weight storage to `cpu_offload` so the bf16 masters survive for
re-merging — ~28GB of host RAM, no VRAM cost. Single-GPU only; under sequence
parallelism each rank holds its own weights and a rank-0-only merge would
desync them, so the endpoint refuses.

Note the base DiT is distilled to a **4-step** schedule, while most published
Wan LoRAs are trained for many-step inference — expect useful strengths well
below 1.0, and sweep rather than assuming.

From the Mac (port 8090 is NOT exposed via RunPod's HTTP proxy — and its
`ssh.runpod.io` proxy cannot forward ports, so use the pod's direct IP):

```bash
ssh -N -L 8090:localhost:8090 -p <RUNPOD_TCP_PORT_22> root@<pod-ip>
# open http://localhost:8090 ; click "🔊 Enable audio" once
```

## Hard-won constraints (do not "simplify" these away)

- **fps must divide 16000** (audio sample rate): 16, 20, 25. We run 16 so the
  ~15 fps generation speed keeps up with playback (at 20 the stream drifts).
- **FP8 must stay scoped to the block matmuls.** `--fp8 all` wraps the
  modulation/embedding/head linears too — no FLOPs, maximum quantization
  sensitivity — and renders noise. `--fp8 blocks` (480 matmuls) is correct and
  is the default.
- **FA3 needs `XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`**, which the launcher
  exports. FA3 is reached through xformers, and xformers refuses to import when
  the installed flash-attn is outside its pinned range (`<=2.8.2`) — with 2.8.3
  on the pod that became a *silent* downgrade: `doctor` reported `attention:
  sdpa` on a Hopper card, the self-attentions ran ~1.9x slower, and nothing
  raised. The pin guards xformers' own FA2 bindings, which this demo never uses.
  It is exported at the top of the launcher rather than on the launch line
  because the plan probe is what resolves the backend.
- **SageAttention is a dead end on Hopper, for two independent reasons**
  (measured on H200 / torch 2.8 cu128 / CUDA 12.8, SageAttention v2.2.0 built
  from source with `arch=compute_90a,code=sm_90a`):
  1. Its Hopper kernel is *numerically broken here*.
     `sageattn_qk_int8_pv_fp8_cuda_sm90` returns rel err ~45 against an fp32
     SDPA reference at every shape, dtype and layout tried, and NaN with
     `smooth_k=False, smooth_v=False`. A clean source rebuild reproduced the
     error bit-for-bit, so this is not a build artifact. `sageattn()`
     auto-dispatches to exactly this kernel on sm90, which is where the old
     blanket "SageAttention renders noise" verdict came from.
  2. Its kernels that *are* accurate are the Ampere/Ada ones, and on Hopper
     they are slower than what we already run. At the 368x640 self-attention
     shape (q=2760, kv=5520, H=40, D=128): FA3 1.16 ms, SDPA 2.19 ms,
     flash-attn 2.8.3 2.27 ms, sage int8/fp16 2.28 ms, sage int8/fp8 2.92 ms.
     FA3 — already the `--attn auto` default on Hopper — is 2.5x the fastest
     correct sage kernel.

  Even a *fixed* sm90 kernel would be worth little: 40 blocks x 3 denoise steps
  x 1.16 ms is ~139 ms of a ~1960 ms chunk, so self-attention is ~7% of the
  budget. The block linears are what FP8 already attacks, and they dominate.
  `--attn sage` is kept because this verdict is per-architecture: on sm89/sm120
  sage is the fastest option, and the backend verifies before trusting.
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
6. Try plain `torch.compile` as the next performance lever. FP8 is on
   (`--fp8 blocks`) and SageAttention has now had its isolated numerical test —
   it fails on Hopper and is not the lever (see the constraints above).

Do not replace the working WebSocket transport merely for architectural purity.
It already has a shared media clock and audio-master playback. A Scope-native
WebRTC A/V path is worthwhile only if it preserves those properties and the
current latency.
