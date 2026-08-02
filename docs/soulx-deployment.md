# SoulX-LiveAct persona demo — deployment runbook

Chat → speech + action → live video, as one container. The demo used to be six
shell scripts with `/workspace/soulx` and "the GPU is an H200" compiled into
them; it is now one launcher plus a Docker image that decides its own GPU
topology.

## Quick start

```bash
docker build -t soulx-demo -f docker/Dockerfile .          # from soulx-liveact-demo/
docker run --gpus all -v /workspace/soulx:/workspace/soulx soulx-demo fetch
docker run --gpus all -p 8090:8090 -v /workspace/soulx:/workspace/soulx \
  soulx-demo run --res 416x720 --vae taew2_1 --foreground
```

Then from your Mac — RunPod's `ssh.runpod.io` proxy **cannot** forward ports, so
use the pod's direct host:

```bash
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 8090:127.0.0.1:8090 root@<pod-ip> -p <RUNPOD_TCP_PORT_22>
open http://localhost:8090
```

Without Docker, `scope-soulx` works the same on a pod that already has the env:

```bash
SOULX_ROOT=/workspace/soulx ./scope-soulx run --preset quality
```

## The two dials

| flag | values | notes |
|---|---|---|
| `--res` | `416x720`, `368x640`, `336x592`, any `WxH` multiple of 16 | portrait is free — cost tracks pixel count, not orientation |
| `--vae` | `wan`, `taew2_1`, `lighttaew2_1`, `lightvaew2_1`, `v3` | `wan` is the full Wan2.1 VAE; the rest replace only the per-chunk decode |

Presets bundle both: `--preset fast` (368x640 + taew2_1), `quality` (416x720 +
taew2_1), `reference` (416x720 + wan, **not realtime**, for A/B only).

`vae.encode` is always the full Wan VAE — the reference image goes through it
whichever decoder is selected.

## What the launcher decides for you

`scope-soulx doctor` prints it. `--gpus auto` (the default):

- **one card that fits the model** → single GPU, no NCCL, `persona_aux` on the
  spare card if there is one
- **no card that fits** → sequence-parallel across two, `persona_aux` needs its
  own card (a second CUDA context inside a torchrun rank destabilises NCCL)
- **FP8** is enabled at `blocks` scope on sm_89+, silently off below
- **FlashAttention-3** is Hopper-only; on Ada/Blackwell the plan resolves to SDPA

Force it with `--gpus 2` / `--gpus 0,1`, `--fp8 off`, `--attn sdpa`.

## Measured latency, 1×H200

A chunk is 32 frames = 2.0s of video at 16fps, so **the budget is 2.0s/chunk** —
and comfortably under, because a wall-clock pacer keeps ~1.6s of lead and any
overrun shows up as a micro-freeze. Target ~1.8s.

| size | taew2_1 + compile | taew2_1 | wan + compile | wan |
|---|---|---|---|---|
| 720×416 | **2.50s · 0.80× ✗** | 3.01s · 0.66× | — | 3.64s · 0.55× |
| 640×368 | ~1.8s · 1.1× ✓ (est) | 2.18s · 0.92× | — | — |
| 592×336 | **1.42s · 1.41× ✓** | 1.78s · 1.12× | 1.74s · 1.15× ✓ | 2.19s · 0.91× |

Compile is worth 17–20% wherever measured. Generation scales ~linearly with
pixel count (slightly better, since attention is superlinear in tokens);
break-even is ~215k px. **720×416 does not stream live** even compiled — it is
the quality reference.

These numbers are **H200 with FA3 unavailable-or-not** — see the table header in
`docs/soulx-1xh200-log.md` for exactly which. On Blackwell they do not carry
over: FA3 is gone (Hopper-only) and the SM count/clocks differ. Re-run
`scope-soulx bench --sizes 416x720,368x640` on the target box.

A resolution change forces a **full recompile** (`max-autotune` benchmarks
kernels against actual shapes): ~8 min at 592×336, ~15 min at 720×416 before the
first chunk. The Docker image keeps `TORCHINDUCTOR_CACHE_DIR` on the mounted
volume, so that cost is paid once per resolution per pod, not once per container.
Restarting a *session* inside a live process is ~1s.

## Decoders

Measured against the full Wan2.1 VAE on real dumped latents:

| decoder | ms | MAE/255 | size | notes |
|---|---|---|---|---|
| `wan` | 452 | — | 507MB | full Wan2.1 VAE, the reference |
| `lightvaew2_1` | 133 | 3.34 | 32MB | 75% pruned, **causal Conv3D** — models time |
| `taew2_1` | 32 | **2.77** | 23MB | default |
| `lighttaew2_1` | 32 | 3.76 | 45MB | |
| `v3` | 31 | 6.41 | 25MB | reconstructed; visibly blurry on motion — do not use |

**MAE is a weak signal here.** v3 scores a respectable 6.41 yet smears visibly on
a fast-moving hand, which a per-frame average cannot see. Judge decoders on
moving video: `ultraflash/dec_video_shootout.py` decodes one set of dumped
latents through every candidate so the decoder is the only variable.

`lightvaew2_1` is the fallback if a Conv2D TAE ever smears — it is the only fast
candidate that actually models time.

### Gotcha: opposite normalization

The two lightx2v TAEs disagree on latent scaling, and the wrong flag looks like a
broken model rather than a wrong setting:

- `taew2_1` → `need_scaled=False` (True gives MAE 24.48)
- `lighttaew2_1` → `need_scaled=True` (False gives MAE 21.56)

This is encoded per-checkpoint in `soulx_runtime.DECODERS`; do not "fix" it to
one value.

## Layout

| Path | Purpose |
|---|---|
| `scope-soulx` | the launcher — `run`, `doctor`, `fetch`, `bench`, `stop`, `logs` |
| `docker/Dockerfile` | CUDA 12.8 / torch 2.8 cu128; runs on Hopper, Ada, Blackwell |
| `docker/compose.yaml` | same, with the volume + GPU reservation wired up |
| `SoulX-LiveAct/soulx_runtime.py` | paths, GPU planner, decoder registry |
| `SoulX-LiveAct/interactive_demo.py` | the server (Flask + WebSocket + HLS) |
| `SoulX-LiveAct/persona_aux.py` | Qwen + Kokoro TTS, separate process |
| `$SOULX_ROOT/weights/` | LiveAct + wav2vec2 (~40GB, on the network volume) |
| `$SOULX_ROOT/decoders/` | tiny decoder checkpoints |
| `$SOULX_ROOT/assets/` | reference images |
| `$SOULX_ROOT/logs/` | `demo.log`, `persona_aux.log` |

Paths come from `SOULX_ROOT` (default `/workspace/soulx`); override any of
`SOULX_WEIGHTS`, `SOULX_DECODERS`, `SOULX_ASSETS` individually.

## Environment

Docker is the supported path and needs no conda. On a bare pod, the constraint
that shaped the old scripts still applies: the conda **installer** lives on the
container root disk and is destroyed on recreation, while the **environment** on
the network volume survives. Point `SOULX_ENV_SH` at a script that restores
`PATH` and sources `activate.d`, and `scope-soulx` will source it.

Two traps that have each cost a run:

- **`set -u` is unsafe** after sourcing a conda env — `activate.d` references an
  unbound `NVCC_PREPEND_FLAGS`, and `PYTHONPATH` is often unset.
- **`SIZE` is clobbered** by conda's binutils activation
  (`SIZE=x86_64-conda-linux-gnu-size`), so any `${SIZE:-...}` silently picks up
  conda's value. Nothing here uses a bare `SIZE` any more.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | LLM decides `{say, action, pose}`; `/change <instruction>` edits the current reference and restarts its causal session |
| `POST /say` | speak exact text (Kokoro TTS, lip-synced) |
| `POST /action` | perform a motion; `{pose}` or `{persist:true}` makes it stick |
| `POST /session/{start,stop}` | restart the stream |
| `GET /status` | `{active, chunks, error, queued_speech_s}` |
| `WS /ws` | binary stream: `struct("<Bd", mtype, ts) + payload`, mtype 0 = JPEG, 1 = PCM16 |

Conversational appearance changes require `FAL_KEY` in the demo process
environment. The default endpoint is `fal-ai/nano-banana/edit`; override it
with `SOULX_APPEARANCE_MODEL`. Keep the key server-side—never expose it through
the page or pass it as a launcher argument. On the bare-pod layout,
`scope-soulx` reads an optional mode-600 `/root/.config/soulx/secrets.env` (or
`SOULX_SECRETS_FILE`); do not store it on a volume that ignores `chmod`. In
Docker, use the platform's secret/env-file support.

## Known open items

- **Nothing here has run on Blackwell yet.** The refactor gates FP8/FA3 by
  compute capability and the image is cu128, but the first 2×RTX PRO 6000 run is
  the verification. Start with `scope-soulx doctor`, then `--preset fast`.
- **vLLM's FP8 path on sm_120 is unverified.** `doctor` checks that
  `torch._scaled_mm` works; vLLM may dispatch a cutlass kernel instead. If the
  stream renders noise, `--fp8 off` isolates it immediately.
- **Output is slightly jerkier than the pre-optimization baseline**
  (user-reported, 2026-08-01) with taew2_1 + FA3 both active. Most likely the
  **pacing margin**: it was observed at 416×720, which runs at 1.02× realtime —
  0.04s of slack per chunk before chat and TTS add load. Prefer a config with
  real headroom (`--preset fast` = 1.38×) for anything user-facing. Second
  candidate is the decoder: every TAE here is **Conv2D** and models no time, so
  inter-frame jitter is architecturally possible and per-frame MAE cannot detect
  it; `--vae lightvaew2_1` is the fallback. Recorded clips are frame-driven and
  will NOT reproduce pacing jerkiness — judge it on the live stream only.
- **Lipsync under compile is unverified.** Block compile is a ~20% win and
  visually clean, but nobody has confirmed audio/video alignment survives it.
- On a shared pod the GPUs may be **held by other containers**. Benchmark only on
  a verified-idle GPU or the numbers lie.
