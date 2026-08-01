# SoulX-LiveAct persona demo — deployment runbook

Runs the interactive persona (chat → speech + action → live video) on **one H200**.
Everything lives on the pod's network volume under `/workspace/soulx`; nothing here
is in git except this document.

## Quick start

```bash
ssh soulx-pod
cd /workspace/soulx
STREAM_SIZE="592*336" setsid nohup bash deploy_demo.sh > deploy.log 2>&1 &
```

Then from your Mac:

```bash
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 8090:127.0.0.1:8141 soulx-pod
open http://localhost:8090
```

The RunPod `ssh.runpod.io` proxy **cannot** forward ports — use the direct host
(`root@<ip> -p <RUNPOD_TCP_PORT_22>`, aliased as `soulx-pod`). See
[[runpod-access]] notes.

## Default configuration

`deploy_demo.sh` defaults to the configuration chosen after benchmarking:

| Component | Setting | Why |
|---|---|---|
| Decoder | **taew2_1** tiny decoder | 32ms vs the Wan VAE's 452ms, MAE 2.77, official 23MB checkpoint |
| Compile | block-level `torch.compile` | ~20% latency off; 40 blocks compiled individually |
| Quantization | FP8 W8A8, `--fp8_scope blocks` | 480 block matmuls only |
| Text encoder | umt5 resident on GPU | 11GB fits in 143GB; CPU offload would stall action transitions |
| persona_aux | co-located on the generator GPU | `world_size==1` → no NCCL, so the process-split rule does not apply |

Overrides are env vars: `STREAM_SIZE`, `TINY_DECODER`, `COMPILE`, `FAST_DECODE`,
`STREAM_FP8`, `GEN_GPU`, `STREAM_PORT`, `AUX_PORT`.

## Measured latency, 1×H200

A chunk is 32 frames = 2.0s of video at 16fps, so **the budget is 2.0s/chunk** —
and comfortably under, because a wall-clock pacer keeps ~1.6s of lead and any
overrun shows up as a micro-freeze.

| size | tiny+compile | tiny | wanVAE+compile | wanVAE |
|---|---|---|---|---|
| 720×416 | **2.50s · 0.80× ✗** | 3.01s · 0.66× | — | 3.64s · 0.55× |
| 640×368 | ~1.8s · 1.1× ✓ (est) | 2.18s · 0.92× | — | — |
| 592×336 | **1.42s · 1.41× ✓** | 1.78s · 1.12× | 1.74s · 1.15× ✓ | 2.19s · 0.91× |

Compile is worth 17–20% wherever it has been measured (3.01→2.50 at 720×416;
1.76→1.42 and 2.19→1.74 at 592×336).

Generation time scales **~linearly with pixel count** (slightly faster than
linear, since attention is superlinear in tokens). Break-even is ~215k px.

**720×416 does not stream live** even with both optimizations — it is the quality
reference. For live use pick 592×336, or 640×368 if compile holds up.

**Portrait is free.** Latency tracks pixel count, not orientation: `336*592`
measured identically to `592*336` (1.73s both, Wan VAE + compile). So for a
vertical deployment just swap the numbers — `368*640` for vertical 640p.

Note a resolution change forces a **full recompile** (`max-autotune` benchmarks
kernels against the actual shapes), so budget ~8 min at 592×336 and ~15 min at
720×416 before the first chunk. Restarting a *session* inside a live process is
cheap by comparison (~1s first chunk) because inductor's kernel cache persists.

## Decoders

`--tiny_decoder` selects what `--fast_decode` uses. Measured against the full
Wan2.1 VAE on real dumped latents:

| decoder | ms | MAE/255 | size | notes |
|---|---|---|---|---|
| wan (reference) | 452 | — | 507MB | full Wan2.1 VAE |
| lightvaew2_1 | 133 | 3.34 | 32MB | 75% pruned, **causal Conv3D** — models time |
| **taew2_1** | 32 | **2.77** | 23MB | default |
| lighttaew2_1 | 32 | 3.76 | 45MB | |
| v3 | 31 | 6.41 | 25MB | reconstructed; visibly blurry on motion — do not use |

**MAE is a weak signal here.** v3 scored a respectable 6.41 yet smears visibly on
a fast-moving hand, which a per-frame average cannot see. Judge decoders on moving
video: `/workspace/uflash/dec_video_shootout.py` decodes one set of dumped latents
through every candidate so the decoder is the only variable.

`lightvaew2_1` is the fallback if a Conv2D TAE ever smears — it is the only
candidate that actually models time.

### Gotcha: opposite normalization

The two lightx2v TAEs disagree on latent scaling, and the wrong flag looks like a
broken model rather than a wrong setting:

- `taew2_1` → `need_scaled=False` (True gives MAE 24.48)
- `lighttaew2_1` → `need_scaled=True` (False gives MAE 21.56)

This is encoded in the `_NEEDS_SCALE` map in the patched harness; do not "fix" it.

## Files on the pod

| Path | Purpose |
|---|---|
| `deploy_demo.sh` | **the launcher** — start here |
| `run_single_fast.sh` | older single-GPU launcher (stock compile toggle) |
| `sweep_res.sh`, `make_demos*.sh` | resolution sweep + demo recording drivers |
| `SoulX-LiveAct/pruna_bench.py` | harness with `--compile_blocks`; what deploy runs |
| `SoulX-LiveAct/interactive_demo.py` | original harness (no block compile) |
| `/workspace/uflash/ckpt_lx2v/` | taew2_1 / lighttaew2_1 / lightvaew2_1 checkpoints |
| `/workspace/uflash/record_demo.py` | scripted 20s clip recorder (drives /action + /say) |
| `/workspace/uflash/dec_shootout.py` | decoder MAE/latency benchmark |
| `/workspace/uflash/dump/`, `dump_wave/` | dumped latents for offline decoder tests |

Patches are reversible: `interactive_demo.py.bak4` (fast_decode),
`*.bak_tinydec` (decoder selection).

## Environment

`env.sh` no longer needs conda. The conda **installer** lived on the container
root disk and is destroyed on container recreation; the **environment**
(`/workspace/soulx/env`) is on the network volume and survives. `env.sh` now sets
`PATH` and sources `activate.d` directly. Original kept as `env.sh.conda-bak`.

Two traps that have each cost a run:

- **`set -u` is unsafe** after sourcing `env.sh` — conda's `activate.d` references
  unbound `NVCC_PREPEND_FLAGS`, and `PYTHONPATH` is often unset. Leave `set +u` on.
- **`SIZE` is clobbered** by conda's binutils activation (`SIZE=x86_64-conda-linux-gnu-size`),
  so `${SIZE:-...}` silently keeps conda's value. Use `STREAM_SIZE`.

## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | LLM decides `{say, action, pose}` |
| `POST /say` | speak exact text (Kokoro TTS, lip-synced) |
| `POST /action` | perform a motion; `{pose}` or `{persist:true}` makes it stick |
| `POST /session/{start,stop}` | restart the stream (resets latent dumping) |
| `GET /status` | `{active, chunks, error, queued_speech_s}` |
| `WS /ws` | binary stream: `struct("<Bd", mtype, ts) + payload`, mtype 0 = JPEG, 1 = PCM16 |

## Known open items

- **Output is slightly jerkier than the pre-optimization baseline** (user-reported,
  2026-08-01), with taew2_1 + FA3 both active. Most likely cause is not either
  change but the **pacing margin**: it was observed at 416×720, which runs at
  1.02× realtime — 0.04s of slack per chunk, measured idle, before chat and TTS
  add load. Prefer a config with real headroom (368×640 = 1.38×) for anything
  user-facing. Second candidate is the decoder: every TAE here is **Conv2D** and
  models no time, so inter-frame jitter is architecturally possible and per-frame
  MAE cannot detect it; `lightvaew2_1` (causal Conv3D) is the fallback.
  Full disambiguation plan in `TODO.md`. Note recorded clips are frame-driven and
  will NOT reproduce pacing jerkiness — judge it on the live stream only.

- **Lipsync under compile is unverified.** Compiling the 40 blocks is a ~20% win
  and visually clean, but nobody has confirmed audio/video alignment survives it.
  This is the gate before compile becomes the unconditional default.
- The pod's GPUs are **shared with other containers** — PIDs outside our namespace
  have held GPUs 0/2/3. Benchmark only on a verified-idle GPU.
- Compile warmup is expensive (~469s of Triton autotune, plus a slow first chunk).
  Sessions restarted within a live process are cheap; process restarts are not.
