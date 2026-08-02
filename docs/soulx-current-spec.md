# SoulX-LiveAct persona demo — current spec

State as of **2026-08-02**, branch `soulx-persistent-state`, RunPod 4×H200 pod.
Companion to `soulx-deployment.md` (how to deploy) — this is *what is deployed
and what it measures*. Numbers marked **[v]** were verified on this pod today;
anything else is inherited from earlier sessions and is labelled.

---

## 1. What it is

An interactive AI persona: a single reference image becomes a character who
speaks (Kokoro TTS, lip-synced) and performs natural-language actions, streamed
live to a browser. Built on [Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct)
— 18B = Wan2.1 14B video + 4B audio module, chunked autoregressive with a
persistent KV cache.

## 2. Running configuration [v]

```
scope-soulx run --res 368x640 --vae wan --gpus "3," --port 8090
```

Resolved plan, as logged by `soulx_runtime`:

```
[plan] size=368x640 vae=wan attn=fa3 fp8=blocks compile=blocks world_size=1
[attn] FlashAttention-3 enabled for self-attention (fa3F@2.8.0.post2-3-g3ba6f82)
[decoder] full Wan2.1 VAE (streaming causal cache) -- the reference
```

| property | value |
|---|---|
| resolution | 368×640 portrait (both dims must be ×16) |
| frame_len | 920 tokens = (640/16)×(368/16) |
| chunk latent | `(16, 8, 80, 46)` |
| generation | **1.69 s / 32-frame chunk = 18.9 fps = 1.18× realtime** [v] |
| VRAM | 69.9 GB, flat [v] |
| stream fps | 16 (must divide 16000 for the audio window maths) |
| chunk | 32 frames = 2.0 s of video; chunk 0 is 21 frames |
| attention | FlashAttention-3 (xformers-bundled; Hopper-only) |
| quantization | FP8 W8A8 on the 480 block matmuls (`--fp8 blocks`) |
| compile | `torch.compile` over the 40 DiT blocks, max-autotune-no-cudagraphs |
| warmup | 204 s (lightvae) / 545 s (wan — the VAE decode path compiles too) [v] |

**GPU layout** — generator GPU3, `persona_aux` (Qwen2.5-1.5B brain + Kokoro TTS)
GPU0, ports 8090 / 8091. `world_size=1`, so no NCCL and no sequence parallelism:
one 143 GB card holds the ~81 GB model whole.

> **This pod's GPUs are shared with containers outside our PID namespace** —
> GPU0 ~50 GB and GPU1 ~76 GB are held by processes invisible to `ps`. The
> planner picks by *total* VRAM, not free, so it must be pinned. `--gpus N`
> means a **count**; an explicit card index needs a trailing comma
> (`--gpus "3,"` = card 3, `--gpus 3` = cards 0,1,2).

## 3. Performance ladder [v]

All at 368×640, one H200, identical settings, chunks 30–90 idle, VRAM flat.
A chunk is 2.0 s of video, so 2.0 s/chunk is exactly 1.0× realtime.

| `--vae` | s/chunk | realtime | slack | notes |
|---|---|---|---|---|
| `taew2_1` | 1.45 | 1.38× | 0.55 s | Conv2D TAE, 23 MB, the default |
| `lightvaew2_1` (windowed) | 1.55 | 1.29× | 0.45 s | causal Conv3D, cold cache per chunk |
| **`lightvaew2_1` (streaming)** | **1.52** | **1.32×** | **0.48 s** | 68.2 GB |
| `wan` (windowed) | 1.79 | 1.12× | 0.21 s | full Wan2.1 VAE, the quality reference |
| **`wan` (streaming)** | **1.69** | **1.18×** | **0.31 s** | **current**, 69.9 GB |

**Quality verdict (user, on the live stream, 2026-08-02): `368x640` + `wan` with
the streaming causal cache is the best the demo has looked so far.** That is the
configuration to beat, and it is why the reference decoder — not a tiny one — is
what ships here. The tiny TAEs remain the option to reach for when a slower box
needs the 0.24 s/chunk back, not the default to optimize toward.

Two conclusions worth keeping:

1. **The full Wan VAE now streams above realtime.** The `reference` preset's
   "NOT realtime" note is stale — it predates FA3 + block compile. On this box
   the quality reference is affordable, so quality is no longer the thing being
   traded away for latency.
2. **The decoder microbenchmark overstates in-situ cost.** 452 ms of bench
   decode costs ~340 ms of wall time; 133 ms costs ~100 ms. Decode overlaps
   other work. Choose a decoder on quality, not on that table.

Decoder registry (`soulx_runtime.DECODERS`), MAE measured offline on real
dumped 52×90 latents vs the full Wan VAE:

| key | kind | ms | MAE/255 | size | status |
|---|---|---|---|---|---|
| `wan` | causal Conv3D | 452 | reference | 507 MB | **current**; ships with LiveAct |
| `taew2_1` | Conv2D TAE | 32 | 2.77 | 23 MB | default |
| `lighttaew2_1` | Conv2D TAE | 32 | 3.76 | 45 MB | **never run live**; dominated by taew2_1 |
| `lightvaew2_1` | causal Conv3D | 133 | 3.34 | 32 MB | the fast causal option |
| `v3` | Conv2D TAE | 31 | 6.41 | 25 MB | dropped — reconstructed ckpt, smears |

> GOTCHA the registry encodes on purpose: the two TAEs use OPPOSITE latent
> normalization. `taew2_1` needs `need_scaled=False` (True → MAE 24.48);
> `lighttaew2_1` needs `need_scaled=True` (False → MAE 21.56). Backwards, it
> looks like a broken model rather than a wrong flag.

## 4. Streaming causal decode [v, new today]

The Wan2.1 VAE is a **causal** 3D VAE (`CausalConv3d` + per-conv `feat_cache`),
and `lightvaew2_1` is the same class at `pruning_rate=0.75`. But `decode()`
clears the cache on entry *and* exit, so every chunk decoded cold, and the demo
faked the lost history: `decode(cat(pre_latent[:,-3:], latent))[:, :, 9:]` —
11 latent frames in, 41 out, 9 discarded.

`soulx_runtime.StreamingDecode` now calls lightx2v's
`cached_decode_withflag(zs, is_first, is_last)`, which clears only on `is_first`.
`interactive_demo.py` branches on `self.stream_dec` at both decode sites
(session loop + warmup) and drops the overlap; `reset()` on session start.

Both causal decoders are wired. `wan` has no registry entry to build — it
decodes through the very VAE object the demo constructs for `encode` — so its
wrapper goes around that instance, and the `torch.compile` target moves from
`decode` to `cached_decode_withflag` accordingly. **Not** applied under sequence
parallelism: `decode` splits 1D there while `cached_decode_withflag` only has a
2D-grid path, so it would not be the same computation.

Offline, 14 real consecutive dumped chunks (uncompiled), arms order-swapped:

| decoder | arm | steady ms | seam /255 | interior /255 | seam/interior |
|---|---|---|---|---|---|
| lightvaew2_1 | windowed | 190 | 3.06 | 1.28 | 2.40× |
| lightvaew2_1 | streaming | **148** | **2.33** | 1.22 | **1.92×** |
| wan | windowed | 641 | 3.35 | 1.44 | 2.33× |
| wan | streaming | **497** | **2.66** | 1.36 | **1.95×** |

Live: lightvae 1.55 → 1.52 s (−30 ms), wan 1.79 → **1.69 s (−100 ms)**.
Inter-arm MAE 0.5/255 (lightvae) and 0.95/255 (wan), against decoder-vs-Wan
error of 3.34 — no drift over 14 chunks either way.

- **Frame counts unchanged: 21 then 32.** A warm cache emits the full `T*4`
  because only a *cold* first latent frame costs the 3 warm-up frames — so A/V
  sync, which depends on those counts, is untouched.
- The live gain is smaller than the offline one (30 vs 42 ms, 100 vs ~114 ms)
  because this latent is 0.79× the dumped one's area and decode partly overlaps
  other work.

`SOULX_STREAM_DECODE=0` restores the windowed path on the same binary, so this
stays A/B-able without a redeploy. The TAEs (`taew2_1`, `lighttaew2_1`, `v3`)
are unaffected — they would each need their own cross-call cache.

## 5. Transport and rendering [v, fixed today]

WebSocket at `/ws`, framing: 1 byte type + f64le timestamp + payload.
`type 0` = JPEG frame (q82), `type 1` = PCM16 @ 16 kHz.

Measured on the live stream, 240 frames:

| metric | value | reading |
|---|---|---|
| media gap | 62.5 ms, p50 = p95 = max | the timeline is perfect |
| arrival gap | p50 **1.6 ms**, max **1961 ms** | one ~50 ms burst per 2 s chunk |
| stalls >150 ms | 8 per 240 frames | exactly one per chunk |
| duplicates / non-monotonic ts | 0 / 0 | nothing is being padded or reordered |
| chunk-grid, phase 31 | 1.67× mean frame delta | the seam is real but small |
| idle audio | ~1 PCM msg per chunk | audio flows even when she is silent |

**The server has always emitted a whole chunk as a burst and then gone quiet for
~1.95 s.** That is by design; the client is what has to smooth it.

`chat.html`'s render loop paces on the audio clock — but `tsOffset` is assigned
*only* inside the `type === 1` branch, i.e. only once the user enables sound.
The no-audio fallback used to run `drawImage(frames[last]); frames.length = 0`:
newest frame drawn, other 31 discarded. **One frame per chunk — a 0.5 fps
slideshow**, which is what "jerky"/"tics" meant for two days. Sound ON was
always smooth; sound OFF was always a slideshow.

Fixed: the fallback paces on a local `performance.now()` clock seeded from the
first frame's ts and drains on frame timestamps like the audio path, re-seeding
on >1 s desync in either direction (starvation, backgrounded tab, session
restart zeroing the timeline) or >90 queued frames. Plus
`TEMPLATES_AUTO_RELOAD` — a template edit was otherwise costing a full weight
load + compile warmup to see.

## 6. Control model

- **Chunked AR**: `blksz_lst=[6,8]` latent frames; 3 denoise steps/chunk; KV
  cache per step. Chunk 0 emits 21 frames, then 32/chunk.
- **Prompting follows Wan-Streamer v0.3 "world + event stream"**: a persistent
  pose-neutral `WORLD_PROMPT` (identity/scene) + a sticky sustained state
  (held posture) + a transient transition (motion verb, held `hold` chunks then
  dropped). Re-encoded only when the composed string changes. State lives in
  causal history, not in a re-imposed idle prompt.
- **Brain**: Qwen2.5-1.5B in `persona_aux.py` emits `{say, action, pose,
  hold_seconds}`. It frequently omits `pose`/`hold_seconds` and misgenders the
  character, so `_infer_posture()` derives a sustained pose + longer hold from
  motion keywords. Without that fallback, posture changes silently degrade to
  short gestures.
- **Lipsync**: wav2vec2 embeddings; per-chunk window `[(i-1)*32, (i-1)*32+53)`
  + 2 lookahead; audio tempo-stretched by `25/fps` (the model is 25 fps native).
- HTTP: `/chat`, `/say`, `/action`, `/persona`, `/session/{start,stop}`,
  `/status`, `/ws`, HLS under `/stream/live/`.

## 7. Access

- **Public**: Cloudflare quick tunnel →
  `https://females-brandon-coaches-intend.trycloudflare.com`
  (`./bin/cloudflared tunnel --url http://localhost:8090`). Random hostname,
  **no authentication**, dies with the process. Anyone with the link can start
  sessions, upload reference images, and consume the pod.
- **Local**: `ssh -f -N -L 8090:127.0.0.1:8090 soulx-pod` → `http://localhost:8090`.
- RunPod's `https://<podid>-<port>.proxy.runpod.net` does **not** work here —
  only port 22 is declared on this pod template, and the proxy's own 404 is
  easy to misread as "the service answered".
- `ssh.runpod.io` is PTY-only: no exec, no scp, no `-L`. Use the direct IP+port
  (`soulx-pod` in `~/.ssh/config`, key `id_runpod_new`).

## 8. Layout

**Repo** (`soulx-liveact-demo/`):
- `scope-soulx` — the single launcher (`run`/`doctor`/`fetch`/`bench`/`stop`/`logs`)
- `SoulX-LiveAct/soulx_runtime.py` — paths, GPU planner, decoder registry, `StreamingDecode`
- `SoulX-LiveAct/interactive_demo.py` — session server (Flask + flask-sock)
- `SoulX-LiveAct/persona_aux.py` — Qwen + Kokoro, separate process
- `SoulX-LiveAct/templates/chat.html` — the live-call UI
- `docker/` — CUDA 12.8 + torch 2.8 cu128 image
- `ultraflash/` — offline analysis harnesses (hardcode the pod's dump layout)

**Pod** (`/workspace/soulx/`): `weights/LiveAct` (51 GB) · `weights/chinese-wav2vec2-base`
· `decoders/` → symlink to `/workspace/uflash/ckpt_lx2v` · `assets/reference.png`
→ symlink to `/workspace/soulx_setup/chano39-portrait.png` · `env/` (conda, on
the network volume) · `env.sh` (reproduces activation; the conda *install* does
not survive a container recreate) · `logs/` · `bin/cloudflared`.

Diagnostics left on the pod: `jitter.py` (arrival vs media time),
`types.py` (message mix), `grab.py` (pull frames off the ws),
`stream_dec_test.py` (windowed vs streaming decode).

## 9. Known issues / open

- **Chunk seam remains** at 1.67–1.92× the interior frame delta. The decoder can
  no longer be the cause; the residual belongs to the generator's own chunk
  boundary (KV cache / latent side).
- **No authentication** on any endpoint, now publicly reachable.
- `lighttaew2_1` never run live; dominated by `taew2_1` on every axis.
- Reference image must be pre-framed for portrait — the demo centre-crops, and
  the stock 16:9 chano39 loses her face entirely at 9:16.
- `conda activate` exports `SIZE`/`AR`/`LD`/... — launcher overrides are named
  `STREAM_*` to dodge the collision. Do not rename them back.
- Kill by explicit PID or `scope-soulx stop` (which uses `[i]nteractive_demo.py`
  bracket-matching). A bare `pkill -f interactive_demo.py` kills your own ssh
  session, whose command line contains that string.

## 10. In the working tree but NOT deployed

A **SageAttention backend** (`--attn sage`, `--sage_scope self|all|self+cross`,
`model_liveact/sage_backend.py`, the `nosage` shim deleted) is uncommitted and
was deliberately not pushed to the pod, so today's measurements run on the same
attention stack that produced the earlier baselines. The pod runs `--attn fa3`.
