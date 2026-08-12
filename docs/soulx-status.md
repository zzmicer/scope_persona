# SoulX-LiveAct persona demo — status

**Read this first.** Where the demo is, what is proven, and what is left.
State as of **2026-08-12**, branch `soulx-persistent-state`, commit `dc645b5`,
RunPod 4×H200 pod `0oa7yefi1wg0v1`.

Numbers marked **[v]** were measured on the pod in the session that recorded
them. Anything else is inherited and labelled.

---

## Where to read what

There are five documents and they do not overlap. If you are lost, it is
probably because you want a different one than the one you are in.

| Doc | Answers |
| --- | --- |
| **this file** | What is the state? What is next? |
| `soulx-deployment.md` | How do I run it? Flags, Docker, env, HTTP API. |
| `soulx-current-spec.md` | What does the deployed system consist of? (detail, as of 2026-08-02 — the config section is superseded by §2 below) |
| `soulx-optimization-brief.md` | How do I make the model faster? Handoff for pruning/optimizing. |
| `soulx-1xh200-log.md` | How did 4 GPUs become 1? Session log of the three wins. |

Project-wide task state lives in `TODO.md` at the repo root; that is the source
of truth for what is in flight.

---

## 1. What it is

An interactive AI persona. One reference image becomes a character who speaks
(Kokoro TTS, lip-synced) and performs natural-language actions, streamed live to
a browser over WebRTC. Built on
[Soul-AILab/SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) — 18B
(Wan2.1 14B + 4B audio), autoregressive, motion controlled by per-chunk prompt
swaps over a persistent KV cache.

Since 2026-08-11 it is also a **public room**: many viewers share one stream and
take turns driving. See §4.

A chunk is 32 frames = 2.0s of video at 16fps, so **the budget is 2.0s/chunk**
and everything below is measured against it.

---

## 2. What is deployed right now

```bash
SOULX_PYTHON=/workspace/soulx/env/bin/python SOULX_PACER_LEAD=0.7 \
./scope-soulx run --only-gpu 3 --res 400x640 --vae flashvaed \
  --attn fa3 --fp8 blocks --compile blocks --webrtc --room --port 3002
```

Live at **http://213.181.104.236:11003** — no tunnel required.
Operator console at `/?console=1`.

**1.74 s/chunk against a 2.0 s budget = 1.15× realtime, 18.4 fps, 70.9 GB [v]**,
flat across 240 chunks. Room mode costs the video path nothing (timings are
identical with it on).

Only **GPU 3** is usable — cards 0–2 belong to other tenants. `--only-gpu 3` is
the correct flag; `--gpus 3` would mean *three cards*.

### Port map (RunPod maps TCP only)

| Public | Pod | What |
| --- | --- | --- |
| 11000 | 22 | ssh |
| 11001 | 3000 | **coturn** (TURN over TCP) |
| 11002 | 3001 | nginx — RunPod's own, do not use |
| 11003 | 3002 | **the app** |

---

## 3. Done — and what each result actually says

### Realtime on one H200 (2026-08-01, inherited)
3.64s → 1.96s per chunk at 720×416 from three wins that cost no quality: an
official tiny decoder, block-level `torch.compile`, and FlashAttention-3.
Full log in `soulx-1xh200-log.md`.

### The SR cascade, and why it is now off (2026-08-11)
UltraFlash latent SR ran at 1440×832 and was briefly *faster* than plain SoulX,
because the sidecar's tiny decoder replaced SoulX's own 451ms VAE decode. Then
the single-card stress test showed what it really costs when it cannot be
pipelined:

| stage | 2 cards | GPU 3 only | change |
| --- | --- | --- | --- |
| **SR DiT** | 508ms | **1752ms** | 3.4× |
| dec flashvaed | ~466ms | 752–840ms | 1.7× |
| **wall** | 1.47s | **2.79s** | 1.90× |

On two cards SR's 0.99s hid entirely behind the generator (wall = `max()`);
co-located it becomes `sum()` plus contention. **Memory was never the limit** —
52GB free, no OOM. Compute was.

**So SR was dropped, and that is the current config.** The comparison that
settles it: 288×512 ×2 SR measured 1.72 s/chunk — *the same wall time for a
smaller generated frame* (147k px vs 400×640's 256k px) **[v]**. Removing the
sidecar returned exactly the contention it was costing and buys a natively
larger frame instead of an upscaled one. It also drops the mosaic/cross-hatch
artifact for free: that is a documented property of UltraFlash's SR DiT (trained
with AIGC degradation on photoreal Wan output, so against flat cel-shaded anime
it invents pore/hair-grade detail), not a bug and not the generator's fault.

### WebRTC works off this pod (2026-08-11) — and it needed no code
Cloudflare TURN is a dead end here, permanently: `turn.cloudflare.com` resolves
IPv6-only and the container has no IPv6 path, so UDP never answers ALLOCATE, and
the TLS/443 fallback *does* allocate a relay but then answers CHANNEL_BIND with
401.

The fix is configuration. RunPod maps **TCP** ports, so run the pod's own coturn
(already installed) on one and dial it over TCP. `relay-ip` is the **container**
IP on purpose: the browser never contacts the relay directly — it reaches its
allocation over the TCP control connection — and the peer that does send to it
(aiortc) is on the same host. Confirmed by
`[webrtc] TURN relay via turn:213.181.104.236:11001?transport=tcp` **[v]**.

Two traps worth keeping: `SOULX_TURN_CF_ID`/`CF_TOKEN` must be **absent** from
`turn.env`, because `_cloudflare_ice()` is tried first and silently shadows the
static URL. And when hand-testing an ALLOCATE, RFC 5389 §15.4 says the length
field for MESSAGE-INTEGRITY covers the message up to and *including* MI and
excludes FINGERPRINT — getting it wrong returns a 401 indistinguishable from bad
credentials.

### The room (2026-08-11)
`--room` turns the demo into a link you can hand to strangers. Detail in §4.
**53 unit tests + 36 end-to-end against the live server [v]**, all passing.

---

## 4. The room — the shape, and why

One live session on one GPU is not a limitation to hide; it is the mechanic.
Everyone who opens the link lands in the **same already-running stream**. One
visitor at a time holds the seat for 90s and drives her; everyone else watches
and talks. The queue is the show.

This was chosen over private per-visitor rooms deliberately: the GPU stays at
100% instead of idling and paying a session restart per person; waiting is
entertaining because you watch a stranger discover it; "37 watching" carries a
rough demo where an empty room makes every flaw yours to notice; and one link
with no signup beats an email gate. Capacity: ~40 drivers/hour plus unlimited
watchers on one H200.

**Talking and driving have opposite access rules, on purpose.** Room chat is
open to everyone and never reaches the model — it is people talking to each
other around the stream. Driving is exclusive: `/action`, `/say`, `/chat`,
`/anchor`, `/action_hold`, `/lora`, `/persona` and `/session/*` now require the
turn token. They were wide open before this.

**A watcher's message *is* the suggestion.** There is no separate suggestion
box; the driver's client renders every room line with a one-tap replay, so
watchers type where they already type.

### Code layout
| File | Owns |
| --- | --- |
| `stage.py` | The seat: turn queue, tokens, no-show grace, outfit credits |
| `room.py` | The crowd: presence, open chat, flood control |
| `templates/room.html` | The public page (three screens in one document) |
| `templates/chat.html` | The operator console, unchanged, at `/?console=1` |

`stage.py` and `room.py` are stdlib-only and clock-injected, so the whole social
layer tests on a laptop with no model loaded.

### Rules encoded in the UI, each one earned
1. **Buttons first, free text second.** The palette holds only Wave / Heart /
   Wink / Laugh / Nod / Turn. Sit, stand and lie are *absent*: the I2V reference
   path (`img_emb`, `cross_attn.k_img/v_img`) is re-imposed every chunk and no
   prompt wording or LoRA strength reaches it, so whole-body re-posing silently
   does nothing. A dead button loses the user to a known limitation.
2. **Acknowledge in <100ms, render in ~1.7s.** Unacknowledged latency is what
   reads as broken; one chunk of delay is fine.
3. **Turn end clears the sustained pose.** A held state outlives the turn that
   set it, and decays anyway past ~30s (arm wardrobe drifts bare → sleeves →
   gloves).
4. **The outfit change is a ritual**: one token per turn, globally capped,
   because it is real money plus a session restart.
5. **No image uploads**, which deletes the entire likeness/consent surface.

`room.html` also **falls back to the WebSocket path** when a relay is
unreachable. `chat.html` still does not, so there a bad relay is a black screen.

---

## 5. Not done

### Launch blockers for a public link
- **Moderation of free text.** The one thing standing between this and a link
  you hand to strangers. Structurally smaller than it looks — room chat never
  reaches Qwen/T5, and only one driver per 90s can — but the driver's text does
  reach the model, and room chat is displayed to everyone.
- **`FAL_KEY` is unset**, so the 🎁 outfit ritual is inert (the button correctly
  hides itself). Set it server-side to enable.

### Product
- **Restore ~416×720**, the vertical quality preset. At ×2 SR on one card this
  is ~0.5× realtime, so it needs a *cheaper SR*, not a bigger budget.
- **Replace the SR stage.** Any replacement must consume Wan2.1 latents, be
  causal with a cross-call cache (or A/V sync drifts), and fit the leftover
  VRAM. Keep Flash-VAED as the HR decoder — it is registry-backed, so swapping
  SR should not disturb it.
- **The chat brain drops actions.** Qwen2.5-1.5B puts the motion text in `say`
  and leaves `action` null (0 of 10 on "wave at me"), so chat renders nothing
  visually. This is a *brain* problem, not latency — the video path is a healthy
  1.6s with no drift. Buttons work around it today; free-text driving does not.
- **Scaling: add rooms, not a pool.** room = 1 GPU = 1 character = 1 queue.
  Different characters per room is a content feature and the queue logic does
  not change.

### Operational
- Rotate `RUNPOD_API_KEY` (sits in the pod env) and the old Cloudflare TURN
  credentials (kept in `turn.env.cf-backup`).
- The pod's **root disk is 100% full**; everything must live on `/workspace`.
  Compile caches are already redirected there.

---

## 6. Gotchas that cost time before

- `--gpus N` is a **count**, not an index. Naming one card is `--gpus "3,"`, or
  just use `--only-gpu 3`.
- This pod's GPUs are **shared with other containers**; free VRAM is not yours.
- FA3 is reached *through xformers*. If the installed flash-attn is outside
  xformers' pinned range, xformers silently falls back to SDPA and the stream
  just runs ~1.9× slower with no error anywhere.
- Posture is held by the **state text**, not the reference image. The reference
  governs identity, wardrobe and scene; she relaxes to neutral within ~15s
  whatever it shows. A "pin this pose" must therefore *keep* the pose text.
- The wall-clock pacer means extra speed buys **slack in the 2.0s budget**, not
  a higher frame rate — and that slack is what prevents micro-freezes.
- Measure the websocket (arrival vs media time) before optimizing anything: the
  "jerkiness" once blamed on the pipeline was the frontend drawing 1 frame per
  chunk and discarding 31.

---

## 7. Admin

Kill switch, absent entirely unless `SOULX_ADMIN_TOKEN` is set in the server
environment (`/root/.config/soulx/admin.env` on the pod):

```bash
curl -X POST http://213.181.104.236:11003/room/admin \
  -H 'X-Admin-Token: <token>' -H 'content-type: application/json' \
  -d '{"do":"close"}'     # close | open | end | reset
```

`close` stops granting turns (watchers keep watching), `end` boots the current
driver, `reset` also clears the sustained pose back to neutral.
