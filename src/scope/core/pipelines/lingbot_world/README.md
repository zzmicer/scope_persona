# LingBot-World-V2 — interactive text-controlled world

Support for [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2)
(LingBot-World-Infinity): a Wan2.2-based 14B causal world model that generates an
interactive world from a single image. Camera motion is conditioned on per-latent-frame
framewise pose deltas (Plücker embeddings); scene content and events are driven by the
text prompt; generation is chunk-by-chunk (4 latent frames) over a rolling KV cache
(`local_attn_size` window + attention sink), which is what makes the horizon unbounded.

## What's here

- `actions.py` — text commands → camera trajectories. "walk forward", "turn left",
  "orbit around her", "look up", "stay", with speed/duration modifiers. Text that is
  not a motion command (or is prefixed `scene:`/`event:`/`prompt:`) becomes an event
  prompt swap (cross-attn cache refresh) — e.g. "she smiles and waves at the camera".
- `session.py` — `LingbotWorldSession`: turn-based generation with persistent KV
  cache, adapted from upstream `_generate_causal_fast`. Each turn: new poses (+
  optional new prompt) → denoise chunks (4 steps, no CFG) → windowed VAE decode
  (2-latent overlap) → new frames.
- `demo.py` — CLI demo. Pass a start image + world prompt, then steer with text
  (interactive stdin or `--script` file). Rewrites the output mp4 after every turn.

## Running (H200/H100-class GPU, ~70GB VRAM in use)

On the pod, with the upstream repo and weights in `/workspace`:

```bash
pip install -r /workspace/lingbot-world-v2/requirements.txt  # torch>=2.4 assumed
pip install flash-attn --no-build-isolation                  # or a prebuilt wheel
hf download robbyant/lingbot-world-v2-14b-causal-fast \
    --local-dir /workspace/lingbot-world-v2-14b-causal-fast

PYTHONPATH=/workspace/scope_lingbot python scope/core/pipelines/lingbot_world/demo.py \
    --lingbot-repo /workspace/lingbot-world-v2 \
    --ckpt-dir /workspace/lingbot-world-v2-14b-causal-fast \
    --image girl.jpg \
    --prompt "A cheerful young woman stands on a mountain trail..." \
    --out session.mp4 \
    --script commands.txt        # omit for interactive stdin
```

Measured on 1×H200 (2026-07-12): ~4.5s per 4-latent-frame chunk at 480×832
(≈3.5 video-fps), model load ~60s, session VRAM ~70GB, no offload.

## Command language

| input | effect |
| --- | --- |
| `walk forward for 3 seconds` | camera moves forward (speed: slow/normal/fast) |
| `turn left`, `look up` | yaw / pitch |
| `strafe left`, `move right` | lateral translation |
| `orbit around her` | orbit: sideways translation + counter-yaw |
| `stay` | hold camera (lets prompt events play out) |
| `she waves at the camera` | event: prompt swap, camera idles 2s |
| `scene: a dragon lands on the ridge` | explicit event prefix |

## Reproducing `beauty.mp4`

The Scope UI shows an **Event Proposals** panel for `lingbot-world`, matching the
reference interaction: press/click **1** Run Through Hair, **2** Rest Chin In
Hands, and **3** Hold Candle. **F** adds a butterfly and **G** blankets the room
with snow. Event prompts are composed with the persistent character/world prompt
instead of replacing it, which keeps identity, clothing, camera framing, and the
bedroom anchored while the action or object changes.

For a direct-to-MP4 GPU check without WebRTC, use the included script:

```bash
python src/scope/core/pipelines/lingbot_world/demo.py \
  --lingbot-repo /workspace/lingbot-world-v2 \
  --ckpt-dir /workspace/lingbot-world-v2-14b-causal-fast \
  --image /assets/beauty_seed.png \
  --prompt "A beautiful young woman with long dark hair in a black ribbed sweater sits beside a blue bed in a dim bedroom, intimate close-up fixed camera" \
  --script src/scope/core/pipelines/lingbot_world/beauty_commands.txt \
  --out /workspace/lingbot_out/beauty_interaction.mp4
```

## Persona notes / limitations

- Control is **camera-level** (world navigation) + **prompt-level** (events,
  character behavior). There is no released skeletal/pose control channel — the
  `wasd_action.npy` / `ijkl_action.npy` files in upstream examples are not consumed
  by the released code path (`wasd_action` is hardcoded `None`).
- Character identity persists only while on-screen and within the KV-cache window
  (default `local_attn_size=18` latent frames ≈ 4.5s + 6 sink frames). Long
  occlusions can drift identity — same class of problem as the VACE identity work.
- Session length is bounded by the precomputed image-conditioning horizon
  (`--max-frames`, default 321 ≈ 20s), not by the model itself; raising it costs one
  longer VAE encode at session start.
- License: CC BY-NC-SA 4.0 (non-commercial).
