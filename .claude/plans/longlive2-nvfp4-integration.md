# LongLive 2.0 (NVFP4) Integration Plan

Target: add a new **`longlive2`** pipeline to Daydream Scope built on **Wan2.2‑TI2V‑5B**,
running the **NVFP4** (W4A4) inference path on the user's **RTX 5090** (Blackwell, sm_120, 32 GB).
TI2V‑5B is a single combined model → supports **both text (T2V) and image (I2V)** conditioning
from the same weights, so both modalities are in scope.

Upstream: https://github.com/NVlabs/LongLive (branch `main` = 2.0; `v1.0` = the LongLive we already support).

## Checkpoints (verified)

| Model | HF repo | Precision | Steps | FPS* | VBench | Files |
|---|---|---|---|---|---|---|
| LongLive‑2.0‑5B | `Efficient-Large-Model/LongLive-2.0-5B` | BF16 | 4 | 24.8 | 85.06 | `model_bf16.pt` |
| ‑NVFP4‑S4 | `Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4` | NVFP4 | 4 | 29.7 | 84.51 | `model_te.pt`, `model_4o6.pt` |
| ‑NVFP4‑S2 | `Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2` | NVFP4 | 2 | 45.7 | 83.14 | `model_te.pt`, `model_4o6.pt` |

*Reported on H100; 5090 runs NVFP4 natively. `model_te.pt` = TransformerEngine path (primary);
`model_4o6.pt` = fouroversix fallback. LoRA is pre‑merged → single `generator_ckpt` for inference.

Base deps: `Wan-AI/Wan2.2-TI2V-5B` (config + VAE 2.2), UMT5 text encoder (already vendored via
`Kijai/WanVideo_comfy`), optional LightVAE `Skywork/Matrix-Game-3.0` (`MG-LightVAE.pth`).

## Architecture gap vs current LongLive 1 (`pipelines/longlive`)

| | LongLive 1 (have) | LongLive 2.0 (target) |
|---|---|---|
| Base model | Wan2.1‑T2V‑1.3B | Wan2.2‑TI2V‑5B |
| VAE | Wan 2.1, 16 ch, 8× | Wan 2.2, 48 ch, 16× (changes resolution scale factor) |
| Attention | `flex_attention` (ported) | flash‑attn 2.8.3 (built from source) |
| Quant | torchao FP8 (opt) | TransformerEngine NVFP4 W4A4 + fouroversix fallback |
| KV cache | Python cache mgmt | + KV‑cache quant via custom CUDA dequant kernel (`utils/kernel`) |
| Fused kernels | — | Triton adaLN + RoPE |
| frame/block, sink | 3 / 3 | 8 / 8 |
| Modalities | T2V (+VACE extend) | T2V + I2V (TI2V), multi‑shot |

Two hard constraints:
1. **NVFP4 first still needs a correct 5B forward.** BF16‑5B is an intermediate correctness gate,
   not a separate feature — same code path minus the quant wrapper.
2. **Native stack (TransformerEngine + fouroversix + flash‑attn 2.8.3 + kv_dequant CUDA ext) builds
   only on the 5090/Linux box, never on macOS.** All kernel work develops/tests on the Blackwell machine.

## Phases

### Phase 0 — Environment & dependency spike (on the 5090, de‑risk first)
- Reproduce upstream NVFP4 inference outside Scope: CUDA 12.8, PyTorch 2.8+, build `fouroversix`
  (`CUDA_ARCHS=120`), `transformer-engine[pytorch]`, flash‑attn 2.8.3, `utils/kernel` kv_dequant ext.
- Run `inference.py --nproc_per_node=1` for a known‑good reference clip + 5090 timing.
- Decide how native deps enter Scope's `uv` env (optional `[longlive2]` extra; won't install on Mac).
  **Biggest risk — validate before writing pipeline code.**

### Phase 1 — `pipelines/wan2_2/` component layer  (mirror of `wan2_1/`)
- Port 5B `causal_model.py`; Wan 2.2 VAE (`vae2_2`, 48‑ch) into `create_vae` factory; reuse UMT5.
- `WanDiffusionWrapper`‑equivalent loader for 5B `config.json` + `generator_ckpt`.
- Fix `validate_resolution` scale factor for 16× compression.

### Phase 2 — `pipelines/longlive2/` scaffold  (BF16 correctness gate)
- `schema.py` (`LongLive2Config` + artifacts), `model.yaml` (num_frame_per_block=8, local_attn_size=32,
  sink_size=8, timestep_shift=5.0, 4‑step DMD), `pipeline.py`, `modular_blocks.py`, registry entry.
- Reuse existing modular‑blocks / cache management. **Gate: match upstream BF16 output.**

### Phase 3 — NVFP4 quantization path (target)
- Port `setup_nvfp4_pipeline`: TE quantization, checkpoint detection (`is_te_nvfp4_checkpoint`),
  weight materialization (`utils/quant`). Load `model_te.pt` (TE primary, `model_4o6.pt` fallback).
- Wire KV‑cache quant (`kv_quant: mse`, cuda backend) into cache management.
- Config field `precision` / `steps` (S2 2‑step vs S4 4‑step).

### Phase 4 — Scope integration & realtime
- WebRTC streaming; prompt switching via embedding blender → multi‑shot sink/rope offset.
- I2V first‑frame conditioning (`utils/i2v_conditioning.py`) — reference‑image character anchoring for persona.
- `estimated_vram_gb` (NVFP4‑5B ≈ 8–10 GB). Drop sequence‑parallel (single‑GPU only).

### Phase 5 — VACE/LoRA parity (optional, later) — only if needed; LL2 ships merged LoRA.

## What can be done now (macOS) vs on the 5090
- **Now (reviewable on Mac):** package scaffolding, schema/config/artifacts, registry wiring, model.yaml,
  docs, and structural ports of pure‑Python modules with clear upstream references.
- **5090 only:** native builds, running the model, NVFP4 numeric validation, FPS/VRAM measurement.
