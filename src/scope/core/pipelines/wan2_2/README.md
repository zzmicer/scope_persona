# wan2_2 — Wan2.2-TI2V-5B component layer (scaffold)

Mirrors `../wan2_1/` to support the **Wan2.2-TI2V-5B** base model used by
LongLive 2.0. This was scaffolded on macOS; the real model is CUDA-only, so the
heavy transformer/VAE numerics are **stubs** to finish on a 5090.

## 5B vs 2.1 (from upstream config)

| field      | Wan2.1-1.3B | Wan2.2-TI2V-5B |
|------------|-------------|----------------|
| dim        | 1536        | **3072**       |
| ffn_dim    | 8960        | **14336**      |
| num_heads  | 12          | **24** (head_dim 128) |
| num_layers | 30          | 30             |
| in/out_dim | 16          | **48** (VAE latent ch) |
| VAE stride | (4, 8, 8)   | **(4, 16, 16)** (16x spatial) |
| sample_shift | 5.0       | 5.0            |

## Ported vs stubbed

- `components/generator.py` — **PORTED** (mechanical). `WanDiffusionWrapper`
  loader: config.json read, meta-device load, flow-matching x0 conversion,
  scheduler binding. Reuses `wan2_1` `FlowMatchScheduler` (formulation shared).
  Open `# TODO(longlive2)`: RoPE freq split for head_dim=128, and `seq_len`
  derivation for the 16x VAE grid.
- `modules/causal_model.py` — **STUB**. Real `__init__` signature + 5B config
  defaults + attributes the wrapper/utils read (`dim`, `num_heads`, `blocks`,
  `freqs`). Forward raises `NotImplementedError` with a per-section porting
  checklist. Port from `wan_5b/modules/causal_model.py` + `model.py`.
- `vae/vae2_2.py` + `vae/__init__.py` — **STUB** wrapper (48-ch, 16x) with a
  `create_vae`-compatible factory mirroring `wan2_1.vae`. Constants left `None`.
  Port from `wan_5b/modules/vae2_2.py`.

## Upstream file mapping (NVlabs/LongLive `main`, `wan_5b/`)

- `wan_5b/modules/causal_model.py` -> `modules/causal_model.py`
- `wan_5b/modules/model.py`        -> Wan building blocks used by the above
- `wan_5b/modules/vae2_2.py`       -> `vae/vae2_2.py`
- `wan_5b/configs/wan_ti2v_5B.py`  -> the dims baked into the stubs

## Must-do on the 5090

1. Implement `CausalWanModel` (attention via flash-attn/flex, RoPE for
   head_dim=128, adaLN, sink+local block-mask, KV cache) and remove the
   `NotImplementedError`.
2. Implement `Wan2_2_VAEWrapper` with the real 48-element mean/std constants and
   the `WanVAE_` encoder/decoder (16x spatial).
3. Fix `WanDiffusionWrapper.seq_len` / RoPE table for the 16x-VAE token grid and
   validate end-to-end against an upstream-generated reference latent.
