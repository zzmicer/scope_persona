# NVFP4 (W4A4) inference for the wan2_2 longlive2 pipeline

Self-contained subpackage that quantizes the **5B generator** to NVFP4 and loads
the NVFP4 checkpoints. Ported faithfully from
[NVlabs/LongLive](https://github.com/NVlabs/LongLive) (`utils/`). Targets
Blackwell GPUs: **sm_120 (RTX 5090)** and **sm_100 (B200)**.

It does **not** build or run on CPU/macOS — that is expected. Every
GPU/Blackwell-only dependency is import-guarded, so the module still *imports*
cleanly on Mac and `nvfp4_available()` returns `False`, letting longlive2 fall
back to bf16.

## Public API

```python
from scope.core.pipelines.wan2_2.nvfp4 import (
    nvfp4_available,      # bool: CUDA + (fouroversix or transformer_engine)
    nvfp4_capabilities,   # dict: per-dep probe (cuda/fouroversix/transformer_engine/triton)
    setup_nvfp4_pipeline, # the orchestrator
)
```

`setup_nvfp4_pipeline(pipeline, config, device, *, verbose=False) -> pipeline`
detects the checkpoint type, quantizes `pipeline.generator.model`, loads the
weights, and returns the mutated `pipeline` with the quantized generator on
`device`. The orchestrator (longlive2/pipeline.py) is responsible for calling
it; this subpackage does not edit the pipeline.

## Two backends

| Backend | Checkpoint | Flag | Dep |
| --- | --- | --- | --- |
| **TransformerEngine** (primary) | `model_te.pt` (merged BF16) | `model_quant_use_transformer_engine: true` | `transformer-engine` |
| **FourOverSix** (fallback) | `model_4o6.pt` (materialized NVFP4 *or* BF16 base) | `model_quant_use_transformer_engine: false` | `fouroversix` |

- **TE path**: a merged BF16 generator is wrapped with TE NVFP4 `Linear`
  modules (`quantize_model_for_transformer_engine_nvfp4`), then materialized via
  `_materialize_transformer_engine_weights_for_inference`. `te_inference_only`
  and `te_low_precision_weights` are threaded through.
- **FourOverSix path**: a materialized NVFP4 state dict (detected by
  `*.quantized_weight_values` buffers) is loaded directly into the quantized
  architecture; or a BF16 base is quantized with `fouroversix.quantize_model`
  and materialized via `_materialize_quantized_weights_for_inference`.
- **TE → FourOverSix fallback**: when `model_quant_te_fallback_to_fouroversix:
  true`, the mixed materializer is used.

Optional LoRA (BF16 base only): if `config.adapter` + `config.lora_ckpt` are
set, the adapter is loaded + merged (`merge_and_unload`) before quantization.
Materialized NVFP4 checkpoints ignore LoRA (master weights already quantized).

## Capability / import guards

- `fouroversix`, `transformer_engine`, `modelopt` — imported lazily inside the
  functions that need them (`quant.py`). Missing on CPU ⇒ `fouroversix_available()`
  is `False`; calling a quant path raises a clear `RuntimeError`.
- `triton` — guarded in every kernel module; `kernels.triton_available()` reports
  status; wrappers raise if called without it.
- `longlive_kv_dequant_cuda` — guarded; `quant.dequantize_kv_cache` falls back to
  the Triton per-block path when the extension is missing/stale.
- `flash_attn` — used by the wan2_2 attention modules (outside this subpackage);
  also expected only on the pod.

## Config keys threaded through

Read from the runtime `config` (see upstream `configs/nvfp4/inference_nvfp4.yaml`):

- `model_quant` (gate; must be true)
- `model_quant_use_transformer_engine`
- `model_quant_te_inference_only`
- `model_quant_te_low_precision_weights`
- `model_quant_te_fallback_to_fouroversix`
- `model_quant_scale_rule`, `model_quant_activation_scale_rule`,
  `model_quant_weight_scale_rule`, `model_quant_gradient_scale_rule` (= `mse`)
- `model_quant_backend`, `model_quant_filtered_modules`,
  `model_quant_use_default_filtered_modules`,
  `model_quant_te_recipe_kwargs`, `model_quant_te_module_kwargs`
- `generator_ckpt` (the `model_te.pt` / `model_4o6.pt` path), `use_ema`
- `adapter`, `lora_ckpt` (optional LoRA)
- `streaming_vae`, `vae_device` (VAE placement)
- KV-cache quant (consumed by the longlive2 KV cache via `quant.py`):
  `kv_quant`, `kv_quant_scale_rule` (= `mse`), `kv_quant_backend` (= `cuda`)

## On-pod setup

```bash
# Blackwell PyTorch build provides torch + triton.

# FourOverSix (fallback backend) — build for sm_120 (RTX 5090):
CUDA_ARCHS=120 pip install fouroversix

# TransformerEngine (primary backend):
pip install transformer-engine

# flash-attn (wan2_2 attention):
pip install flash-attn==2.8.3 --no-build-isolation

# nvidia-modelopt (only for static_blockwise_fp4_fake_quant block-scale quant):
pip install nvidia-modelopt

# Fused KV-cache dequant CUDA extension (see kernels/README.md):
cd kernels/kv_dequant && python setup.py build_ext --inplace
```

See `kernels/README.md` for the exact KV-dequant build commands (sm_120a /
sm_100a / multi-arch).
