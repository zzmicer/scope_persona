# NVFP4 GPU kernels

All kernels here are **GPU-only** and import-guarded so this package loads on
CPU/macOS. They only run on Blackwell (sm_120 RTX 5090 / sm_100 B200).

## Triton kernels (no build step)

- `nvfp4_kernel.py` — `fp4_dequantize`, `static_blockwise_fp4_fake_quant`
- `adaln_triton.py` — fused adaLN modulation (`adaln_modulate_triton`)
- `rope_triton.py`  — fused RoPE (`rope_apply_triton`)

These are pure-Python Triton. `import triton` is guarded; the `@triton.jit`
kernels are only defined when Triton is importable, and the public wrappers
raise a clear `RuntimeError` if called without it. Triton ships with the
Blackwell PyTorch build on the pod — no separate build.

`static_blockwise_fp4_fake_quant(quantize_block_scales=True)` additionally
imports `modelopt.torch.quantization` lazily; install `nvidia-modelopt` on the
pod if you exercise that path.

## Fused KV-cache dequant CUDA extension (`kv_dequant/`)

A `torch` C++/CUDA extension (`longlive_kv_dequant_cuda`) exposing
`torch.ops.longlive_kernels.dequantize_kv_cache_fp4`. It uses `cuda_fp4.h` /
`__nv_cvt_fp4x2_to_halfraw2` (wraps the `cvt.rn.f16x2.e2m1x2` PTX instruction),
which requires **CUDA 12.8+** and an **architecture-specific** Blackwell target
(`sm_120a` / `sm_100a`) — plain `sm_120` / `sm_100` lack the instruction.

### Build on the pod

```bash
# from this directory:
cd kv_dequant

# RTX 5090 (sm_120, consumer Blackwell) — the default:
python setup.py build_ext --inplace

# B200 (sm_100):
LONGLIVE_KV_DEQUANT_ARCHS=100a python setup.py build_ext --inplace

# Both arches in one .so:
LONGLIVE_KV_DEQUANT_ARCHS=120a,100a python setup.py build_ext --inplace
```

`--inplace` drops the compiled `longlive_kv_dequant_cuda*.so` next to
`kv_dequant.py`, where its guarded `from . import longlive_kv_dequant_cuda`
finds it. If the extension is missing or stale, `quant.dequantize_kv_cache`
automatically falls back to the Triton per-block path (with a one-time warning).

Requires a matching CUDA toolkit + the PyTorch CUDA dev headers
(`torch.utils.cpp_extension`). Run inside the same env that provides the
Blackwell PyTorch build.
