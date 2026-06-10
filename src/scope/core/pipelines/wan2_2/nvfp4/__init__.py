# SPDX-License-Identifier: Apache-2.0
#
# NVFP4 (W4A4) inference subpackage for the Daydream Scope wan2_2 longlive2
# pipeline. Targets Blackwell GPUs (sm_120 RTX 5090 / sm_100 B200).
#
# This package imports cleanly on CPU/macOS: every GPU/Blackwell-only dependency
# (transformer_engine, fouroversix, triton, the kv_dequant C++/CUDA extension,
# flash_attn) is import-guarded inside the submodules. Call ``nvfp4_available()``
# to decide whether to take the NVFP4 path or fall back to bf16.
"""NVFP4 inference: quantize the 5B generator and load NVFP4 checkpoints.

Backends (mirrors upstream LongLive):

* **TransformerEngine** (``model_te.pt``) — primary path; merged BF16 generator
  wrapped with TE NVFP4 ``Linear`` modules. Requires ``transformer_engine``.
* **FourOverSix** (``model_4o6.pt``) — fallback; either a BF16 base quantized at
  load time or a pre-materialized NVFP4 state dict. Requires ``fouroversix``.

Usage from the longlive2 pipeline::

    from scope.core.pipelines.wan2_2.nvfp4 import nvfp4_available, setup_nvfp4_pipeline

    if nvfp4_available():
        pipeline = setup_nvfp4_pipeline(pipeline, config, device)
    else:
        ...  # bf16 fallback
"""
from __future__ import annotations

from .setup_pipeline import setup_nvfp4_pipeline

__all__ = ["setup_nvfp4_pipeline", "nvfp4_available", "nvfp4_capabilities"]


def nvfp4_capabilities() -> dict[str, bool]:
    """Return a per-dependency capability map for the NVFP4 path.

    Keys:

    * ``cuda`` — a CUDA device is visible to torch.
    * ``fouroversix`` — the FourOverSix quantization library is importable.
    * ``transformer_engine`` — TransformerEngine is importable.
    * ``triton`` — Triton (for the fused fp4 / adaLN / RoPE kernels) is importable.

    All probes are import-guarded; this never raises.
    """
    caps = {
        "cuda": False,
        "fouroversix": False,
        "transformer_engine": False,
        "triton": False,
    }

    try:
        import torch

        caps["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    try:
        from .quant import fouroversix_available

        caps["fouroversix"] = bool(fouroversix_available())
    except Exception:
        pass

    try:
        import importlib.util

        caps["transformer_engine"] = (
            importlib.util.find_spec("transformer_engine") is not None
        )
    except Exception:
        pass

    try:
        from .kernels import triton_available

        caps["triton"] = bool(triton_available())
    except Exception:
        pass

    return caps


def nvfp4_available() -> bool:
    """Return True only when NVFP4 inference can actually run on this machine.

    Requires a CUDA device plus at least one usable quantization backend
    (``fouroversix`` for the FourOverSix checkpoint, or ``transformer_engine``
    for the TE checkpoint). Returns ``False`` on CPU/macOS so the longlive2
    pipeline can fall back to bf16 cleanly.
    """
    caps = nvfp4_capabilities()
    if not caps["cuda"]:
        return False
    return caps["fouroversix"] or caps["transformer_engine"]
