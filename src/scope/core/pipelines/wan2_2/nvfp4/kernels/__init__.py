# SPDX-License-Identifier: Apache-2.0
#
# NVFP4 GPU kernels for the Daydream Scope wan2_2 longlive2 pipeline.
#
# All kernels here are GPU-only (Triton / CUDA C++ extension). Importing this
# package on CPU/macOS is safe: the heavy ``triton`` import is guarded inside
# each module, and ``triton_available()`` reports whether the Triton kernels can
# actually run.
"""GPU kernels: Triton adaLN / RoPE / fp4-dequant + the fused KV-dequant CUDA ext."""
from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language  # noqa: F401

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on CPU/macOS.
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    """Return True when Triton is importable (i.e. the Triton kernels can run)."""
    return _TRITON_AVAILABLE


__all__ = ["triton_available"]
