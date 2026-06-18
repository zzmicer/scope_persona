# SPDX-License-Identifier: Apache-2.0
#
# Modified from upstream NVlabs/LongLive (utils/kernel/__init__.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
#
# Vendored into Daydream Scope. The compiled CUDA extension
# (``longlive_kv_dequant_cuda``) is only present after running ``setup.py
# build_ext`` on the pod (see ../README.md). This package imports cleanly on
# CPU/macOS; the fused dequant path is selected at runtime by ``quant.py`` only
# when the extension and a CUDA device are both available.
"""Custom CUDA kernels used by LongLive (fused NVFP4 KV-cache dequantization)."""
