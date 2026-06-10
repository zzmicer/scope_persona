# SPDX-License-Identifier: Apache-2.0
#
# Modified from upstream NVlabs/LongLive (utils/kernel/setup.py).
# Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
#
# Vendored into Daydream Scope. The only change vs upstream is the default GPU
# architecture: upstream hard-codes sm_100a (B200). For Daydream Scope's primary
# Blackwell target (RTX 5090, sm_120) we default to sm_120a, and allow building
# extra arches via the LONGLIVE_KV_DEQUANT_ARCHS env var.
#
# Build on the pod:
#   python setup.py build_ext --inplace            # sm_120a (RTX 5090)
#   LONGLIVE_KV_DEQUANT_ARCHS=120a,100a \
#       python setup.py build_ext --inplace        # RTX 5090 + B200
#
# The arch MUST be the "a" (architecture-specific) variant — plain sm_120 / sm_100
# lack the cvt.rn.f16x2.e2m1x2 instruction this kernel relies on.
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = Path(__file__).resolve().parent

# Default to sm_120a (RTX 5090 / consumer Blackwell). Override for B200 (sm_100a)
# or multi-arch builds via LONGLIVE_KV_DEQUANT_ARCHS="120a,100a".
_ARCHS = [
    arch.strip()
    for arch in os.environ.get("LONGLIVE_KV_DEQUANT_ARCHS", "120a").split(",")
    if arch.strip()
]
_GENCODE_FLAGS = [
    f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in _ARCHS
]

setup(
    name="longlive_kv_dequant_cuda",
    ext_modules=[
        CUDAExtension(
            name="longlive_kv_dequant_cuda",
            sources=[
                str(THIS_DIR / "kv_dequant.cpp"),
                str(THIS_DIR / "kv_dequant_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    # iter-37: need arch-specific Blackwell (sm_120a / sm_100a) for
                    # cvt.rn.f16x2.e2m1x2. Plain sm_120 / sm_100 lack it.
                    *_GENCODE_FLAGS,
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
