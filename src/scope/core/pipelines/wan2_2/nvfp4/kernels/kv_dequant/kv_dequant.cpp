// SPDX-License-Identifier: Apache-2.0
//
// Modified from upstream NVlabs/LongLive (utils/kernel/kv_dequant.cpp).
// Original: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache-2.0.
//
// Vendored into Daydream Scope, unchanged. Compiled on the pod via setup.py.
#include <torch/extension.h>

TORCH_LIBRARY(longlive_kernels, m)
{
    m.def("dequantize_kv_cache_fp4(Tensor[] values, Tensor[] scale_factors, Tensor[] amax, int num_heads, int block_token_size, int dtype_code, float e2m1_max, float e4m3_max) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "LongLive custom CUDA kernels";
}
