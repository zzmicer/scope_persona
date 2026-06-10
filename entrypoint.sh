#!/bin/bash
set -e

# Detect GPU compute capability (major.minor)
GPU_CC=$(/app/.venv/bin/python -c "import torch; cc = torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "0")

# Default attention backend based on GPU capability
if [ -z "$ATTENTION_BACKEND" ]; then
  if [ "$GPU_CC" -ge 100 ]; then
    ATTENTION_BACKEND="sa3"
  else
    ATTENTION_BACKEND="sa2"
  fi
fi

echo "GPU SM $GPU_CC — ATTENTION_BACKEND=$ATTENTION_BACKEND"

case "$ATTENTION_BACKEND" in
  sa3)
    if ! uv pip show sageattn3 > /dev/null 2>&1; then
      echo "Installing sageattn3 (first boot, compiling CUDA extensions)..."
      uv pip install --no-build-isolation --no-deps "sageattn3 @ git+https://github.com/thu-ml/SageAttention#subdirectory=sageattention3_blackwell"
      echo "sageattn3 installed successfully."
    fi
    ;;
  sa2)
    if ! uv pip show sageattention > /dev/null 2>&1; then
      echo "Installing SageAttention 2++ from source..."
      uv pip install --no-build-isolation "sageattention @ git+https://github.com/thu-ml/SageAttention@v2.2.0"
      echo "SageAttention 2++ installed successfully."
    fi
    ;;
  flash)
    echo "Using Flash Attention 2 (should be pre-installed in image)."
    ;;
  *)
    echo "Unknown ATTENTION_BACKEND=$ATTENTION_BACKEND — expected sa3, sa2, or flash"
    exit 1
    ;;
esac

# LongLive 2.0 NVFP4 backend (Blackwell only, e.g. RTX 5090 / B200).
# Enable with LONGLIVE2_NVFP4=1. Installs TransformerEngine + FourOverSix and
# compiles the KV-cache dequant CUDA extension for the detected architecture.
if [ "$LONGLIVE2_NVFP4" = "1" ]; then
  if [ "$GPU_CC" -ge 120 ]; then
    KV_ARCH="120a"
  elif [ "$GPU_CC" -ge 100 ]; then
    KV_ARCH="100a"
  else
    KV_ARCH=""
  fi
  if [ -n "$KV_ARCH" ]; then
    if ! uv pip show transformer-engine > /dev/null 2>&1; then
      echo "Installing NVFP4 backend (TransformerEngine + FourOverSix + KV-dequant kernel, sm_${KV_ARCH})..."
      uv pip install --no-build-isolation "transformer-engine[pytorch]" || echo "TransformerEngine install failed."
      CUDA_ARCHS="${GPU_CC}" uv pip install --no-build-isolation fouroversix || echo "FourOverSix install failed."
      KVDIR="/app/src/scope/core/pipelines/wan2_2/nvfp4/kernels/kv_dequant"
      if [ -d "$KVDIR" ]; then
        ( cd "$KVDIR" && LONGLIVE_KV_DEQUANT_ARCHS="$KV_ARCH" uv run --no-sync python setup.py build_ext --inplace ) \
          || echo "KV-dequant kernel build failed."
      fi
    fi
  else
    echo "LONGLIVE2_NVFP4=1 but GPU SM $GPU_CC is not Blackwell (need >=100) — skipping NVFP4 backend; longlive2 falls back to bf16."
  fi
fi

# Pre-download model weights onto the persistent /workspace volume so the first
# request doesn't block on a multi-GB download. Set e.g. PREFETCH_PIPELINES="longlive2"
# (space-separated for multiple pipelines).
if [ -n "$PREFETCH_PIPELINES" ]; then
  for p in $PREFETCH_PIPELINES; do
    echo "Prefetching model weights for pipeline: $p"
    uv run --no-sync download_models --pipeline "$p" \
      || echo "Prefetch failed for $p — will lazy-download on first use."
  done
fi

# Download LoRA models
LORA_DIR="/workspace/models/lora"
mkdir -p "$LORA_DIR"

if [ -n "$CIVIT_TOKEN" ]; then
  if [ ! -f "$LORA_DIR/nsfw.safetensors" ]; then
    echo "Downloading LoRA: nsfw.safetensors..."
    wget -O "$LORA_DIR/nsfw.safetensors" \
      "https://civitai.com/api/download/models/1514371?type=Model&format=SafeTensor&token=$CIVIT_TOKEN"
  else
    echo "LoRA nsfw.safetensors already exists, skipping download."
  fi
else
  echo "CIVIT_TOKEN not set, skipping LoRA downloads."
fi

exec "$@"
