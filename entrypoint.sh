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
