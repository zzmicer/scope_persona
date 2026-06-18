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

# LongLive 2.0 NVFP4 backend (Blackwell only, e.g. RTX 5090 / RTX PRO 6000 / B200).
# Enable with LONGLIVE2_NVFP4=1. Default backend is FourOverSix (model_4o6.pt);
# TransformerEngine (model_te.pt) is opt-in via LONGLIVE2_NVFP4_TE=1.
#
# VALIDATED on RTX PRO 6000 Blackwell (sm_120) 2026-06-18 — the exact recipe and
# its pitfalls are documented in docs/longlive2-runpod-bringup.md:
#   * FourOverSix MUST be the LongLive-bundled v1.1.0 (built WITH cutlass), NOT the
#     PyPI `fouroversix` (currently 1.0.5). 1.0.5 serializes a 6-field
#     quantized_weight_metadata; the released model_4o6.pt needs 1.1.0's 4-field
#     format, else the strict state-dict load fails with size mismatches.
#   * Building fouroversix 1.1.0 downgrades nvidia-cudnn-cu12 to 9.10.x, which makes
#     the Wan2.2 VAE 3D-conv pick a ~72GB workspace algo and OOM. Restore cudnn
#     afterwards.
#   * fouroversix needs transformers>=5 (WeightConverter); base lock pins 4.57.x.
if [ "$LONGLIVE2_NVFP4" = "1" ]; then
  if [ "$GPU_CC" -ge 120 ]; then
    KV_ARCH="120a"
  elif [ "$GPU_CC" -ge 100 ]; then
    KV_ARCH="100a"
  else
    KV_ARCH=""
  fi
  if [ -n "$KV_ARCH" ]; then
    if ! python -c "import fouroversix._C" > /dev/null 2>&1; then
      echo "Installing NVFP4 backend (FourOverSix 1.1.0 + cutlass + KV-dequant, sm_${KV_ARCH})..."

      # CUDA math + cuDNN *dev* headers needed by the native builds (cutlass nvcc,
      # kv_dequant, and the optional TE build). Some CUDA APT packages ship on hold
      # and the dev headers aren't in the base image. Best-effort.
      echo "Installing CUDA/cuDNN dev headers..."
      apt-mark unhold libcublas-12-8 libcudnn9-cuda-12 2>/dev/null || true
      apt-get update || true
      apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-libraries-dev-12-8 \
        libcudnn9-dev-cuda-12 ninja-build \
        || echo "CUDA dev header install failed — native builds may fail."

      # transformers>=5 (WeightConverter) — required by fouroversix. Runs before the
      # server starts, so the umt5 tokenizer (AutoTokenizer, v5-safe) is consistent.
      echo "Upgrading transformers to >=5 for FourOverSix..."
      uv pip install --no-build-isolation "transformers>=5.0.0" || echo "transformers>=5 upgrade failed."

      # FourOverSix 1.1.0 from the LongLive repo, built WITH cutlass for this arch.
      # The cutlass submodule lives in fouroversix/.gitmodules (ignored by the parent
      # clone), so clone it directly into place; the fouroversix source dir has no
      # .git, so setup.py builds once third_party/cutlass exists.
      echo "Building FourOverSix 1.1.0 (+cutlass, sm_${GPU_CC})..."
      FOS_DIR="/workspace/_longlive_src"
      rm -rf "$FOS_DIR" && mkdir -p "$FOS_DIR"
      if git clone --depth 1 https://github.com/NVlabs/LongLive.git "$FOS_DIR/LongLive"; then
        ( cd "$FOS_DIR/LongLive/fouroversix" \
          && git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass \
          && VIRTUAL_ENV=/app/.venv CUDA_ARCHS="${GPU_CC}" MAX_JOBS="${MAX_JOBS:-24}" FORCE_BUILD=1 \
             uv pip install --no-build-isolation . ) \
          || echo "FourOverSix 1.1.0 cutlass build failed."
      else
        echo "Could not clone LongLive repo for FourOverSix build."
      fi

      # The fouroversix build downgrades nvidia-cudnn-cu12 (-> 9.10.x), which causes
      # a ~72GB VAE-decode workspace OOM. Restore a recent cudnn.
      echo "Restoring nvidia-cudnn-cu12 (undo fouroversix build downgrade)..."
      uv pip install --no-build-isolation -U "nvidia-cudnn-cu12" || echo "cudnn restore failed."

      # KV-cache dequant CUDA extension.
      KVDIR="/app/src/scope/core/pipelines/wan2_2/nvfp4/kernels/kv_dequant"
      if [ -d "$KVDIR" ]; then
        ( cd "$KVDIR" && LONGLIVE_KV_DEQUANT_ARCHS="$KV_ARCH" uv run --no-sync python setup.py build_ext --inplace ) \
          || echo "KV-dequant kernel build failed."
      fi

      # TransformerEngine (model_te.pt backend) — OPT-IN only. fouroversix is the
      # default; TE builds from source (cu12/torch2.9 has no wheel) and is heavier.
      if [ "$LONGLIVE2_NVFP4_TE" = "1" ] && ! uv pip show transformer-engine > /dev/null 2>&1; then
        echo "Installing optional TransformerEngine backend..."
        uv pip install --no-build-isolation "transformer-engine[pytorch]" || echo "TransformerEngine install failed."
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
