source /root/miniconda3/etc/profile.d/conda.sh
conda activate /workspace/soulx/env
export HF_HOME=/workspace/.hf
export PIP_CACHE_DIR=/workspace/.pip_cache
export CUDA_HOME=/workspace/soulx/env
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache
export TRITON_CACHE_DIR=/workspace/.triton_cache
export TOKENIZERS_PARALLELISM=false
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
