# The conda *installation* lived on the container root disk and is gone after a
# container recreate; the env itself is on the network volume and survives.
# `conda activate` is only PATH/vars + activate.d, so reproduce it directly.
export CONDA_PREFIX=/workspace/soulx/env
export PATH=/workspace/soulx/env/bin:$PATH
for _f in /workspace/soulx/env/etc/conda/activate.d/*.sh; do
  [ -e "$_f" ] && . "$_f"
done
unset _f

export HF_HOME=/workspace/.hf
export PIP_CACHE_DIR=/workspace/.pip_cache
export CUDA_HOME=/workspace/soulx/env
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_cache
export TRITON_CACHE_DIR=/workspace/.triton_cache
export TOKENIZERS_PARALLELISM=false
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
