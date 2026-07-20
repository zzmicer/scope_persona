#!/bin/bash
# Build SoulX-LiveAct python env entirely on /workspace (root disk ~8GB free)
set -ex
export CONDA_PKGS_DIRS=/workspace/.conda_pkgs
export PIP_CACHE_DIR=/workspace/.pip_cache
export HF_HOME=/workspace/.hf
source /root/miniconda3/etc/profile.d/conda.sh

mkdir -p /workspace/soulx
cd /workspace/soulx

if [ ! -d /workspace/soulx/env ]; then
  conda create -y -p /workspace/soulx/env python=3.10
fi
conda activate /workspace/soulx/env

conda install -y conda-forge::sox
# nvcc 12.8 to match torch cu128 wheels (system nvcc is 12.4)
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit

if [ ! -d SoulX-LiveAct ]; then
  git clone https://github.com/Soul-AILab/SoulX-LiveAct
fi

pip install -r SoulX-LiveAct/requirements.txt
pip install flask  # demo.py imports it; missing from requirements
pip install vllm==0.11.0
echo PIP_CORE_DONE

if [ ! -d SageAttention ]; then
  git clone https://github.com/thu-ml/SageAttention.git
fi
cd SageAttention
git checkout v2.2.0
export CUDA_HOME=/workspace/soulx/env
EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=48 python setup.py install
echo SAGE_DONE
cd /workspace/soulx

if [ ! -d LightX2V ]; then
  git clone https://github.com/ModelTC/LightX2V
fi
cd LightX2V
python setup_vae.py install
echo LIGHTVAE_DONE

python -c "import torch, sageattention, flask, xfuser; from lightx2v.models.video_encoders.hf.wan.vae import WanVAE; print('IMPORTS_OK', torch.__version__)"
echo ENV_SETUP_FINISHED
