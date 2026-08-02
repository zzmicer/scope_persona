# SoulX-LiveAct persona demo.
#
# CUDA 12.8 + torch 2.8 cu128 is the floor, not a preference: sm_120 (Blackwell
# RTX PRO 6000 / RTX 50xx) has no kernels in cu126 wheels, and the image is
# meant to run unchanged on Hopper, Ada and Blackwell.
#
# Weights are NOT baked in -- they are ~40GB and belong on the pod's network
# volume. Mount it at /workspace/soulx and run `scope-soulx fetch` once.
#
#   docker build -t soulx-demo -f docker/Dockerfile .
#   docker run --gpus all -p 8090:8090 -v /workspace/soulx:/workspace/soulx \
#              soulx-demo run --preset fast --foreground
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is not optional: the demo muxes video+audio into a live HLS stream.
# sox and libsndfile are for the TTS path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        ffmpeg sox libsndfile1 libgl1 libglib2.0-0 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

# Big, stable layers first. torch 2.8.0 from PyPI is the cu128 build.
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

COPY SoulX-LiveAct/requirements.txt /tmp/requirements.txt
RUN sed -i '/^torch/d;/^torchvision/d;/^torchaudio/d' /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt \
    && pip install flask flask-sock huggingface_hub[cli] requests

# vLLM supplies the W8A8 FP8 GEMM (fp8_gemm.py) and persona_aux's Qwen. 0.11.0
# pins torch 2.8.0, which is why the torch install above is exact.
RUN pip install vllm==0.11.0

# lightx2v's VAE package: the Wan VAE wrapper the demo loads AND the WanVAE_tiny
# class behind every --vae option except `wan`.
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V /tmp/LightX2V \
    && cd /tmp/LightX2V && python setup_vae.py install && rm -rf /tmp/LightX2V

# Deliberately NOT installed: SageAttention. Its kernels produce noise/NaN on
# this pipeline, the demo ships a shim that blocks the import anyway, and
# building it added ~20 minutes to every environment build.

WORKDIR /opt/soulx
COPY SoulX-LiveAct/ /opt/soulx/SoulX-LiveAct/
COPY ultraflash/ /opt/soulx/ultraflash/
COPY nosage/ /opt/soulx/nosage/
COPY scope-soulx /opt/soulx/scope-soulx
COPY docker/entrypoint.sh /opt/soulx/entrypoint.sh
# Reference images, used to seed a fresh volume's assets/ dir.
COPY *.png /opt/soulx/assets/
RUN chmod +x /opt/soulx/scope-soulx /opt/soulx/entrypoint.sh \
    && ln -s /opt/soulx/scope-soulx /usr/local/bin/scope-soulx

ENV SOULX_ROOT=/workspace/soulx \
    SOULX_SRC=/opt/soulx/SoulX-LiveAct \
    HF_HOME=/workspace/soulx/.hf \
    TOKENIZERS_PARALLELISM=false \
    XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep the inductor/triton caches on the mounted volume. torch.compile warmup is
# 8-15 minutes and is re-paid on every resolution change; on the volume it is
# paid once per resolution for the life of the pod instead of once per container.
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/soulx/.inductor_cache \
    TRITON_CACHE_DIR=/workspace/soulx/.triton_cache

EXPOSE 8090 8091
ENTRYPOINT ["/opt/soulx/entrypoint.sh"]
CMD ["run", "--preset", "fast", "--foreground"]
