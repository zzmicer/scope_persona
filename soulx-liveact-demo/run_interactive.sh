#!/bin/bash
# Interactive chat demo on port 8090, GPUs 2+3 — safe mode (SDPA, no fp8, no compile)
source /workspace/soulx/env.sh
cd /workspace/soulx/SoulX-LiveAct
# Qwen/Kokoro run as a separate process on physical GPU 0. Never expose that
# device inside torchrun: a third CUDA context in rank0 destabilizes NCCL.
CUDA_VISIBLE_DEVICES=0 python persona_aux.py --port 8091 --preload >> /workspace/soulx/persona_aux.log 2>&1 &
AUX_PID=$!
trap 'kill "$AUX_PID" 2>/dev/null || true' EXIT INT TERM
PYTHONPATH=/workspace/soulx/nosage:$PYTHONPATH USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=2,3 \
torchrun --nproc_per_node=2 --master_port=29617 \
  interactive_demo.py \
  --ckpt_dir /workspace/soulx/weights/LiveAct \
  --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
  --size 720*416 --fps 16 --port 8090 \
  --image /workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png \
  --aux_device cpu --aux_url http://127.0.0.1:8091 \
  --no_fp8_gemm --no_compile \
  --autostart
