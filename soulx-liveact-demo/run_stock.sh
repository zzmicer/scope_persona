#!/bin/bash
# Stock GUI demo on port 8090, GPUs 2+3
source /workspace/soulx/env.sh
cd /workspace/soulx/SoulX-LiveAct
USE_CHANNELS_LAST_3D=1 CUDA_VISIBLE_DEVICES=2,3 \
torchrun --nproc_per_node=2 --master_port=29617 \
  demo.py \
  --ckpt_dir /workspace/soulx/weights/LiveAct \
  --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
  --size 720*416 \
  --port 8090 \
  --video_save_path /workspace/soulx/generated_videos
