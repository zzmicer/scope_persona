#!/bin/bash
# Download SoulX-LiveAct checkpoints to /workspace (root disk is full)
set -x
export HF_HOME=/workspace/.hf
mkdir -p /workspace/soulx/weights
hf download Soul-AILab/LiveAct --local-dir /workspace/soulx/weights/LiveAct --exclude "assets/*" \
  && echo LIVEACT_DOWNLOAD_DONE || echo LIVEACT_DOWNLOAD_FAILED
hf download TencentGameMate/chinese-wav2vec2-base --local-dir /workspace/soulx/weights/chinese-wav2vec2-base \
  && echo WAV2VEC_DOWNLOAD_DONE || echo WAV2VEC_DOWNLOAD_FAILED
echo ALL_DOWNLOADS_FINISHED
