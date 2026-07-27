#!/bin/bash
# Interactive chat demo on port 8090, GPUs 2+3 — safe mode (SDPA, no fp8, no compile)
#
# STREAM_SIZE is WIDTH*HEIGHT (both multiples of 16); portrait is just the
# transpose and costs the same, since only (H/16)*(W/16) tokens reach the model:
#   STREAM_SIZE=720*416 ./run_interactive.sh   # landscape (default)
#   STREAM_SIZE=416*720 ./run_interactive.sh   # vertical, identical fps/VRAM
#   STREAM_SIZE=480*832 ./run_interactive.sh   # taller 9:16, ~1.33x tokens, slower
#
# STREAM_IMAGE must be framed for STREAM_SIZE: the reference is centre-cropped to
# the stream aspect, so a 16:9 source keeps only its middle third in portrait (the
# stock chano39 image loses her face that way) -- pass the pre-framed portrait crop
# /workspace/soulx_setup/chano39-portrait.png when running vertical.
#
# Do NOT rename these back to SIZE/IMAGE: `conda activate` inside env.sh exports
# the binutils tool vars (SIZE=<host>-size, plus AR/AS/LD/NM/STRIP/...), which
# silently overwrote a SIZE=... override here and fed the generator a file path.
STREAM_SIZE="${STREAM_SIZE:-720*416}"
STREAM_FPS="${STREAM_FPS:-16}"
STREAM_PORT="${STREAM_PORT:-8090}"
STREAM_IMAGE="${STREAM_IMAGE:-/workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png}"
# STREAM_FP8: off (default) | blocks | all. FP8 W8A8 was disabled here because it
# produced noise -- but it was being applied to EVERY linear, including the
# modulation/embedding/head layers that carry no FLOPs and hate quantization.
# "blocks" covers only the attention/FFN matmuls. Verify output before trusting.
STREAM_FP8="${STREAM_FP8:-off}"
case "$STREAM_FP8" in
  off|0|"") FP8_ARGS=(--no_fp8_gemm) ;;
  blocks|1) FP8_ARGS=(--fp8_scope blocks) ;;
  all)      FP8_ARGS=(--fp8_scope all) ;;
  *) echo "STREAM_FP8 must be off|blocks|all (got $STREAM_FP8)" >&2; exit 1 ;;
esac
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
  --size "$STREAM_SIZE" --fps "$STREAM_FPS" --port "$STREAM_PORT" \
  --image "$STREAM_IMAGE" \
  --aux_device cpu --aux_url http://127.0.0.1:8091 \
  "${FP8_ARGS[@]}" --no_compile \
  --autostart
