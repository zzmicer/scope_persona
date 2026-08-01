#!/bin/bash
# SoulX-LiveAct persona demo, single H200.
#
# Default config = the one chosen after benchmarking:
#   taew2_1 tiny decoder  (official, 23MB, MAE 2.77 vs the full Wan VAE at 32ms)
#   block-level torch.compile  (17-20% latency off; warmup is minutes, amortized)
#   FP8 W8A8 scoped to the 480 block matmuls
#
# Measured s/chunk on 1xH200 against a 2.0s realtime budget:
#   size     tiny+compile   tiny        wanVAE+compile   wanVAE
#   720x416  2.50  (0.80x)  3.01 (0.66) -                3.64 (0.55)
#   368x640  1.80  (1.11x)  -           -                -
#   592x336  1.42  (1.41x)  1.78 (1.12) 1.74 (1.15x)     2.19 (0.91)
# NOTE 720x416 does NOT stream live even compiled -- it is the quality config.
# Portrait is free: cost tracks pixel count, not orientation.
#
# SOULX_ATTN=fa3 switches self-attention from PyTorch SDPA to FlashAttention-3.
# Gated microbenchmark on real shapes: 2.1-2.3x on the attention itself, which is
# ~60-100ms of a chunk. Numerically equivalent (max|delta| 5e-4 vs SDPA).
# Default stays sdpa -- every other number here was measured on it.
#
# Env: STREAM_SIZE, TINY_DECODER, COMPILE, FAST_DECODE, SOULX_ATTN, GEN_GPU,
#      STREAM_PORT, AUX_PORT, STREAM_FP8.
set -u
ENVSH=/workspace/soulx/env.sh

STREAM_SIZE="${STREAM_SIZE:-368*640}"
STREAM_FPS="${STREAM_FPS:-16}"
STREAM_PORT="${STREAM_PORT:-8141}"
AUX_PORT="${AUX_PORT:-8241}"
STREAM_IMAGE="${STREAM_IMAGE:-/workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png}"
STREAM_FP8="${STREAM_FP8:-blocks}"
FAST_DECODE="${FAST_DECODE:-1}"
TINY_DECODER="${TINY_DECODER:-taew2_1}"
COMPILE="${COMPILE:-1}"
ATTN="${SOULX_ATTN:-sdpa}"
GEN_GPU="${GEN_GPU:-1}"
AUX_GPU="${AUX_GPU:-$GEN_GPU}"     # co-located: world_size==1, so no NCCL to upset
FA3_PATH=

case "$STREAM_FP8" in
  off|0|"") FP8_ARGS=(--no_fp8_gemm) ;;
  blocks|1) FP8_ARGS=(--fp8_scope blocks) ;;
  all)      FP8_ARGS=(--fp8_scope all) ;;
  *) echo "STREAM_FP8 must be off|blocks|all (got $STREAM_FP8)" >&2; exit 1 ;;
esac
case "$FAST_DECODE" in
  1|on|yes) DEC_ARGS=(--fast_decode --tiny_decoder "$TINY_DECODER") ;;
  *)        DEC_ARGS=() ;;
esac
case "$COMPILE" in
  1|on|yes) COMPILE_ARGS=(--compile_blocks --compile_mode max-autotune-no-cudagraphs) ;;
  *)        COMPILE_ARGS=(--no_compile) ;;
esac

set +u; source "$ENVSH"   # stays +u: conda activate.d and PYTHONPATH are unbound-unsafe
cd /workspace/soulx/SoulX-LiveAct

CUDA_VISIBLE_DEVICES=$AUX_GPU python persona_aux.py --port "$AUX_PORT" --preload \
  >> /workspace/soulx/persona_aux.log 2>&1 &
AUX_PID=$!
trap 'kill "$AUX_PID" 2>/dev/null || true' EXIT INT TERM

SOULX_ATTN=$ATTN \
PYTHONPATH=/workspace/soulx/nosage:$PYTHONPATH USE_CHANNELS_LAST_3D=1 \
CUDA_VISIBLE_DEVICES=$GEN_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python pruna_bench.py \
  --ckpt_dir /workspace/soulx/weights/LiveAct \
  --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
  --size "$STREAM_SIZE" --fps "$STREAM_FPS" --port "$STREAM_PORT" \
  --image "$STREAM_IMAGE" \
  --aux_device cpu --aux_url "http://127.0.0.1:$AUX_PORT" \
  "${FP8_ARGS[@]}" "${DEC_ARGS[@]}" "${COMPILE_ARGS[@]}" \
  --autostart
