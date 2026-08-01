#!/bin/bash
# SoulX-LiveAct on ONE H200: no torchrun, no sequence parallel, no SR sidecar.
#
# world_size==1 makes interactive_demo.py import the non-SP model_memory.WanModel,
# and -- the point of this layout -- there is no NCCL at all. The rule that forced
# persona_aux onto its own GPU (a second CUDA context inside a torchrun rank
# destabilizes NCCL) does not apply, so aux co-locates on the generator GPU by
# default. Budget: ~60-76GB generator + ~4GB aux against 143GB.
#
# The Wan2.1 VAE decode (452ms at 720x416) is replaced by the v3 tiny decoder
# (31ms, MAE 6.4/255) via --fast_decode. That is the 421ms/chunk that makes
# realtime on one GPU reachable; the rest comes from resolution.
#
# Measured on 1xH200 (GPU1), FP8 blocks + fast_decode, s/chunk vs a 2.0s budget:
#   720x416 3.01 | 640x368 2.18 | 592x336 1.78 | 320x576 1.61 | 560x320 1.57
#
# t5 deliberately stays on the GPU: 11GB fits here, and --t5_cpu would stall the
# chunk loop on every action transition.
STREAM_SIZE="${STREAM_SIZE:-592*336}"
STREAM_FPS="${STREAM_FPS:-16}"
STREAM_PORT="${STREAM_PORT:-8090}"
AUX_PORT="${AUX_PORT:-8091}"
STREAM_IMAGE="${STREAM_IMAGE:-/workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png}"
STREAM_FP8="${STREAM_FP8:-blocks}"
FAST_DECODE="${FAST_DECODE:-1}"
COMPILE="${COMPILE:-0}"
GEN_GPU="${GEN_GPU:-1}"
AUX_GPU="${AUX_GPU:-$GEN_GPU}"     # co-located by default; no NCCL to upset
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "$STREAM_FP8" in
  off|0|"") FP8_ARGS=(--no_fp8_gemm) ;;
  blocks|1) FP8_ARGS=(--fp8_scope blocks) ;;
  all)      FP8_ARGS=(--fp8_scope all) ;;
  *) echo "STREAM_FP8 must be off|blocks|all (got $STREAM_FP8)" >&2; exit 1 ;;
esac

case "$FAST_DECODE" in
  1|on|yes) DEC_ARGS=(--fast_decode) ;;
  *)        DEC_ARGS=() ;;
esac

# torch.compile is off by default because every upstream run script disables it
# and the reason was never recorded. Expect a long first-chunk warmup when on --
# read the timings from chunk 20+, not chunk 0.
case "$COMPILE" in
  1|on|yes) COMPILE_ARGS=() ;;
  *)        COMPILE_ARGS=(--no_compile) ;;
esac

source /workspace/soulx/env.sh
set +u

cd /workspace/soulx/SoulX-LiveAct
CUDA_VISIBLE_DEVICES=$AUX_GPU python persona_aux.py --port $AUX_PORT --preload \
  >> /workspace/soulx/persona_aux.log 2>&1 &
AUX_PID=$!
trap 'kill "$AUX_PID" 2>/dev/null || true' EXIT INT TERM

PYTHONPATH=/workspace/soulx/nosage:$PYTHONPATH USE_CHANNELS_LAST_3D=1 \
CUDA_VISIBLE_DEVICES=$GEN_GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python interactive_demo.py \
  --ckpt_dir /workspace/soulx/weights/LiveAct \
  --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
  --size "$STREAM_SIZE" --fps "$STREAM_FPS" --port "$STREAM_PORT" \
  --image "$STREAM_IMAGE" \
  --aux_device cpu --aux_url http://127.0.0.1:$AUX_PORT \
  "${FP8_ARGS[@]}" "${DEC_ARGS[@]}" "${COMPILE_ARGS[@]}" \
  $EXTRA_ARGS \
  --autostart
