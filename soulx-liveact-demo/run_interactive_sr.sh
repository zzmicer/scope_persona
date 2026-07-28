#!/bin/bash
# SoulX-LiveAct + UltraFlash latent SR, streaming at 2x native resolution.
#
#   GPU 0  persona_aux (Qwen + Kokoro)
#   GPU 1  SR sidecar  (latent upsampler + sparse SR DiT + v3 tiny decoder)
#   GPU 2,3 SoulX generator, sequence-parallel
#
# Each stage is its own process on purpose: a second CUDA context inside a
# torchrun rank destabilizes NCCL, which is why persona_aux was already split
# out and why the SR runs over loopback rather than in-process.
#
# Budget per 32-frame (2.0s) chunk, measured: generator 1.99s on GPUs 2+3,
# SR stage 0.78s on GPU 1. The SR overlaps nothing today -- it is a serial
# round-trip inside the chunk loop -- so expect the generator's own decode
# (~0.45s, now skipped) to partly pay for it. Watch the chunk log.
STREAM_SIZE="${STREAM_SIZE:-720*416}"
STREAM_FPS="${STREAM_FPS:-16}"
STREAM_PORT="${STREAM_PORT:-8090}"
STREAM_IMAGE="${STREAM_IMAGE:-/workspace/soulx_setup/chano39-Anime-Original-anime-9101906.png}"
STREAM_FP8="${STREAM_FP8:-blocks}"
SR_SCALE="${SR_SCALE:-2}"
SR_PORT="${SR_PORT:-8092}"
SR_GPU="${SR_GPU:-1}"
AUX_GPU="${AUX_GPU:-0}"
GEN_GPUS="${GEN_GPUS:-2,3}"

case "$STREAM_FP8" in
  off|0|"") FP8_ARGS=(--no_fp8_gemm) ;;
  blocks|1) FP8_ARGS=(--fp8_scope blocks) ;;
  all)      FP8_ARGS=(--fp8_scope all) ;;
  *) echo "STREAM_FP8 must be off|blocks|all (got $STREAM_FP8)" >&2; exit 1 ;;
esac

source /workspace/soulx/env.sh
set +u

# --- SR sidecar -------------------------------------------------------------
PYTHONPATH=/workspace/soulx/nosage:$PYTHONPATH CUDA_VISIBLE_DEVICES=$SR_GPU \
  python /workspace/uflash/sr_service.py --port "$SR_PORT" --scale "$SR_SCALE" \
  >> /workspace/soulx/sr_service.log 2>&1 &
SR_PID=$!

# --- persona aux ------------------------------------------------------------
cd /workspace/soulx/SoulX-LiveAct
CUDA_VISIBLE_DEVICES=$AUX_GPU python persona_aux.py --port 8091 --preload \
  >> /workspace/soulx/persona_aux.log 2>&1 &
AUX_PID=$!
trap 'kill "$AUX_PID" "$SR_PID" 2>/dev/null || true' EXIT INT TERM

# The generator blocks on /sr for every chunk, so refuse to start until the
# sidecar has its weights resident -- otherwise chunk 0 eats the load time.
echo "waiting for SR sidecar on :$SR_PORT ..."
for _ in $(seq 1 120); do
  if curl -sf --max-time 3 "http://127.0.0.1:$SR_PORT/health" >/dev/null 2>&1; then
    echo "SR sidecar ready"; break
  fi
  sleep 5
done

PYTHONPATH=/workspace/soulx/nosage:$PYTHONPATH USE_CHANNELS_LAST_3D=1 \
CUDA_VISIBLE_DEVICES=$GEN_GPUS \
torchrun --nproc_per_node=2 --master_port=29618 \
  interactive_demo.py \
  --ckpt_dir /workspace/soulx/weights/LiveAct \
  --wav2vec_dir /workspace/soulx/weights/chinese-wav2vec2-base \
  --size "$STREAM_SIZE" --fps "$STREAM_FPS" --port "$STREAM_PORT" \
  --image "$STREAM_IMAGE" \
  --aux_device cpu --aux_url http://127.0.0.1:8091 \
  --sr_url "http://127.0.0.1:$SR_PORT" --sr_scale "$SR_SCALE" \
  "${FP8_ARGS[@]}" --no_compile \
  --autostart
