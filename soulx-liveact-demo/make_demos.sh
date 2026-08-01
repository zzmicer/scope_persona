#!/bin/bash
# One 20s action demo per resolution, same scripted actions in each so the clips
# are directly comparable. Each run is a cold start (~4-5 min model load) plus
# the recording itself, which at sub-realtime resolutions takes longer than 20s
# of wall clock -- recording is driven by frame count, not wall time.
set -u
ENVSH=/workspace/soulx/env.sh
GPU="${GPU:-1}"
OUT=/workspace/soulx/demos
mkdir -p "$OUT"

# "WxH stream_port aux_port"
RUNS=(
  "592*336 8112 8212"
  "560*320 8113 8213"
  "320*576 8114 8214"
)

teardown() {
  pkill -f "[i]nteractive_demo.py" 2>/dev/null
  pkill -f "[p]ersona_aux.py" 2>/dev/null
  pkill -f "[r]un_single_fast.sh" 2>/dev/null
  sleep 20
}

teardown
for spec in "${RUNS[@]}"; do
  set -- $spec
  size=$1; sport=$2; aport=$3
  tag=$(echo "$size" | tr '*' 'x')
  log="$OUT/$tag.run.log"
  echo "=========== $size ==========="

  GEN_GPU=$GPU STREAM_SIZE="$size" STREAM_PORT=$sport AUX_PORT=$aport COMPILE=0 \
    setsid nohup bash /workspace/soulx/run_single_fast.sh > "$log" 2>&1 < /dev/null &

  ok=0
  for _ in $(seq 1 90); do
    if grep -qE "^chunk 0*10:" "$log" 2>/dev/null; then ok=1; break; fi
    if grep -qE "Traceback|out of memory|Address already in use" "$log" 2>/dev/null; then
      echo "  START FAILED:"; grep -m1 -E "Traceback|out of memory|Address already in use" "$log"; break
    fi
    sleep 10
  done
  if [ "$ok" != "1" ]; then echo "  skipping $size"; teardown; continue; fi

  echo "  stream up, recording..."
  set +u; source $ENVSH; set -u
  python /workspace/uflash/record_demo.py \
    --port $sport --seconds 20 \
    --out "$OUT/persona_$tag.mp4" --workdir "$OUT/work_$tag" 2>&1 | sed 's/^/  /'

  teardown
done

echo
echo "=============== DEMOS ==============="
ls -la "$OUT"/*.mp4 2>/dev/null
