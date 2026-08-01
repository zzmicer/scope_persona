#!/bin/bash
# Step 4: where does 1xH200 + --fast_decode cross the 2.0s chunk budget?
#
# Measured anchor: 720*416 (299,520 px) = 3.01s. Generation time scales ~linearly
# with pixel count, so the budget should be crossed near 199k px. The pacer wants
# SLACK, not a photo finish -- 1.8s is the target, not 2.0s.
#
# Each run is its own process on a fresh port; ports are not reused because a
# lingering socket from a killed run already cost one launch.
set -u
GPU="${GPU:-1}"
LOGDIR=/workspace/soulx/sweep
mkdir -p "$LOGDIR"

# "WxH stream_port aux_port"
RUNS=(
  "640*368 8093 8193"
  "592*336 8094 8194"
  "560*320 8095 8195"
  "320*576 8096 8196"
)

teardown() {
  pkill -f "[i]nteractive_demo.py" 2>/dev/null
  pkill -f "[p]ersona_aux.py" 2>/dev/null
  pkill -f "[r]un_single_fast.sh" 2>/dev/null
  sleep 20              # let CUDA free and sockets clear
}

teardown
for spec in "${RUNS[@]}"; do
  set -- $spec
  size=$1; sport=$2; aport=$3
  tag=$(echo "$size" | tr '*' 'x')
  log="$LOGDIR/$tag.log"
  echo "=== $size -> $log ==="

  GEN_GPU=$GPU STREAM_SIZE="$size" STREAM_PORT=$sport AUX_PORT=$aport \
    setsid nohup bash /workspace/soulx/run_single_fast.sh > "$log" 2>&1 < /dev/null &

  # wait for chunk 30, or bail on a hard failure
  for _ in $(seq 1 80); do
    if grep -qE "^chunk 0*30:" "$log" 2>/dev/null; then echo "  ok"; break; fi
    if grep -qE "Traceback|out of memory|Address already in use" "$log" 2>/dev/null; then
      echo "  FAILED:"; grep -mE1 "Traceback|out of memory|Address already in use" "$log"; break
    fi
    sleep 10
  done

  grep -E "^chunk (10|20|30):" "$log" | sed 's/^/  /'
  teardown
done

echo
echo "================ SWEEP SUMMARY ================"
printf "%-12s %8s %9s %8s %s\n" size px s/chunk realtime mem
for spec in "${RUNS[@]}"; do
  set -- $spec
  size=$1; tag=$(echo "$size" | tr '*' 'x')
  w=${size%\**}; h=${size#*\*}
  line=$(grep -E "^chunk 0*30:" "$LOGDIR/$tag.log" 2>/dev/null | tail -1)
  if [ -z "$line" ]; then printf "%-12s %8s %9s\n" "$size" $((w*h)) "no data"; continue; fi
  s=$(echo "$line" | sed -E 's/.* in ([0-9.]+)s .*/\1/')
  m=$(echo "$line" | sed -E 's/.*mem=([0-9.]+GB).*/\1/')
  rt=$(python3 -c "print(f'{2.0/$s:.2f}x')")
  printf "%-12s %8s %9s %8s %s\n" "$size" $((w*h)) "$s" "$rt" "$m"
done
