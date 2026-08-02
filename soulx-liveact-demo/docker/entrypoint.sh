#!/usr/bin/env bash
# Prepare the mounted volume, then hand off to scope-soulx.
#
# The volume is the only mutable state: weights, tiny decoders, the reference
# image, and the inductor cache that makes torch.compile warmup a one-time cost.
set -euo pipefail

ROOT="${SOULX_ROOT:-/workspace/soulx}"
mkdir -p "$ROOT"/{weights,decoders,assets,logs,.hf,.inductor_cache,.triton_cache}

# Seed the reference image from the ones vendored in the image, so a fresh volume
# can run immediately. An image already on the volume always wins.
if ! compgen -G "$ROOT/assets/*.png" > /dev/null 2>&1; then
  cp -n /opt/soulx/assets/*.png "$ROOT/assets/" 2>/dev/null || true
fi

if [[ ! -d "$ROOT/weights/LiveAct" ]]; then
  echo "entrypoint: no weights at $ROOT/weights/LiveAct" >&2
  echo "entrypoint: run once ->  docker run ... soulx-demo fetch" >&2
fi

exec /opt/soulx/scope-soulx "$@"
