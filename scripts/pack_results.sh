#!/usr/bin/env bash
# Bundle every result of runs/ WITHOUT model weights and checkpoints, plus results/, into
# results_bundle_<date>.tar.gz - small enough to download from a cloud GPU (RunPod).
#
#   bash scripts/pack_results.sh [runs-dir]        # = make pack
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${1:-runs}
OUT="results_bundle_$(date +%Y-%m-%d_%H%M).tar.gz"

tar -czf "$OUT" \
  --exclude='*/model/*' --exclude='*/checkpoints/*' --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' \
  "$RUNS" results 2>/dev/null || tar -czf "$OUT" --exclude='*/model/*' --exclude='*/checkpoints/*' "$RUNS" results

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "download it, then locally:  tar -xzf $OUT"
