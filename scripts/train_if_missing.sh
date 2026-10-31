#!/usr/bin/env bash
# Train one run unless it is already finished (resumable pipelines, used by the Makefile).
#
#   RUNS=runs OVERRIDES="training.epochs=1" bash scripts/train_if_missing.sh <name> <config.yaml> [key=value ...]
#
# The run is written to $RUNS/<name>. Extra key=value pairs and $OVERRIDES are passed to
# scripts/train.py --set (OVERRIDES last, so it wins - used by `make smoke`).
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1
config=$2
shift 2
runs=${RUNS:-runs}
if [[ -f "$runs/$name/metrics.json" ]]; then
  echo "skip $name (done)"
  exit 0
fi
echo "==> $name ($config)"
# shellcheck disable=SC2086  # OVERRIDES is a space-separated list of key=value pairs
exec ${PYTHON:-python3} scripts/train.py "$config" --set name="$name" output_dir="$runs/$name" "$@" ${OVERRIDES:-}
