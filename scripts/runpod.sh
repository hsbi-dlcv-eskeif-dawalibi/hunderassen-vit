#!/usr/bin/env bash
# Start the whole pipeline on a RunPod GPU pod (see docs/RUNPOD.md).
#
# Run inside the pod, in the cloned repository on the persistent volume (/workspace/hunderassen-vit):
#   bash scripts/runpod.sh                 # make all          (stations 1-3, summary, bundle)
#   bash scripts/runpod.sh station1        # only station 1
#   bash scripts/runpod.sh all SEEDS=1     # quick pass with one seed
#
# Training runs in a tmux session, so it continues when the browser tab is closed:
#   tmux attach -t train      (detach again with Ctrl+b, then d)
set -euo pipefail
cd "$(dirname "$0")/.."

# Everything that should survive a pod restart lives on the volume (/workspace).
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/workspace/.cache/pip}
export ANIMAL_NUM_WORKERS=${ANIMAL_NUM_WORKERS:-8}
export TOKENIZERS_PARALLELISM=false

if ! command -v tmux >/dev/null 2>&1; then
  echo "==> installing tmux"
  (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null
fi

# Use the PyTorch of the RunPod image (no virtual environment, no second CUDA download).
if [[ ! -f .setup_done ]]; then
  NO_VENV=1 bash scripts/setup.sh
  touch .setup_done
fi

TARGET=${1:-all}
shift || true
mkdir -p logs
LOG="logs/${TARGET}_$(date +%Y-%m-%d_%H%M).log"
CMD="export HF_HOME=$HF_HOME ANIMAL_NUM_WORKERS=$ANIMAL_NUM_WORKERS TOKENIZERS_PARALLELISM=false; \
make $TARGET $* 2>&1 | tee $LOG; echo; echo '==> finished: $TARGET (log: $LOG)'"

if tmux has-session -t train 2>/dev/null; then
  echo "A tmux session 'train' is already running: tmux attach -t train"
  exit 1
fi
tmux new-session -d -s train "bash -c \"$CMD\"; bash"
cat <<MSG

==> started 'make $TARGET $*' in tmux session 'train'
    watch:     tmux attach -t train        (leave: Ctrl+b, then d)
    log file:  $LOG
    GPU:       watch -n 5 nvidia-smi
    resume:    run this script again - finished runs are skipped
MSG
