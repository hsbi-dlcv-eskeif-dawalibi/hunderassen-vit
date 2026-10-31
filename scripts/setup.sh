#!/usr/bin/env bash
# One-time setup of the project: Python environment, PyTorch, project package, dataset, checks.
#
#   make setup              # same as: bash scripts/setup.sh
#   NO_VENV=1 make setup    # use the current Python (e.g. Google Colab, conda, cluster module)
#
# Afterwards everything for station 1 runs with:   make station1
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ------------------------------------------------------------------ Python environment
if [[ -n "${COLAB_RELEASE_TAG:-}" || -d /content && -n "${COLAB_GPU:-}" ]]; then
  NO_VENV=1  # Colab already has PyTorch with CUDA
fi
PY=${PYTHON:-$(command -v python3 || command -v python)}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'

if [[ -z "${NO_VENV:-}" ]]; then
  if [[ ! -x .venv/bin/python ]]; then
    say "Creating virtual environment .venv"
    "$PY" -m venv .venv
  fi
  PY=.venv/bin/python
fi
"$PY" -m pip install -q --upgrade pip

# ------------------------------------------------------------------ PyTorch
if "$PY" -c 'import torch' 2>/dev/null; then
  say "PyTorch already installed: $("$PY" -c 'import torch; print(torch.__version__)')"
elif command -v nvidia-smi >/dev/null 2>&1; then
  say "Installing PyTorch with CUDA (NVIDIA GPU found)"
  "$PY" -m pip install torch torchvision
elif [[ "$(uname -s)" == "Darwin" ]]; then
  say "Installing PyTorch for macOS (Apple Silicon uses MPS)"
  "$PY" -m pip install torch torchvision
else
  say "Installing PyTorch for CPU (no NVIDIA GPU found - fine for tests, too slow for training)"
  "$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ------------------------------------------------------------------ project
say "Installing the project (src/animal_vit) and dev tools"
"$PY" -m pip install -q -e ".[dev]"
if [[ -d .git ]] && command -v pre-commit >/dev/null 2>&1; then pre-commit install >/dev/null || true; fi

say "Device check"
"$PY" - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("Apple MPS")
else:
    print("CPU only - training ViT-B/16 will be very slow; use the AI cluster or Colab for `make station1`.")
EOF

# ------------------------------------------------------------------ dataset + checks
say "Downloading Oxford-IIIT Pet (~800 MB, only the first time)"
"$PY" scripts/prepare_data.py

say "Running the tests"
"$PY" -m pytest -q

say "Done"
cat <<'EOF'
Next:
  make station1          # all station-1 runs, data quality report, results/ms1/REPORT.md, paper
  make station1 SEEDS=1  # quick pass with one seed
  make help              # all commands
EOF
