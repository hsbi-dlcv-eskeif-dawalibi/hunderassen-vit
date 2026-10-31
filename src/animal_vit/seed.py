"""Reproducibility helpers."""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (if installed) for reproducible runs.

    Note: random head pruning used to run *before* the Trainer seeded anything, so
    every run pruned different heads. Call this first in every entry point.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
