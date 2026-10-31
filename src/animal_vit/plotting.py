"""Plotting helpers shared by the training pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: scripts save figures, they never open windows
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, out_path: Path, dpi: int = 300) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return out_path


def plot_confusion_matrix(
    y_true, y_pred, class_names: Sequence[str], out_path: Path, normalize: str | None = None
) -> Path:
    """Confusion matrix of a run (one row/column per class, also for classes missing in a subset)."""
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize=normalize)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=list(class_names)).plot(
        cmap="Blues",
        xticks_rotation=70,
        values_format=".2f" if normalize else "d",
        ax=ax,
        colorbar=False,
    )
    ax.set_xlabel("Predicted label", fontsize=14, labelpad=12)
    ax.set_ylabel("True label", fontsize=14, labelpad=12)
    ax.set_title("Confusion matrix", fontsize=15, pad=10)
    return _save(fig, out_path)
