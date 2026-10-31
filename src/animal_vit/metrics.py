"""Classification metrics.

The original notebooks reported *weighted* precision/recall/F1 only. Because the test
split is heavily imbalanced, weak minority classes (Gorilla, Mouse) disappear in a
weighted average, so macro averages are reported as well.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    lse = np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    return logits - lse


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Accuracy, weighted + macro P/R/F1, cross entropy and mean prediction entropy."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    preds = logits.argmax(axis=-1)

    metrics: dict[str, float] = {"accuracy": float(accuracy_score(labels, preds))}
    for average in ("weighted", "macro"):
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=average, zero_division=0)
        suffix = "" if average == "weighted" else "_macro"
        metrics.update({f"precision{suffix}": float(p), f"recall{suffix}": float(r), f"f1{suffix}": float(f1)})

    log_probs = _log_softmax(logits)
    metrics["cross_entropy"] = float(-log_probs[np.arange(len(labels)), labels].mean())
    metrics["prediction_entropy"] = float(-(np.exp(log_probs) * log_probs).sum(axis=-1).mean())
    metrics["ece"] = expected_calibration_error(logits, labels)
    return metrics


def expected_calibration_error(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    ECE = sum_b |B_b| / N * |acc(B_b) - conf(B_b)|. 0 means perfectly calibrated.
    """
    probs = np.exp(_log_softmax(np.asarray(logits, dtype=np.float64)))
    confidences = probs.max(axis=-1)
    correct = probs.argmax(axis=-1) == np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def make_compute_metrics() -> Callable[[Any], dict[str, float]]:
    """Adapter for ``transformers.Trainer(compute_metrics=...)``."""

    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits, labels = eval_prediction.predictions, eval_prediction.label_ids
        # With output_attentions=True the model returns a tuple (logits, attentions, ...).
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return classification_metrics(logits, labels)

    return compute_metrics
