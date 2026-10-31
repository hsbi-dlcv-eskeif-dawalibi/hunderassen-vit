"""Tests for the metric helpers."""

import numpy as np
import pytest

from animal_vit.metrics import classification_metrics, expected_calibration_error


def test_perfect_predictions():
    labels = np.array([0, 1, 2, 2])
    logits = np.eye(3)[labels] * 20.0
    m = classification_metrics(logits, labels)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["cross_entropy"] < 1e-6
    assert m["ece"] < 1e-6


def test_macro_exposes_weak_minority_class():
    # 9 samples of class 0 (all correct), 1 sample of class 1 (wrong)
    labels = np.array([0] * 9 + [1])
    logits = np.tile([5.0, 0.0], (10, 1))
    m = classification_metrics(logits, labels)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["f1_macro"] < m["f1"]  # weighted average hides the failing class


def test_uniform_logits_have_max_entropy():
    m = classification_metrics(np.zeros((4, 15)), np.arange(4))
    assert m["prediction_entropy"] == pytest.approx(np.log(15))


def test_ece_of_overconfident_model():
    labels = np.array([0, 1, 0, 1])
    logits = np.tile([10.0, 0.0], (4, 1))  # ~100% confident, 50% correct
    assert expected_calibration_error(logits, labels) == pytest.approx(0.5, abs=1e-3)
