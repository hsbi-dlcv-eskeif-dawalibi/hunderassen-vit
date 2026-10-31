"""The station-2 teacher is chosen from station 1 by validation accuracy, never by test accuracy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    sys.path.insert(0, str(ROOT / "scripts"))  # scripts import _bootstrap
    spec = importlib.util.spec_from_file_location("select_teacher", ROOT / "scripts" / "select_teacher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_run(runs: Path, name: str, val: float, test: float, weights: bool = True, loss: float | None = None) -> None:
    run = runs / name
    (run / "model").mkdir(parents=True)
    if weights:
        (run / "model" / "model.safetensors").write_bytes(b"")
    (run / "metrics.json").write_text(json.dumps({"best_validation_metric": val, "test_metrics": {"accuracy": test}}))
    if loss is not None:
        (run / "history.json").write_text(json.dumps([{"eval_accuracy": val, "eval_loss": loss}]))


def test_teacher_selected_by_validation_not_test(tmp_path):
    for s in (1, 2):
        _fake_run(tmp_path, f"ms1_vit_b16_s{s}", val=0.90, test=0.99)  # best on test, not on validation
        _fake_run(tmp_path, f"ms1_vit_b16_lora_r8_s{s}", val=0.93, test=0.91)
        _fake_run(tmp_path, f"ms1_vit_b16_lora_r8_aug_strong_s{s}", val=0.99, test=0.99)  # ablation: never a teacher
        _fake_run(tmp_path, f"ms1_convnext_t_s{s}", val=0.92, test=0.92)
        _fake_run(tmp_path, f"ms1_resnet50_s{s}", val=0.88, test=0.88)
    _fake_run(tmp_path, "ms1_vit_b16_head_s1", val=0.99, test=0.5, weights=False)  # no weights: not loadable
    choice = _load_script().select(tmp_path)
    assert choice["vit"]["group"] == "ms1_vit_b16_lora_r8"
    assert choice["cnn"]["group"] == "ms1_convnext_t"
    assert set(choice["vit"]["runs"]) == {"1", "2"}


def test_tie_in_validation_accuracy_broken_by_validation_loss(tmp_path):
    for s in (1, 2):
        _fake_run(tmp_path, f"ms1_convnext_t_s{s}", val=0.9747, test=0.99, loss=0.30)  # alphabetically first
        _fake_run(tmp_path, f"ms1_resnet50_s{s}", val=0.9747, test=0.90, loss=0.10)
    choice = _load_script().select(tmp_path)
    assert choice["cnn"]["group"] == "ms1_resnet50"
