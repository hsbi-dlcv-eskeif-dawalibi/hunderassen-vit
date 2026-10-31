"""Tests for YAML config loading (no deep-learning dependencies needed)."""

from pathlib import Path

import pytest

from animal_vit.config import ExperimentConfig
from animal_vit.paths import CONFIGS_DIR

#: Configs read by scripts/train.py (clip.yaml has its own schema, see test_clip_config).
ALL_CONFIGS = [p for p in sorted(CONFIGS_DIR.rglob("*.yaml")) if p.name != "clip.yaml"]


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_all_configs_load(path: Path):
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.name
    assert isinstance(cfg.training.learning_rate, float)


def test_scientific_notation_is_parsed_as_float(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("training:\n  learning_rate: 2e-4\n")  # PyYAML reads this as a string
    assert ExperimentConfig.from_yaml(path).training.learning_rate == pytest.approx(2e-4)


def test_overrides(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("name: a\n")
    cfg = ExperimentConfig.from_yaml(
        path,
        [
            "name=b",
            "modifications.attention_temperature=1.5",
            "model.patch_size=none",
            "training.extra.weight_decay=0.05",
        ],
    )
    assert cfg.name == "b"
    assert cfg.modifications.attention_temperature == 1.5
    assert cfg.model.patch_size is None
    assert cfg.training.extra == {"weight_decay": 0.05}


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        ExperimentConfig.from_dict({"training": {"learnig_rate": 1e-3}})


def test_none_is_a_string_for_string_fields(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  augmentation: none\n")  # YAML reads this as null
    cfg = ExperimentConfig.from_yaml(path, ["distillation.teacher=none"])
    assert cfg.data.augmentation == "none"  # str field: the preset name
    assert cfg.distillation.teacher is None  # optional field: really no teacher
    cfg = ExperimentConfig.from_yaml(path, ["data.augmentation=none"])
    assert cfg.data.augmentation == "none"
