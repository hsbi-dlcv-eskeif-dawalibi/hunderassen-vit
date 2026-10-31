"""Tests for the "why so high?" building blocks (no downloads)."""

import numpy as np
import pytest

from animal_vit.config import ExperimentConfig
from animal_vit.why import (
    HARD_SUBSET,
    IMAGENET_IDS,
    NOT_IN_IMAGENET,
    check_imagenet_ids,
    imagenet_head_logits,
    nearest_neighbours,
)

DOG_BREEDS = [
    "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Boxer", "Chihuahua",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Miniature Pinscher", "Newfoundland", "Pomeranian", "Pug",
    "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]  # fmt: skip


def test_imagenet_mapping_covers_all_breeds_once():
    assert len(IMAGENET_IDS) == 22
    assert set(IMAGENET_IDS) | set(NOT_IN_IMAGENET) == set(DOG_BREEDS)
    assert not set(IMAGENET_IDS) & set(NOT_IN_IMAGENET)
    ids = [i for i, _ in IMAGENET_IDS.values()]
    assert len(set(ids)) == len(ids) and all(0 <= i < 1000 for i in ids)
    assert set(HARD_SUBSET) <= set(DOG_BREEDS)


def test_check_imagenet_ids_detects_wrong_order():
    good = {i: f"{name}, extra" for i, name in IMAGENET_IDS.values()}
    check_imagenet_ids(good)
    bad = dict(good)
    bad[162] = "tabby cat"
    with pytest.raises(ValueError, match="162"):
        check_imagenet_ids(bad)


def test_imagenet_head_logits_restricts_to_mapped_breeds():
    logits = np.zeros((2, 1000))
    logits[0, IMAGENET_IDS["Beagle"][0]] = 5.0
    restricted, kept = imagenet_head_logits(logits, DOG_BREEDS)
    assert restricted.shape == (2, 22) and len(kept) == 22
    assert DOG_BREEDS[kept[restricted[0].argmax()]] == "Beagle"
    assert all(DOG_BREEDS[k] not in NOT_IN_IMAGENET for k in kept)


def test_nearest_neighbours_finds_duplicate():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(50, 8))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    query = ref[[7, 21]] + 1e-6
    sim, idx = nearest_neighbours(query, ref, chunk=1)
    assert idx.tolist() == [7, 21]
    assert np.all(sim > 0.999)


def test_pretrained_flag_from_override():
    cfg = ExperimentConfig.from_yaml("configs/why/vit_b16_scratch.yaml", [])
    assert cfg.model.pretrained is False
    cfg = ExperimentConfig.from_yaml("configs/ms1/vit_b16.yaml", ["model.pretrained=false"])
    assert cfg.model.pretrained is False
    assert ExperimentConfig.from_yaml("configs/ms1/vit_b16.yaml", []).model.pretrained is True
