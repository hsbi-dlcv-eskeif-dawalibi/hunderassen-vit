"""Shared test setup: ``src/`` on the path and fixtures for tiny models and a fake dataset."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: Image size of the tiny test models.
SIZE = 32


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A randomly initialised 2-block ViT (3 classes, 32 px) saved as a Hugging Face checkpoint."""
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    path = tmp_path_factory.mktemp("tiny_vit")
    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        image_size=SIZE,
        patch_size=16,
        num_labels=3,
    )
    ViTForImageClassification(config).save_pretrained(path)
    ViTImageProcessor(size={"height": SIZE, "width": SIZE}).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory) -> Path:
    """A 3-class ImageFolder dataset (train/validation/test) of random 40x40 images."""
    from PIL import Image

    root = tmp_path_factory.mktemp("image_folder")
    rng = np.random.default_rng(0)
    for split, n in (("train", 6), ("validation", 3), ("test", 3)):
        for cls in ("cat", "dog", "fox"):
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(folder / f"{i}.png")
    return root


@pytest.fixture()
def use_fake_data(monkeypatch, fake_dataset):
    """Point the ``image_folder`` loader at the fake dataset."""
    from animal_vit import data as data_module

    monkeypatch.setattr(data_module, "DATA_DIR", fake_dataset)
