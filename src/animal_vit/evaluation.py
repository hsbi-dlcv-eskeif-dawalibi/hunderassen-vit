"""Inference helpers shared by the analysis scripts."""

from __future__ import annotations

import numpy as np


def select_device(preferred: str | None = None) -> str:
    """CUDA if available, then Apple MPS, then CPU (unless ``preferred`` is given)."""
    import torch

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict(model, dataset, device: str, batch_size: int = 64, num_workers: int | None = None):
    """Return ``(logits, labels)`` as numpy arrays for a dataset of dict items."""
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn

    if num_workers is None:
        import os

        num_workers = int(os.environ.get("ANIMAL_NUM_WORKERS", 4))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    all_logits, all_labels = [], []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def test_dataset(cfg, size: int, mean, std, with_mask: bool = False):
    """The (untouched) test split of the dataset a run was trained on."""
    from animal_vit.data import build_datasets, build_transforms

    d = cfg.data
    datasets, class_names = build_datasets(
        d.name,
        build_transforms(size, mean, std, "none"),
        val_fraction=d.val_fraction,
        split_seed=d.split_seed,
        download=d.download,
        with_mask=with_mask,
        size=size,
        species=d.species,
    )
    return datasets["test"], class_names
