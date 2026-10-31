"""Measure the shape bias of trained runs on the cue-conflict images.

``shape_bias = shape decisions / (shape + texture decisions)``; 1.0 = decides purely by
shape (like humans), 0.0 = purely by texture. Writes ``<run>/shape_bias.json``.

Example::

    python scripts/make_cue_conflict.py --per-class 10   # once
    python scripts/evaluate_shape_bias.py runs/ms1_*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import DATA_ROOT


class CueConflictDataset:
    """Cue-conflict images from ``index.csv``; the label of each image is its *shape* class."""

    def __init__(self, folder: Path, transform):
        import pandas as pd

        self.folder, self.transform = folder, transform
        self.index = pd.read_csv(folder / "index.csv")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.index.iloc[i]
        image = Image.open(self.folder / row["file"]).convert("RGB")
        return {"pixel_values": self.transform(image), "label": int(row["shape_label"])}


def main() -> None:
    from animal_vit.data import build_transforms
    from animal_vit.evaluation import predict, select_device
    from animal_vit.style_transfer import shape_bias
    from animal_vit.training import load_run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--images", type=Path, default=DATA_ROOT / "cue_conflict")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    for run in args.runs:
        model, _, _, (size, mean, std) = load_run(run, device)
        dataset = CueConflictDataset(args.images, build_transforms(size, mean, std, "none")["eval"])
        logits, shape_labels = predict(model, dataset, device)
        result = shape_bias(logits.argmax(-1), shape_labels, dataset.index["texture_label"].to_numpy())
        (run / "shape_bias.json").write_text(json.dumps(result, indent=2))
        print(
            f"{run.name:30s} shape bias {result['shape_bias']:.3f}  "
            f"(shape acc {result['shape_accuracy']:.3f}, texture acc {result['texture_accuracy']:.3f})"
        )


if __name__ == "__main__":
    main()
