"""Generate shape-vs-texture cue-conflict images with neural style transfer.

Each image keeps the *content* (shape) of a test image of breed A and takes the *style*
(texture) of a test image of a different breed B (Gatys et al., lecture 05). Output:
``data/cue_conflict/<id>.png`` plus ``index.csv`` with shape and texture labels.

Style transfer is an optimisation per image (~5-10 s on a GPU), so a balanced subset
is used: ``--per-class`` content images per breed.

Example::

    python scripts/make_cue_conflict.py --per-class 10      # 25 dog breeds -> 250 images
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import DATA_ROOT


def main() -> None:
    import numpy as np
    import pandas as pd

    from animal_vit.data import build_datasets, build_transforms
    from animal_vit.evaluation import select_device
    from animal_vit.style_transfer import StyleTransfer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "cue_conflict")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--species",
        default="dogs",
        choices=["dogs", "cats", "all"],
        help="must match data.species of the evaluated runs (labels are stored as class indices)",
    )
    args = parser.parse_args()

    datasets, class_names = build_datasets(
        "oxford_pets",
        build_transforms(args.size, (0, 0, 0), (1, 1, 1), "none"),
        download=True,
        species=args.species,
    )
    test = datasets["test"]
    labels = np.array(test.labels)
    rng = np.random.default_rng(args.seed)
    transfer = StyleTransfer(select_device(args.device), args.size)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cls in range(len(class_names)):
        contents = rng.choice(np.flatnonzero(labels == cls), size=args.per_class, replace=False)
        for content_pos in contents:
            style_pos = int(rng.choice(np.flatnonzero(labels != cls)))
            content_img, shape_label, _ = test.load(test.indices[content_pos])
            style_img, texture_label, _ = test.load(test.indices[style_pos])
            name = f"{len(rows):05d}.png"
            transfer(content_img, style_img, steps=args.steps).save(args.out / name)
            rows.append(
                {
                    "file": name,
                    "shape_label": int(shape_label),
                    "texture_label": int(texture_label),
                    "shape_class": class_names[shape_label],
                    "texture_class": class_names[texture_label],
                }
            )
            print(f"[{len(rows)}] shape={class_names[shape_label]:28s} texture={class_names[texture_label]}")
    pd.DataFrame(rows).to_csv(args.out / "index.csv", index=False)
    print(f"wrote {len(rows)} images to {args.out}")


if __name__ == "__main__":
    main()
