"""Data quality report of the station-1 dataset (25 dog breeds of Oxford-IIIT Pet).

Writes ``results/data_quality/``:

``README.md``          findings in plain language + tables (for the "Dataset" section of the paper)
``per_breed.csv``      images per breed and split
``images.csv``         one row per image (mode, size, hash, pet fraction)
``class_counts.png``   bar chart of images per breed
``examples.png``       one example per breed with its segmentation mask outline

Example::

    python scripts/data_report.py              # dogs (project default)
    python scripts/data_report.py --species all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import PETS_DIR, RESULTS_DIR


def plot_class_counts(per_breed, out_path: Path) -> None:
    """Horizontal bar chart: images per breed, stacked by split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = [c for c in ("trainval", "test") if c in per_breed.columns]
    data = per_breed.sort_values("total")
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(data) + 1))
    left = None
    for split in splits:
        ax.barh(data.index, data[split], left=left, label=split)
        left = data[split] if left is None else left + data[split]
    ax.set(xlabel="images", title="Images per breed")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_examples(sources: dict, out_path: Path, size: int = 160) -> None:
    """One training image per breed with the outline of its pet mask."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    source = sources["trainval"]
    first = {}
    for i, label in enumerate(source.labels):
        first.setdefault(label, i)
    n = len(first)
    cols = 5
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (label, i) in zip(np.ravel(axes), sorted(first.items())):
        pos = source.positions[i]
        image = Image.open(source.base._images[pos]).convert("RGB").resize((size, size))
        trimap = np.asarray(Image.open(source.base._segs[pos]).resize((size, size), Image.NEAREST))
        ax.imshow(image)
        ax.contour(np.isin(trimap, (1, 3)), levels=[0.5], colors="yellow", linewidths=0.8)
        ax.set_title(source.class_names[label], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    import pandas as pd

    from animal_vit.data import _PetsSource
    from animal_vit.data_quality import findings, pets_records, summarize

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", default="dogs", choices=["dogs", "cats", "all"])
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "data_quality")
    args = parser.parse_args()

    sources = {
        split: _PetsSource(PETS_DIR, split, download=True, species=args.species) for split in ("trainval", "test")
    }
    print("reading images ...")
    records = list(pets_records(sources))
    summary = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(args.out / "images.csv", index=False)
    per_breed = summary.pop("per_breed")
    per_breed.to_csv(args.out / "per_breed.csv")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plot_class_counts(per_breed, args.out / "class_counts.png")
    plot_examples(sources, args.out / "examples.png")

    lines = [
        f"# Datenqualität: Oxford-IIIT Pet, `{args.species}`",
        "",
        f"{summary['images']} Bilder, {summary['breeds']} Rassen. Splits: "
        + ", ".join(f"{k} {v}" for k, v in summary["per_split"].items())
        + ". Quelle: Parkhi et al., CVPR 2012, CC BY-SA 4.0.",
        "",
        "## Befunde",
        "",
        *[f"- {f}" for f in findings(summary)],
        "",
        "## Bilder pro Rasse",
        "",
        "| Rasse | " + " | ".join(per_breed.columns) + " |",
        "|---" * (len(per_breed.columns) + 1) + "|",
        *[f"| {breed} | " + " | ".join(str(int(v)) for v in row) + " |" for breed, row in per_breed.iterrows()],
        "",
        "Abbildungen: `class_counts.png`, `examples.png` (gelb: Umriss der Maske). Rohdaten: `images.csv`.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
