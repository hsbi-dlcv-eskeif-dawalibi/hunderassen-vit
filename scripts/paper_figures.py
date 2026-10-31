"""German, column-sized figures for the station-1 paper (``paper/ms1``).

The report scripts write English figures into ``results/ms1`` etc. This script redraws the figures
that appear in the paper from the same result files, with German labels and sizes that match the
column width of the CVPR template (8.4 cm), so that the text is printed at its true size.

Inputs (all produced by ``make ms1`` / ``make why`` / ``make explain`` / ``make data-report``):
``results/ms1/methods.csv``, ``runs/*/history.json``, ``results/why/fewshot_backbones.csv``, the Oxford-IIIT Pet files
and the Grad-CAM figures of ``results/explain``. Output: ``results/paper/``.

    python scripts/paper_figures.py
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import PETS_DIR, RESULTS_DIR, RUNS_DIR

COLUMN = 3.35  # inches: width of one column of the CVPR template

LABELS = {
    "ms1_vit_b16_head": "ViT Linear Probing",
    "ms1_vit_b16_lora_r4": "ViT LoRA r=4",
    "ms1_vit_b16_lora_r8": "ViT LoRA r=8",
    "ms1_vit_b16_lora_r16": "ViT LoRA r=16",
    "ms1_vit_b16_lora_r8_all": "ViT LoRA r=8 (alle Layer)",
    "ms1_vit_b16": "ViT volles Fine-Tuning",
    "ms1_resnet50": "ResNet-50 (CNN)",
    "ms1_convnext_t": "ConvNeXt-T (CNN)",
}
# One colour and marker per method, identical in every figure (colour-blind and grayscale friendly).
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
STYLE = {g: (m, c) for g, m, c in zip(LABELS, MARKERS, COLORS)}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "mathtext.default": "regular",
        }
    )


def _comma(ax, axis: str = "y", digits: int = 0) -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:.{digits}f}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def efficiency(methods_csv: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(methods_csv).set_index("group")
    fig, ax = plt.subplots(figsize=(COLUMN, 3.15))
    for group, label in LABELS.items():
        r = df.loc[group]
        marker, color = STYLE[group]
        ax.errorbar(
            r["trainable_params_mean"], 100 * r["accuracy_mean"], yerr=100 * r["accuracy_std"],
            fmt=marker, color=color, capsize=2.5, markersize=4.5, lw=0.9, label=label,
        )  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(8e3, 2.5e8)
    ax.set(xlabel="Trainierbare Parameter (logarithmische Skala)", ylabel="Test-Accuracy (%)")
    _comma(ax, "y", 1)
    ax.grid(alpha=0.3, which="major", lw=0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]  # marker only; error bars are explained in the caption
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              columnspacing=1.2, handletextpad=0.4, labelspacing=0.3)  # fmt: skip
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def curves(runs_dir: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for run in sorted(runs_dir.glob("ms1_*_s[0-9]")):
        group = re.sub(r"_s\d+$", "", run.name)
        history = run / "history.json"
        if group not in LABELS or not history.exists():
            continue
        for entry in json.loads(history.read_text()):
            if "eval_accuracy" in entry:
                rows.append({"group": group, "epoch": round(entry["epoch"]), "acc": entry["eval_accuracy"]})
    df = pd.DataFrame(rows).groupby(["group", "epoch"]).acc.mean().reset_index()
    fig, ax = plt.subplots(figsize=(COLUMN, 2.7))
    for group in LABELS:
        marker, color = STYLE[group]
        part = df[df.group == group]
        ax.plot(part.epoch, 100 * part.acc, marker=marker, color=color, markersize=3, lw=0.9, label=LABELS[group])
    ax.set_xticks(range(1, 11))
    ax.set(xlabel="Epoche", ylabel="Validation-Accuracy (%)", ylim=(68, 99.5))
    _comma(ax, "y", 0)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.9, fontsize=5.4, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fewshot(csv_path: Path, out: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    series = (
        ("ViT-B/16 (ImageNet)", "ViT-B/16", "#1f4e79", "o"),
        ("ResNet-50 (ImageNet)", "ResNet-50", "#c0392b", "s"),
    )
    ticks = [1, 2, 4, 8, 16, 32]
    for key, label, color, marker in series:
        part = df[df.model == key]
        k = part[part.shots != "all"].assign(shots=lambda d: d.shots.astype(int))
        agg = k.groupby("shots").accuracy.agg(["mean", "std"])
        full = float(part[part.shots == "all"].accuracy.iloc[0])
        x = list(agg.index) + [32]
        y = [100 * v for v in agg["mean"]] + [100 * full]
        err = [100 * v for v in agg["std"].fillna(0)] + [0]
        ax.errorbar(x, y, yerr=err, marker=marker, color=color, capsize=2, markersize=3.5, lw=0.9, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks[:-1]] + ["alle"])
    ax.minorticks_off()
    ax.axvline(22.6, color="grey", lw=0.5, ls=":")
    ax.set(xlabel="Trainingsbilder pro Rasse", ylabel="Test-Accuracy (%)", ylim=(80, 97.5))
    _comma(ax, "y", 0)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def examples(out: Path, size: int = 192) -> None:
    """One training image per dog breed (first one in ``trainval.txt``) with the outline of its pet mask."""
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    root = PETS_DIR / "oxford-iiit-pet"
    first: dict[str, str] = {}
    for line in (root / "annotations" / "trainval.txt").read_text().splitlines():
        name, _cls, species, _breed = line.split()
        if species == "2":  # dogs
            breed = re.sub(r"_\d+$", "", name)
            first.setdefault(breed, name)
    cols = 5
    rows = -(-len(first) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(COLUMN, 3.75))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (breed, name) in zip(np.ravel(axes), sorted(first.items())):
        image = Image.open(root / "images" / f"{name}.jpg").convert("RGB").resize((size, size))
        trimap = np.asarray(Image.open(root / "annotations" / "trimaps" / f"{name}.png").resize((size, size), Image.NEAREST))
        ax.imshow(image)
        ax.contour(np.isin(trimap, (1, 3)), levels=[0.5], colors="yellow", linewidths=0.4)
        title = "\n".join(textwrap.wrap(breed.replace("_", " ").title(), 15))
        ax.set_title(title, fontsize=4.6, pad=1.5, linespacing=0.95)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.04, hspace=0.55)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def crop_title(src: Path, out: Path, top: int) -> None:
    """Drop the English title line of a rendered figure (the caption explains it in German)."""
    from PIL import Image

    image = Image.open(src)
    image.crop((0, top, image.width, image.height)).save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "paper")
    args = parser.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    efficiency(RESULTS_DIR / "ms1" / "methods.csv", args.out / "effizienz.png")
    curves(args.runs, args.out / "lernkurven.png")
    fewshot(RESULTS_DIR / "why" / "fewshot_backbones.csv", args.out / "fewshot.png")
    examples(args.out / "beispiele.png")
    crop_title(RESULTS_DIR / "explain" / "head_examples.png", args.out / "gradcam.png", top=52)
    crop_title(RESULTS_DIR / "explain" / "contrastive_convnext_t.png", args.out / "kontrastiv.png", top=0)
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
