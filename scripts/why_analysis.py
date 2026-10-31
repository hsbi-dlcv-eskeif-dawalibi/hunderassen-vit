"""Why do the models reach more than 90 %? Model-based checks of ``make why`` (GPU recommended).

Writes to ``results/why/`` (read by ``scripts/report_why.py``):

``imagenet_head.csv``      original ImageNet-1k heads of the station-1 checkpoints, *no training*:
                           accuracy on the 22 breeds that are ImageNet classes
``test_max_sim.csv``       nearest train/validation image (CLIP ViT-B/16 cosine similarity) for
                           every test image: near-duplicate / leakage check
``duplicates.png``         the most similar test-train pairs, for visual inspection
``fewshot_backbones.csv``  linear probe on frozen ImageNet backbones (ViT-B/16, ResNet-50) with
                           k = 1..16 images per breed and with all images

The from-scratch runs are trained by ``make why`` itself (``configs/why/``).

Example::

    python scripts/why_analysis.py                      # everything
    python scripts/why_analysis.py --max-samples 100    # quick functional test
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR

#: Station-1 checkpoints; their original 1000-class heads are evaluated without training.
IMAGENET_MODELS = {
    "ViT-B/16": "google/vit-base-patch16-224",
    "ResNet-50": "microsoft/resnet-50",
    "ConvNeXt-T": "facebook/convnext-tiny-224",
}
#: Frozen backbones for the few-shot curve (CLIP's curve comes from station 3).
FEWSHOT_MODELS = {"ViT-B/16 (ImageNet)": "google/vit-base-patch16-224", "ResNet-50 (ImageNet)": "microsoft/resnet-50"}
SHOTS = (1, 2, 4, 8, 16)
SEEDS = (1, 2, 3)
CLIP_FOR_DUPLICATES = "openai/clip-vit-base-patch16"


def splits(size, mean, std, max_samples):
    """Train / validation / test of the 25 dog breeds with evaluation transforms only."""
    from animal_vit.data import build_datasets, build_transforms

    return build_datasets(
        "oxford_pets", build_transforms(size, mean, std, "none"), max_samples=max_samples, download=True
    )


def logits_of(model, dataset, device, batch_size, workers):
    """Raw logits and labels of a classifier on a dataset (numpy)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn)
    out, labels = [], []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            out.append(model(pixel_values=batch["pixel_values"].to(device)).logits.float().cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(out), np.concatenate(labels)


def imagenet_heads(args, device, workers) -> None:
    """Accuracy of the untouched ImageNet-1k heads on the breeds that are ImageNet classes."""
    import numpy as np
    import pandas as pd
    from transformers import AutoModelForImageClassification

    from animal_vit.models import load_image_processor
    from animal_vit.why import IMAGENET_IDS, check_imagenet_ids, imagenet_head_logits

    rows = []
    for name, ckpt in IMAGENET_MODELS.items():
        _, size, mean, std = load_image_processor(ckpt)
        datasets, class_names = splits(size, mean, std, args.max_samples)
        model = AutoModelForImageClassification.from_pretrained(ckpt)
        check_imagenet_ids(model.config.id2label)
        logits, labels = logits_of(model, datasets["test"], device, args.batch_size, workers)
        restricted, kept = imagenet_head_logits(logits, class_names)
        mask = np.isin(labels, kept)
        pos = {c: j for j, c in enumerate(kept)}
        target = np.array([pos[c] for c in labels[mask]])
        strict_ids = np.array([IMAGENET_IDS[class_names[c]][0] for c in labels[mask]])
        top5 = np.argsort(-logits[mask], axis=1)[:, :5]
        rows.append(
            {
                "model": name,
                "checkpoint": ckpt,
                "breeds": len(kept),
                "test_images": int(mask.sum()),
                # argmax over our 22 breeds only (same task as our classifiers, without training)
                "accuracy_22way": float((restricted[mask].argmax(1) == target).mean()),
                # argmax over all 1000 ImageNet classes must be exactly the right breed
                "accuracy_1000way_top1": float((logits[mask].argmax(1) == strict_ids).mean()),
                "accuracy_1000way_top5": float((top5 == strict_ids[:, None]).any(1).mean()),
            }
        )
        print(f"  {name:10s} no training: {100 * rows[-1]['accuracy_22way']:.2f} % (22-way)")
    pd.DataFrame(rows).to_csv(args.out / "imagenet_head.csv", index=False)


def duplicates(args, device, workers) -> None:
    """Nearest train/validation neighbour of every test image in CLIP embedding space."""
    import numpy as np
    import pandas as pd

    from animal_vit.clip_zeroshot import image_features, load_clip
    from animal_vit.why import nearest_neighbours

    model, _, size, mean, std = load_clip(CLIP_FOR_DUPLICATES, device)
    datasets, class_names = splits(size, mean, std, args.max_samples)
    feats = {s: image_features(model, datasets[s], device, args.batch_size, workers) for s in datasets}
    ref = np.concatenate([feats["train"][0], feats["validation"][0]])
    ref_split = ["train"] * len(feats["train"][0]) + ["validation"] * len(feats["validation"][0])
    ref_pos = list(range(len(feats["train"][0]))) + list(range(len(feats["validation"][0])))
    ref_labels = np.concatenate([feats["train"][1], feats["validation"][1]])
    sim, idx = nearest_neighbours(feats["test"][0], ref)
    df = pd.DataFrame(
        {
            "test_index": range(len(sim)),
            "max_similarity": sim,
            "neighbour_split": [ref_split[i] for i in idx],
            "neighbour_position": [ref_pos[i] for i in idx],
            "test_breed": [class_names[c] for c in feats["test"][1]],
            "neighbour_breed": [class_names[c] for c in ref_labels[idx]],
        }
    )
    df.to_csv(args.out / "test_max_sim.csv", index=False)
    for t in (0.90, 0.95, 0.98):
        print(f"  test images with a train/val neighbour >= {t}: {(sim >= t).sum()}")
    plot_pairs(df.nlargest(8, "max_similarity"), datasets, args.out / "duplicates.png")


def plot_pairs(top, datasets, path: Path) -> None:
    """Grid: test image (top) and its most similar train/validation image (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def image(split, position):
        ds = datasets[split]
        return ds.load(ds.indices[position])[0].convert("RGB")

    n = len(top)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4), squeeze=False)
    for j, (_, r) in enumerate(top.iterrows()):
        axes[0, j].imshow(image("test", int(r.test_index)))
        axes[0, j].set_title(f"test: {r.test_breed}\ncos {r.max_similarity:.3f}", fontsize=7)
        axes[1, j].imshow(image(r.neighbour_split, int(r.neighbour_position)))
        axes[1, j].set_title(f"{r.neighbour_split}: {r.neighbour_breed}", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Most similar test / train-val pairs (CLIP ViT-B/16)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fewshot(args, device, workers) -> None:
    """Linear probe on frozen ImageNet backbones with k images per breed (and with all images)."""
    import pandas as pd
    from transformers import AutoModel

    from animal_vit.clip_zeroshot import few_shot_indices, linear_probe
    from animal_vit.metrics import classification_metrics
    from animal_vit.models import load_image_processor
    from animal_vit.why import backbone_features

    rows = []
    for name, ckpt in FEWSHOT_MODELS.items():
        _, size, mean, std = load_image_processor(ckpt)
        datasets, _ = splits(size, mean, std, args.max_samples)
        model = AutoModel.from_pretrained(ckpt)
        train, val, test = (backbone_features(model, datasets[s], device, args.batch_size, workers) for s in datasets)
        logits, c, _ = linear_probe(train, val, test)
        m = classification_metrics(logits, test[1])
        rows.append({"model": name, "shots": "all", "seed": 0, "C": c, **_pick(m)})
        print(f"  {name}: all images {100 * m['accuracy']:.2f} %")
        for shots in SHOTS:
            for seed in SEEDS:
                idx = few_shot_indices(train[1], shots, seed)
                logits, c, _ = linear_probe((train[0][idx], train[1][idx]), val, test)
                rows.append(
                    {
                        "model": name,
                        "shots": shots,
                        "seed": seed,
                        "C": c,
                        **_pick(classification_metrics(logits, test[1])),
                    }
                )
            print(f"  {name}: k={shots} done")
    pd.DataFrame(rows).to_csv(args.out / "fewshot_backbones.csv", index=False)


def _pick(m: dict) -> dict:
    return {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}


def main() -> None:
    from animal_vit.evaluation import select_device

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "why")
    parser.add_argument("--max-samples", type=int, default=None, help="subset per split (functional test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", nargs="+", choices=("heads", "duplicates", "fewshot"), default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    workers = int(os.environ.get("ANIMAL_NUM_WORKERS", 4))
    parts = args.only or ("heads", "duplicates", "fewshot")
    if "heads" in parts:
        print("== ImageNet heads without training")
        imagenet_heads(args, device, workers)
    if "duplicates" in parts:
        print("== near duplicates test <-> train/validation")
        duplicates(args, device, workers)
    if "fewshot" in parts:
        print("== few-shot on frozen ImageNet backbones")
        fewshot(args, device, workers)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
