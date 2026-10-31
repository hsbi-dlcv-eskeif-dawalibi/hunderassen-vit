"""Why this breed and not the other? Explanations for trained runs (``make explain``, GPU recommended).

For the station-1 teachers (best ViT and best CNN by validation, ``results/ms1/best_models.json``)
and the other CNN, seed 1, writes to ``results/explain/``:

``contrastive_<model>.png``  for the most frequent confusions (A -> B): image, Grad-CAM for A,
                             Grad-CAM for B, contrastive Grad-CAM A - B, occlusion for A
``head_examples.png``        validation images with the annotated head box and each model's map
``head_focus.csv``           saliency mass on head / rest of the dog / background (validation split,
                             the only split with head boxes)
``agreement.csv``            do two models look at the same place? (IoU of the top-20 % regions)
``faithfulness.csv``         deletion test: class probability when the most salient patches are
                             greyed out first, per method, against random deletion (test split)

Methods: Grad-CAM (pytorch-grad-cam), occlusion (Captum), attention rollout (ViT, own code), see
``src/animal_vit/explain.py`` for sources.

Example::

    python scripts/explain.py                                 # best models from results/ms1/best_models.json
    python scripts/explain.py --runs runs/ms1_vit_b16_lora_r4_s1 runs/ms1_convnext_t_s1 --n 50
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR, RUNS_DIR

LABELS = {
    "vit_b16_head": "ViT linear probing",
    "vit_b16_lora_r4": "ViT LoRA r=4",
    "vit_b16_lora_r8": "ViT LoRA r=8",
    "vit_b16_lora_r16": "ViT LoRA r=16",
    "vit_b16_lora_r8_all": "ViT LoRA r=8 (all)",
    "vit_b16": "ViT full fine-tuning",
    "resnet50": "ResNet-50",
    "convnext_t": "ConvNeXt-T",
}


def label_of(run: Path) -> str:
    key = re.sub(r"^ms1_|_s\d+$", "", run.name)
    return LABELS.get(key, key)


def default_runs(runs_dir: Path, results: Path) -> list[Path]:
    """Seed-1 runs of the ViT and CNN teachers plus the other CNN (all with weights)."""
    groups = []
    teachers = results / "ms1" / "best_models.json"
    if teachers.exists():
        t = json.loads(teachers.read_text())
        groups += [t[f]["group"] for f in ("vit", "cnn") if f in t]
    groups += [g for g in ("ms1_resnet50", "ms1_convnext_t") if g not in groups]
    runs = [runs_dir / f"{g}_s1" for g in groups]
    return [r for r in runs if (r / "model" / "model.safetensors").exists()]


def split_dataset(cfg, size, mean, std, split: str):
    from animal_vit.data import build_datasets, build_transforms

    d = cfg.data
    datasets, class_names = build_datasets(
        d.name,
        build_transforms(size, mean, std, "none"),
        val_fraction=d.val_fraction,
        split_seed=d.split_seed,
        with_mask=True,
        size=size,
        species=d.species,
    )
    return datasets[split], class_names


def subset(dataset, n: int | None, seed: int = 0):
    """Random subset (fixed seed) of a split; the splits are sorted by class."""
    import numpy as np

    if n and n < len(dataset.indices):
        dataset.indices = sorted(np.random.default_rng(seed).permutation(dataset.indices)[:n].tolist())
    return dataset


def image_path(dataset, i: int) -> Path:
    """Original image file of item ``i`` (Oxford-IIIT Pet source)."""
    src = dataset.load
    return Path(src.base._images[src.positions[dataset.indices[i]]])


def maps_for(model, x, labels, methods, rng):
    """Saliency maps (B, H, W), each summing to 1, for every requested method."""
    import numpy as np
    import torch
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from animal_vit.explain import grad_cam_maps, normalise, occlusion_maps
    from animal_vit.saliency import attention_rollout, upsample

    out = {}
    size = x.shape[-1]
    for method in methods:
        if method == "gradcam":
            out[method] = normalise(grad_cam_maps(model, x, [ClassifierOutputTarget(int(c)) for c in labels]))
        elif method == "rollout":
            with torch.no_grad():
                out[method] = upsample(attention_rollout(model, x), size)
        elif method == "occlusion":
            out[method] = normalise(occlusion_maps(model, x, labels))
        elif method == "random":
            out[method] = normalise(rng.random((len(x), size, size)))
    return {k: np.asarray(v) for k, v in out.items()}


def faithfulness(model, name, dataset, methods, args, device, rows) -> None:
    """Deletion test on a random test subset."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn
    from animal_vit.explain import DELETION_STEPS, deletion_curve

    rng = np.random.default_rng(0)
    sums = {m: np.zeros(len(DELETION_STEPS)) for m in methods}
    count = 0
    for batch in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn):
        x, y = batch["pixel_values"].to(device), batch["labels"].to(device)
        maps = maps_for(model, x, y, methods, rng)
        for m in methods:
            sums[m] += len(x) * deletion_curve(model, x, y, maps[m])
        count += len(x)
        torch.cuda.empty_cache() if device == "cuda" else None
    for m in methods:
        for step, value in zip(DELETION_STEPS, sums[m] / count):
            rows.append({"model": name, "method": m, "removed": step, "probability": value, "n": count})
    print(f"  {name}: deletion test on {count} test images")


def head_focus(model, name, dataset, methods, args, device, rows, store, mean, std) -> None:
    """Saliency on head box / rest of the dog / background on the validation split."""
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn
    from animal_vit.explain import box_focus, box_mask, head_box, map_box
    from animal_vit.paths import PETS_DIR

    xml_dir = PETS_DIR / "oxford-iiit-pet" / "annotations" / "xmls"
    rng = np.random.default_rng(0)
    focus = {(m, r): [] for m in methods for r in ("head", "body", "background")}
    pos = 0
    std_a, mean_a = np.asarray(std)[:, None, None], np.asarray(mean)[:, None, None]
    for batch in DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn):
        x, y = batch["pixel_values"].to(device), batch["labels"].to(device)
        size = x.shape[-1]
        maps = maps_for(model, x, y, methods, rng)
        rgbs = (np.clip(batch["pixel_values"].numpy() * std_a + mean_a, 0, 1) * 255).astype(np.uint8)
        for j in range(len(x)):
            path = image_path(dataset, pos + j)
            box = head_box(xml_dir / f"{path.stem}.xml")
            if box is None:
                continue
            with Image.open(path) as im:
                head = box_mask(map_box(box, *im.size, size), size)
            dog = batch["mask"][j].numpy().astype(bool)
            regions = {"head": head, "body": dog & ~head, "background": ~dog}
            for m in methods:
                for r, mask in regions.items():
                    focus[(m, r)].append(box_focus(maps[m][j], mask))
            # exactly the image the model saw (resize + centre crop), for the example figure
            store.setdefault(name, {})[pos + j] = (maps["gradcam"][j], head, rgbs[j].transpose(1, 2, 0))
        pos += len(x)
    for (m, r), vals in focus.items():
        vals = np.asarray(vals, dtype=float)
        rows.append(
            {
                "model": name,
                "method": m,
                "region": r,
                "focus_mean": np.nanmean(vals),
                "focus_median": np.nanmedian(vals),
                "n": int(np.isfinite(vals).sum()),
            }
        )
    print(f"  {name}: head focus on {len(focus[(methods[0], 'head')])} validation images")


def contrastive(model, name, run, dataset, class_names, pairs, out: Path, device, mean, std) -> None:
    """Figure: why A and not B? For the most frequent confusions of station 1."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from animal_vit.explain import ContrastiveTarget, grad_cam_maps, occlusion_maps

    labels = np.array(dataset.labels)
    picked = []
    for a_name, b_name in pairs:
        a, b = class_names.index(a_name), class_names.index(b_name)
        idx = np.flatnonzero(labels == a)
        xs = torch.stack([dataset[int(i)]["pixel_values"] for i in idx]).to(device)
        with torch.no_grad():
            pred = torch.cat([model(pixel_values=c).logits.argmax(-1) for c in xs.split(32)]).cpu().numpy()
        wrong = idx[pred == b][:2].tolist()
        right = idx[pred == a][: 3 - len(wrong)].tolist()
        picked += [(int(i), a, b, i in wrong) for i in wrong + right]
    if not picked:
        return
    cols = ["image", "Grad-CAM: A", "Grad-CAM: B", "contrastive A − B", "occlusion: A"]
    fig, axes = plt.subplots(len(picked), len(cols), figsize=(2.1 * len(cols), 2.2 * len(picked)), squeeze=False)
    std_t, mean_t = torch.tensor(std).view(3, 1, 1), torch.tensor(mean).view(3, 1, 1)
    for row, (i, a, b, wrong) in enumerate(picked):
        x = dataset[i]["pixel_values"][None].to(device)
        maps = [
            grad_cam_maps(model, x, [ClassifierOutputTarget(a)])[0],
            grad_cam_maps(model, x, [ClassifierOutputTarget(b)])[0],
            grad_cam_maps(model, x, [ContrastiveTarget(a, b)])[0],
            np.clip(occlusion_maps(model, x, torch.tensor([a], device=device))[0], 0, None),
        ]
        rgb = (x[0].cpu() * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(rgb)
        verdict = f"predicted {class_names[b]}" if wrong else "predicted correctly"
        axes[row, 0].set_title(f"A = {class_names[a]}\nB = {class_names[b]}\n({verdict})", fontsize=6)
        for col, m in enumerate(maps, start=1):
            axes[row, col].imshow(rgb)
            axes[row, col].imshow(m, cmap="jet", alpha=0.45)
            if row == 0:
                axes[row, col].set_title(cols[col], fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Why A and not B? {name}", fontsize=10)
    fig.tight_layout()
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    fig.savefig(out / f"contrastive_{slug}.png", dpi=130)
    plt.close(fig)
    print(f"  {name}: contrastive figure with {len(picked)} images")


def head_examples(store, out: Path, k: int = 6) -> None:
    """Grid: rows = models, columns = validation images with head box and Grad-CAM."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = list(store)
    common = sorted(set.intersection(*(set(store[n]) for n in names)))[:: max(1, len(store[names[0]]) // k)][:k]
    if not common:
        return
    fig, axes = plt.subplots(len(names), len(common), figsize=(2.0 * len(common), 2.2 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        for c, i in enumerate(common):
            sal, head, img = store[name][i]
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(sal, cmap="jet", alpha=0.45)
            ys, xs = head.nonzero()
            if len(xs):
                ax.add_patch(
                    Rectangle(
                        (xs.min(), ys.min()),
                        xs.max() - xs.min(),
                        ys.max() - ys.min(),
                        fill=False,
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )
            ax.axis("off")
            if c == 0:
                ax.set_title(name, fontsize=7, loc="left")
    fig.suptitle("Grad-CAM and annotated head box (validation split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "head_examples.png", dpi=130)
    plt.close(fig)


def main() -> None:
    import numpy as np
    import pandas as pd

    from animal_vit.evaluation import select_device
    from animal_vit.explain import top_region_iou
    from animal_vit.models import model_family
    from animal_vit.training import load_run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, default=None, help="default: teachers + other CNN, seed 1")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "explain")
    parser.add_argument("--n", type=int, default=300, help="test images for the deletion test")
    parser.add_argument("--n-val", type=int, default=None, help="validation images for head focus (default all)")
    parser.add_argument("--pairs", type=int, default=3, help="most frequent confusions to explain")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    runs = args.runs or default_runs(args.runs_dir, args.out.parent)
    if not runs:
        raise SystemExit("no runs with weights found (train station 1 first)")

    conf = args.out.parent / "ms1" / "confusions.csv"
    pairs = pd.read_csv(conf).head(args.pairs)[["true", "predicted"]].values.tolist() if conf.exists() else []

    faith_rows, head_rows, store = [], [], {}
    for run in runs:
        model, cfg, class_names, (size, mean, std) = load_run(run, device)
        # Gradient-based maps need gradients through the backbone, also for a frozen linear-probing run.
        model.requires_grad_(True)
        name = label_of(run)
        methods = ["gradcam", "occlusion", "random"]
        if model_family(model) == "vit":
            model.config._attn_implementation = "eager"  # attention weights for rollout
            methods.insert(1, "rollout")
        print(f"== {name} ({run.name}) on {device}: {', '.join(methods)}")
        test, _ = split_dataset(cfg, size, mean, std, "test")
        if pairs:
            contrastive(model, name, run, test, class_names, pairs, args.out, device, mean, std)
        faithfulness(model, name, subset(test, args.n), methods, args, device, faith_rows)
        val, _ = split_dataset(cfg, size, mean, std, "validation")
        val = subset(val, args.n_val)
        head_focus(model, name, val, methods, args, device, head_rows, store, mean, std)
        del model

    pd.DataFrame(faith_rows).to_csv(args.out / "faithfulness.csv", index=False)
    pd.DataFrame(head_rows).to_csv(args.out / "head_focus.csv", index=False)
    names = list(store)
    agree = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = set(store[a]) & set(store[b])
            ious = [top_region_iou(store[a][k][0], store[b][k][0]) for k in common]
            agree.append({"model_a": a, "model_b": b, "iou_top20": float(np.nanmean(ious)), "n": len(ious)})
    pd.DataFrame(agree).to_csv(args.out / "agreement.csv", index=False)
    head_examples(store, args.out)
    (args.out / "runs.json").write_text(json.dumps({label_of(r): str(r) for r in runs}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
