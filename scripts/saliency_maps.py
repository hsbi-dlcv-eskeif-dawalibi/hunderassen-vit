"""Saliency maps and foreground focus for trained runs.

- Quantitative: mean *foreground focus* over the test set (saliency mass on the pet
  relative to its area; >1 = looks at the animal) -> ``<run>/saliency.json``.
- Qualitative: a grid of example images with their maps -> ``<run>/saliency_<method>.png``.

Methods: ``gradcam`` (all models) and ``rollout`` (ViT/DeiT only).

Example::

    python scripts/saliency_maps.py runs/ms1_vit_b16_s1 --method rollout gradcam
    python scripts/saliency_maps.py runs/ms1_resnet50_s1 --method gradcam --max-samples 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn
    from animal_vit.evaluation import select_device, test_dataset
    from animal_vit.models import model_family
    from animal_vit.saliency import attention_rollout, foreground_focus, grad_cam, upsample
    from animal_vit.training import load_run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--method", nargs="+", default=["gradcam"], choices=["gradcam", "rollout"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    for run in args.runs:
        model, cfg, class_names, (size, mean, std) = load_run(run, device)
        if "rollout" in args.method and model_family(model) == "vit":
            model.config._attn_implementation = "eager"
        dataset, _ = test_dataset(cfg, size, mean, std, with_mask=True)
        if args.max_samples:
            # Random subset (fixed seed): the test split is sorted by class.
            dataset.indices = sorted(np.random.default_rng(0).permutation(dataset.indices)[: args.max_samples].tolist())
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        summary = {}
        for method in args.method:
            if method == "rollout" and model_family(model) != "vit":
                print(f"{run.name}: rollout skipped (not a ViT)")
                continue
            focus, examples = [], []
            for batch in loader:
                x = batch["pixel_values"].to(device)
                if method == "rollout":
                    with torch.no_grad():
                        maps = attention_rollout(model, x)
                else:
                    maps = grad_cam(model, x)
                maps = upsample(maps, size)
                masks = batch["mask"].numpy()
                focus += [foreground_focus(m, k) for m, k in zip(maps, masks)]
                if len(examples) < args.examples:
                    examples += list(zip(batch["pixel_values"], maps, masks))[: args.examples - len(examples)]
            summary[method] = {
                "foreground_focus_mean": float(np.nanmean(focus)),
                "foreground_focus_median": float(np.nanmedian(focus)),
                "n": len(focus),
            }
            print(f"{run.name:30s} {method:8s} foreground focus {np.nanmean(focus):.3f}")

            fig, axes = plt.subplots(2, len(examples), figsize=(2.2 * len(examples), 4.6))
            std_t, mean_t = torch.tensor(std).view(3, 1, 1), torch.tensor(mean).view(3, 1, 1)
            for col, (img, sal, mask) in enumerate(examples):
                rgb = (img * std_t + mean_t).clamp(0, 1).permute(1, 2, 0).numpy()
                axes[0, col].imshow(rgb)
                axes[0, col].contour(mask, levels=[0.5], colors="white", linewidths=0.8)
                axes[1, col].imshow(rgb)
                axes[1, col].imshow(sal, cmap="jet", alpha=0.5)
                for ax in axes[:, col]:
                    ax.axis("off")
            fig.suptitle(f"{run.name} - {method}")
            fig.tight_layout()
            fig.savefig(run / f"saliency_{method}.png", dpi=150)
            plt.close(fig)
        (run / "saliency.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
