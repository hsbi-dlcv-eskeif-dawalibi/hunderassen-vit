"""Pick the station-2 teachers automatically from the station-1 results.

The task of station 2 is "Modell aus Station 1 als Teacher". This script chooses the best
station-1 model per family by **validation** accuracy (mean over seeds) - never by test
accuracy, so the test split stays untouched for the final comparison:

- ``vit``: best ViT-B/16 adaptation (linear probing, LoRA variants, full fine-tuning)
- ``cnn``: best CNN (ResNet-50, ConvNeXt-T), used for the "which teacher?" experiment

Ties in validation accuracy (one image of the 250 is 0.4 pp, so ties happen) are broken by the
lower mean validation loss of the best checkpoint, the same rule as ``scripts/lr_sweep.py``.
Augmentation-ablation runs (``*_aug_*``) are not candidates. The choice is written to
``<out>/best_models.json`` (``results/ms1/``).

Example::

    python scripts/select_teacher.py                      # write results/ms1/best_models.json
    python scripts/select_teacher.py --print vit          # print the ViT teacher group (for make)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR, RUNS_DIR

FAMILIES = {"vit": r"^ms1_vit_b16(_.*)?$", "cnn": r"^ms1_(resnet50|convnext_t)$"}


def best_val_loss(run: Path) -> float:
    """Validation loss of the checkpoint with the best validation accuracy (inf if unknown)."""
    path = run / "history.json"
    if not path.exists():
        return float("inf")
    evals = [h for h in json.loads(path.read_text()) if "eval_accuracy" in h and "eval_loss" in h]
    return max(evals, key=lambda h: h["eval_accuracy"])["eval_loss"] if evals else float("inf")


def _mean(values) -> float:
    return sum(values) / len(values)


def candidates(runs_dir: Path) -> dict[str, dict]:
    """Station-1 groups with their seeds' validation accuracy and run directories."""
    groups: dict[str, dict] = {}
    for run in sorted(runs_dir.glob("ms1_*_s*")):
        if "_aug_" in run.name or not (run / "metrics.json").exists():
            continue
        if not (run / "model" / "model.safetensors").exists():
            continue
        group = re.sub(r"_s\d+$", "", run.name)
        seed = int(re.search(r"_s(\d+)$", run.name).group(1))
        metrics = json.loads((run / "metrics.json").read_text())
        g = groups.setdefault(group, {"val": [], "loss": [], "test": [], "runs": {}})
        g["val"].append(metrics.get("best_validation_metric") or 0.0)
        g["loss"].append(best_val_loss(run))
        g["test"].append(metrics["test_metrics"]["accuracy"])
        g["runs"][seed] = str(run)
    return groups


def select(runs_dir: Path) -> dict[str, dict]:
    groups = candidates(runs_dir)
    choice = {}
    for family, pattern in FAMILIES.items():
        fam = {k: v for k, v in groups.items() if re.match(pattern, k)}
        if not fam:
            continue
        # Highest validation accuracy first; ties (rounded to 1e-6) -> lower validation loss.
        ranking = sorted(fam.items(), key=lambda kv: (-round(_mean(kv[1]["val"]), 6), _mean(kv[1]["loss"])))
        best, info = ranking[0]
        choice[family] = {
            "group": best,
            "val_accuracy_mean": sum(info["val"]) / len(info["val"]),
            "test_accuracy_mean": sum(info["test"]) / len(info["test"]),
            "runs": {str(k): v for k, v in sorted(info["runs"].items())},
            "ranking": [
                {
                    "group": g,
                    "val_accuracy_mean": _mean(i["val"]),
                    "val_loss_mean": _mean(i["loss"]),
                    "seeds": len(i["val"]),
                }
                for g, i in ranking
            ],
        }
    return choice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", nargs="?", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ms1")
    parser.add_argument("--print", dest="print_family", choices=list(FAMILIES), help="only print the group name")
    args = parser.parse_args()

    choice = select(args.runs_dir)
    if args.print_family:
        if args.print_family not in choice:
            raise SystemExit(f"no finished station-1 {args.print_family} run with saved weights in {args.runs_dir}")
        print(choice[args.print_family]["group"])
        return
    if not choice:
        raise SystemExit(f"no finished station-1 runs in {args.runs_dir} - run station 1 first")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "best_models.json").write_text(json.dumps(choice, indent=2))
    for family, c in choice.items():
        print(f"{family:3s} teacher: {c['group']}  (validation acc {100 * c['val_accuracy_mean']:.2f} %)")


if __name__ == "__main__":
    main()
