"""Summarise a learning-rate sweep by *validation* accuracy (model selection never looks at test).

Used by ``make ms1-cnn-lr``: the CNNs of station 1 were trained with the ViT learning rate (5e-5);
this sweep checks on validation whether a higher rate fits the CNNs better.

    python scripts/lr_sweep.py runs/lr_resnet50_* [--baseline runs/ms1_resnet50_s*]

Prints mean ± std of the validation accuracy, validation loss and ECE per learning rate and the
best rate. Run directories must be called ``<prefix>_lr<rate>_s<seed>``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import _bootstrap  # noqa: F401


def best_eval(run: Path) -> dict:
    """Validation metrics of the best checkpoint (by validation accuracy) from ``history.json``."""
    history = json.loads((run / "history.json").read_text())
    evals = [h for h in history if "eval_accuracy" in h]
    return max(evals, key=lambda h: h["eval_accuracy"]) if evals else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--baseline", nargs="*", type=Path, default=[], help="runs with the original rate")
    parser.add_argument("--baseline-lr", default="5e-05")
    args = parser.parse_args()

    groups: dict[str, list[dict]] = {}
    for run in args.runs:
        m = re.search(r"_lr([0-9.e+-]+)_s\d+$", run.name)
        if m and (run / "metrics.json").exists():
            groups.setdefault(m.group(1), []).append(best_eval(run))
    for run in args.baseline:
        if (run / "metrics.json").exists():
            groups.setdefault(args.baseline_lr, []).append(best_eval(run))
    if not groups:
        raise SystemExit("no finished runs found")

    def stat(rows, key):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            return float("nan"), float("nan")
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print("| learning rate | seeds | val accuracy (%) | val loss | val ECE |")
    print("|---|---|---|---|---|")
    ranking = []
    for lr in sorted(groups, key=float):
        rows = groups[lr]
        acc, acc_sd = stat(rows, "eval_accuracy")
        loss, loss_sd = stat(rows, "eval_loss")
        ece, _ = stat(rows, "eval_ece")
        ranking.append((acc, -loss, lr))
        print(f"| {lr} | {len(rows)} | {100 * acc:.2f} ± {100 * acc_sd:.2f} | {loss:.3f} ± {loss_sd:.3f} | {ece:.3f} |")
    best = max(ranking)[2]
    print(f"\nbest learning rate on validation: {best} (ties broken by lower validation loss)")


if __name__ == "__main__":
    main()
