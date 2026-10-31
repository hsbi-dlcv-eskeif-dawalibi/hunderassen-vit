"""Aggregate runs over seeds into the result tables used in papers and slides.

Runs are grouped by name with the seed suffix ``_s<k>`` removed
(``ms1_vit_b16_s1``, ``ms1_vit_b16_s2`` -> ``ms1_vit_b16``). For each group the mean and
standard deviation of every available metric are reported: test accuracy / macro-F1 /
ECE (``metrics.json``), variant accuracies (``variants.csv``), shape bias
(``shape_bias.json``), foreground focus (``saliency.json``) and cost (``efficiency.json``).

Writes ``results/summary.csv`` and ``results/summary.md``.

Example::

    python scripts/summarize.py runs/*
    python scripts/summarize.py runs/ms1_* --out results/ms1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR


def collect(run: Path) -> dict:
    """Read all result files of one run directory into a flat dict of metrics."""
    row: dict = {}
    if (run / "metrics.json").exists():
        payload = json.loads((run / "metrics.json").read_text())
        m = payload["test_metrics"]
        row.update(accuracy=m["accuracy"], f1_macro=m["f1_macro"], ece=m.get("ece"))
        row["val_accuracy"] = payload.get("best_validation_metric")
        # Cost of the fine-tuning strategy (MS1): trainable parameters and training time.
        if payload.get("trainable_parameters") is not None:
            row["trainable_m"] = payload["trainable_parameters"] / 1e6
        if payload.get("train_runtime_s") is not None:
            row["train_min"] = payload["train_runtime_s"] / 60
    if (run / "variants.csv").exists():
        import pandas as pd

        for _, r in pd.read_csv(run / "variants.csv").iterrows():
            row[f"acc_{r['variant']}"] = r["accuracy"]
    if (run / "shape_bias.json").exists():
        row["shape_bias"] = json.loads((run / "shape_bias.json").read_text())["shape_bias"]
    if (run / "saliency.json").exists():
        for method, values in json.loads((run / "saliency.json").read_text()).items():
            row[f"focus_{method}"] = values["foreground_focus_mean"]
    if (run / "efficiency.json").exists():
        e = json.loads((run / "efficiency.json").read_text())
        row.update(params_m=e["parameters_m"], gflops=e["gflops"], latency_ms=e["latency_ms_bs1"])
    return row


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        if (run / "config.json").exists():
            rows.append({"group": re.sub(r"_s\d+$", "", run.name), "run": run.name, **collect(run)})
    if not rows:
        raise SystemExit("no runs with config.json found")
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("group", "run")]
    grouped = df.groupby("group")[metrics]
    summary = (
        grouped.mean()
        .add_suffix("_mean")
        .join(grouped.std().add_suffix("_std"))
        .join(df.groupby("group").size().rename("n_seeds"))
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "summary.csv")

    def fmt(group: str, metric: str) -> str:
        mean, std = summary.loc[group, f"{metric}_mean"], summary.loc[group, f"{metric}_std"]
        if pd.isna(mean):
            return "-"
        scale = 100 if metric.startswith(("acc", "f1", "val_acc")) else 1
        return f"{mean * scale:.2f} ± {0 if pd.isna(std) else std * scale:.2f}"

    lines = ["| Gruppe | Seeds | " + " | ".join(metrics) + " |", "|---|---" + "|---" * len(metrics) + "|"]
    for group in summary.index:
        cells = [fmt(group, m) for m in metrics]
        lines.append(f"| {group} | {summary.loc[group, 'n_seeds']} | " + " | ".join(cells) + " |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
