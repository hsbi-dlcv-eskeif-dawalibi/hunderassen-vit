"""Shared helpers for the reports (``scripts/report_ms1.py``, ``report_why.py``, ``report_explain.py``).

Every run directory contains ``config.json`` and ``metrics.json`` (trained runs and CLIP runs
alike), optionally ``efficiency.json``, ``variants.csv``, ``classification_report.csv``,
``predictions.csv`` and ``history.json``. These helpers turn a set of runs into tidy tables,
paired comparisons and consistently styled figures.

Statistics: runs that differ only in the seed form a group (name without ``_s<k>``).
Comparisons between two groups are *paired by seed* (same seed = same data order and, in
station 2, the same teacher run), so the seed-to-seed variation largely cancels.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path

SEED_RE = re.compile(r"_s(\d+)$")

#: Marker / line style cycle: readable in grayscale print and for colour-blind readers.
STYLES = [(m, ls) for ls in ("-", "--", ":") for m in ("o", "s", "^", "D", "v", "P", "X")]


def group_of(name: str) -> str:
    """Run name without the seed suffix."""
    return SEED_RE.sub("", name)


def seed_of(name: str) -> int | None:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def base_row(run: Path) -> dict | None:
    """Metrics shared by all run types (``None`` if the run is not finished)."""
    if not (run / "metrics.json").exists() or not (run / "config.json").exists():
        return None
    m, cfg = load_json(run / "metrics.json"), load_json(run / "config.json")
    t = m["test_metrics"]
    row = {
        "run": run.name,
        "group": group_of(run.name),
        "seed": seed_of(run.name),
        "accuracy": t["accuracy"],
        "f1_macro": t["f1_macro"],
        "ece": t.get("ece"),
        "val_accuracy": m.get("best_validation_metric"),
        "trainable_params": m.get("trainable_parameters"),
        "total_params": m.get("total_parameters") or m.get("image_encoder_parameters"),
        "train_min": (m.get("train_runtime_s") or 0) / 60 or None,
        "n_train": m.get("n_train"),
        "checkpoint": cfg.get("model", {}).get("checkpoint"),
    }
    if (run / "efficiency.json").exists():
        e = load_json(run / "efficiency.json")
        row.update(gflops=e.get("gflops"), latency_ms=e.get("latency_ms_bs1"), params_m=e.get("parameters_m"))
    return row


def aggregate(df, by: str = "group"):
    """Mean, std and count of every numeric column per group."""
    numeric = [c for c in df.select_dtypes("number").columns if c != "seed"]
    grouped = df.groupby(by)[numeric]
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    out["n_seeds"] = df.groupby(by).size()
    return out


def decimal_comma(text: str) -> str:
    """German number format for the LaTeX tables of the (German) papers: ``96.21`` -> ``96{,}21``.

    ``{,}`` avoids the extra space TeX puts after a comma in math mode and works in text mode, too.
    """
    return re.sub(r"(?<=\d)\.(?=\d)", "{,}", text)


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def pm(mean, std, scale: float = 100, digits: int = 2) -> str:
    """``mean ± std`` (std omitted for a single run, ``-`` if missing)."""
    if is_missing(mean):
        return "-"
    if is_missing(std):
        return f"{mean * scale:.{digits}f}"
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def cell(summary, group: str, metric: str, scale: float = 100, digits: int = 2) -> str:
    """``mean ± std`` of ``metric`` for ``group`` from an :func:`aggregate` table."""
    if group not in summary.index:
        return "-"
    return pm(summary.loc[group].get(f"{metric}_mean"), summary.loc[group].get(f"{metric}_std"), scale, digits)


def paired_delta(df, group_a: str, group_b: str, metric: str = "accuracy") -> dict | None:
    """``b - a`` per seed for the seeds both groups share: mean, std, standard error, verdict.

    ``significant`` is a deliberately simple rule for n = 3 seeds: the mean difference is
    larger than two standard errors of the paired differences (and larger than zero).
    """
    # Mean per seed: robust if a group accidentally contains two runs with the same seed.
    a = df[df.group == group_a].groupby("seed")[metric].mean()
    b = df[df.group == group_b].groupby("seed")[metric].mean()
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds:
        return None
    deltas = [b[s] - a[s] for s in seeds]
    n = len(deltas)
    mean = sum(deltas) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else float("nan")
    se = std / math.sqrt(n) if n > 1 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "n": n,
        "deltas": deltas,
        "significant": (n > 1 and not math.isnan(se) and abs(mean) > 2 * se),
    }


def delta_text(d: dict | None, scale: float = 100) -> str:
    """Signed difference in percentage points with a verdict."""
    if not d:
        return "-"
    sign = "+" if d["mean"] >= 0 else "−"
    spread = "" if is_missing(d["std"]) else f" ± {d['std'] * scale:.2f}"
    verdict = "" if d["n"] < 2 else (" (> 2·SE)" if d["significant"] else " (im Rauschen)")
    return f"{sign}{abs(d['mean']) * scale:.2f}{spread} pp, n={d['n']}{verdict}"


def per_breed_f1(runs: Sequence[Path]):
    """Mean F1 per class over runs (``classification_report.csv``)."""
    import pandas as pd

    frames = []
    for run in runs:
        path = Path(run) / "classification_report.csv"
        if path.exists():
            rep = pd.read_csv(path, index_col=0)
            rep = rep.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in rep.index])
            frames.append(rep["f1-score"])
    return pd.concat(frames, axis=1).mean(axis=1) if frames else None


def md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> list[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def figure(width: float = 6.5, height: float = 4.0):
    """New figure with the project style (Agg backend, grid)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(alpha=0.3)
    return fig, ax


def save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def validation_curve_rows(runs: Sequence[Path], label_fn) -> list[dict]:
    """Validation accuracy per epoch of all runs, labelled by ``label_fn(run)``."""
    from animal_vit.reporting import epoch_table

    rows = []
    for run in runs:
        if (Path(run) / "history.json").exists():
            for r in epoch_table(load_json(Path(run) / "history.json")):
                rows.append({"label": label_fn(Path(run)), "epoch": r["epoch"], "acc": r["val_accuracy"]})
    return rows
