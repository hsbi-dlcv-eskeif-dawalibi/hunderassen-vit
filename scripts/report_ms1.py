"""Station 1 report: everything needed to write the paper, from all ``runs/ms1_*`` directories.

Runs are grouped by name without the seed suffix (``ms1_vit_b16_lora_r8_s2`` -> ``ms1_vit_b16_lora_r8``)
and reported as mean +- standard deviation over the seeds. Output in ``results/ms1/``:

``REPORT.md``                  all tables, rankings and findings in one file (start here)
``methods.csv``                one row per method (means, stds, cost)
``runs.csv``                   one row per run (all seeds)
``tables.tex``                 LaTeX table bodies to paste into ``paper/ms1``
``efficiency.png``             test accuracy vs. trainable parameters (log scale) - "how efficient?"
``validation_curves.png``      validation accuracy per epoch, mean over seeds
``per_breed_f1.png``           macro-F1 per breed of the best method
``confusions.csv``             most frequent confusions of the best method

Example::

    python scripts/report_ms1.py                 # runs/ms1_*_s*
    python scripts/report_ms1.py runs/ms1_*_s*   # explicit
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR, RUNS_DIR
from animal_vit.report_tools import decimal_comma

#: Display order and names of the station-1 methods (group name without ``ms1_`` prefix).
METHOD_ORDER = [
    ("vit_b16_head", "ViT linear probing"),
    ("vit_b16_lora_r4", "ViT LoRA r=4"),
    ("vit_b16_lora_r8", "ViT LoRA r=8"),
    ("vit_b16_lora_r16", "ViT LoRA r=16"),
    ("vit_b16_lora_r8_all", "ViT LoRA r=8 (all layers)"),
    ("vit_b16", "ViT full fine-tuning"),
    ("resnet50", "ResNet-50 (CNN)"),
    ("convnext_t", "ConvNeXt-T (CNN)"),
]
LABELS = dict(METHOD_ORDER)


def group_of(run: Path) -> str:
    """Group name: run name without the seed suffix ``_s<k>``."""
    return re.sub(r"_s\d+$", "", run.name)


def label_of(group: str) -> str:
    """Readable method name for tables and plots."""
    key = group.removeprefix("ms1_")
    aug = re.search(r"_aug_(none|light|strong)$", key)
    if aug:
        key = key[: aug.start()]
        return f"{LABELS.get(key, key)} [aug {aug.group(1)}]"
    return LABELS.get(key, key)


def read_run(run: Path) -> dict | None:
    """One row per finished run (``metrics.json`` present)."""
    if not (run / "metrics.json").exists():
        return None
    m = json.loads((run / "metrics.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    t = m["test_metrics"]
    row = {
        "group": group_of(run),
        "run": run.name,
        "seed": cfg["training"]["seed"],
        "augmentation": cfg["data"]["augmentation"],
        "trainable": cfg["modifications"]["trainable"],
        "accuracy": t["accuracy"],
        "f1_macro": t["f1_macro"],
        "ece": t.get("ece"),
        "val_accuracy": m.get("best_validation_metric"),
        "trainable_params": m.get("trainable_parameters"),
        "total_params": m.get("total_parameters"),
        "train_min": (m.get("train_runtime_s") or 0) / 60 or None,
        "best_epoch": m.get("best_epoch"),
        "epochs_trained": m.get("epochs_trained"),
    }
    if cfg["modifications"]["trainable"] == "lora":
        lora = cfg["modifications"]["lora"]
        row.update(lora_rank=lora["rank"], lora_alpha=lora["alpha"], lora_targets=str(lora["targets"]))
    if (run / "efficiency.json").exists():
        e = json.loads((run / "efficiency.json").read_text())
        row.update(gflops=e.get("gflops"), latency_ms=e.get("latency_ms_bs1"))
    return row


def aggregate(df):
    """Mean and std per group, plus the number of seeds."""
    numeric = df.select_dtypes("number").columns.drop("seed", errors="ignore")
    grouped = df.groupby("group")[list(numeric)]
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    out["n_seeds"] = df.groupby("group").size()
    out["label"] = [label_of(g) for g in out.index]
    order = {f"ms1_{k}": i for i, (k, _) in enumerate(METHOD_ORDER)}
    out["order"] = [order.get(g, 100) for g in out.index]
    return out.sort_values(["order", "label"])


def pm(row, metric: str, scale: float = 100, digits: int = 2) -> str:
    """``mean ± std`` as text (``-`` if missing)."""
    import math

    mean, std = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "-"
    std = 0 if std is None or (isinstance(std, float) and math.isnan(std)) else std
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def params_text(row) -> str:
    """Trainable parameters in millions and as share of the model."""
    import math

    tr, tot = row.get("trainable_params_mean"), row.get("total_params_mean")
    if tr is None or (isinstance(tr, float) and math.isnan(tr)):
        return "-"
    return f"{tr / 1e6:.3f} M ({100 * tr / tot:.2f} %)"


def plot_efficiency(summary, out: Path) -> None:
    """Accuracy vs. trainable parameters: the core figure for 'how efficiently can a ViT be adapted?'."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # Alternating label offsets: LoRA variants have similar parameter counts and accuracies.
    offsets = [(5, 5), (5, -11), (-5, 9), (5, 14)]
    for i, (_, r) in enumerate(summary.iterrows()):
        if r.get("trainable_params_mean") is None or r["trainable_params_mean"] != r["trainable_params_mean"]:
            continue
        marker = "s" if "CNN" in r["label"] else "o"
        ax.errorbar(
            r["trainable_params_mean"],
            100 * r["accuracy_mean"],
            yerr=100 * (r["accuracy_std"] if r["accuracy_std"] == r["accuracy_std"] else 0),
            fmt=marker,
            capsize=3,
        )
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(r["label"], (r["trainable_params_mean"], 100 * r["accuracy_mean"]), fontsize=7,
                    xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))  # fmt: skip
    ax.set_xscale("log")
    ax.set(
        xlabel="trainable parameters (log scale)", ylabel="test accuracy (%)", title="Accuracy vs. trainable parameters"
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_validation_curves(runs: list[Path], out: Path) -> None:
    """Validation accuracy per epoch, mean over the seeds of each method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from animal_vit.reporting import epoch_table

    rows = []
    for run in runs:
        if (run / "history.json").exists():
            for r in epoch_table(json.loads((run / "history.json").read_text())):
                rows.append({"label": label_of(group_of(run)), "epoch": r["epoch"], "acc": r["val_accuracy"]})
    if not rows:
        return
    df = pd.DataFrame(rows).groupby(["label", "epoch"])["acc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    # Marker and line style as well as colour: readable in grayscale print and for colour-blind
    # readers (author kit, "Color").
    styles = [(m, ls) for ls in ("-", "--", ":") for m in ("o", "s", "^", "D", "v", "x")]
    for (label, part), (marker, linestyle) in zip(df.groupby("label"), styles):
        ax.plot(part["epoch"], 100 * part["acc"], marker=marker, linestyle=linestyle, markersize=4, label=label)
    ax.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy (mean over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def per_breed(runs: list[Path]):
    """Mean per-breed F1 over the given runs (``classification_report.csv``)."""
    import pandas as pd

    frames = []
    for run in runs:
        path = run / "classification_report.csv"
        if path.exists():
            rep = pd.read_csv(path, index_col=0)
            rep = rep.drop(index=[i for i in ("accuracy", "macro avg", "weighted avg") if i in rep.index])
            frames.append(rep["f1-score"])
    return pd.concat(frames, axis=1).mean(axis=1).sort_values() if frames else None


def confusions(runs: list[Path], top: int = 10):
    """Most frequent (true -> predicted) errors over the given runs (``predictions.csv``)."""
    import pandas as pd

    frames = [pd.read_csv(run / "predictions.csv") for run in runs if (run / "predictions.csv").exists()]
    if not frames:
        return None
    df = pd.concat(frames)
    wrong = df[~df["correct"]]
    return (
        wrong.groupby(["true", "predicted"]).size().sort_values(ascending=False).head(top).rename("count").reset_index()
    )


def latex_rows(summary) -> str:
    """Rows for the method table of the paper: method & acc & F1 & trainable & time."""
    lines = ["% Generated by scripts/report_ms1.py - paste into paper/ms1/sec/4_results.tex"]
    for _, r in summary.iterrows():
        tr = r.get("trainable_params_mean")
        train = "-" if tr != tr or tr is None else f"{tr / 1e6:.2f}\\,M"
        time = r.get("train_min_mean")
        time_text = "-" if time != time or time is None else f"{time:.1f}"
        acc = pm(r, "accuracy").replace("±", "$\\pm$")
        f1 = pm(r, "f1_macro").replace("±", "$\\pm$")
        lines.append(f"{r['label']} & " + decimal_comma(f"{acc} & {f1} & {train} & {time_text}") + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ms1")
    args = parser.parse_args()
    runs = sorted(args.runs or RUNS_DIR.glob("ms1_*_s*"))
    rows = [r for r in (read_run(run) for run in runs) if r]
    if not rows:
        raise SystemExit("no finished ms1 runs found (run `make ms1` first)")
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "runs.csv", index=False)
    summary = aggregate(df)
    summary.to_csv(args.out / "methods.csv")
    main_methods = summary[~summary.index.str.contains("_aug_")]
    aug_methods = summary[summary.index.str.contains("_aug_")]

    plot_efficiency(main_methods, args.out / "efficiency.png")
    plot_validation_curves([r for r in runs if "_aug_" not in r.name], args.out / "validation_curves.png")
    (args.out / "tables.tex").write_text(latex_rows(main_methods))

    best_group = main_methods["val_accuracy_mean"].idxmax()  # chosen on validation, not on test
    best_runs = [r for r in runs if group_of(r) == best_group]
    f1 = per_breed(best_runs)
    conf = confusions(best_runs)
    if conf is not None:
        conf.to_csv(args.out / "confusions.csv", index=False)
    if f1 is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 0.25 * len(f1) + 1))
        ax.barh(f1.index, 100 * f1.values)
        ax.set(xlabel="F1 (%)", title=f"F1 per breed - {label_of(best_group)}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "per_breed_f1.png", dpi=200)
        plt.close(fig)

    # ------------------------------------------------------------------ REPORT.md
    lines = [
        "# Station 1 - Ergebnisbericht",
        "",
        "Automatisch erzeugt von `scripts/report_ms1.py`. Alle Werte: Testsplit, Mittelwert ± Standardabweichung",
        "über die Seeds. Auswahl des besten Checkpoints und der besten Methode nur auf Validation.",
        "",
        "## 1. Anpassungsmethoden (Kern von Station 1)",
        "",
        "| Methode | Seeds | Test-Acc (%) | Macro-F1 (%) | Val-Acc (%) | Trainierbare Parameter "
        "| Trainingszeit (min) | Beste Epoche | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in main_methods.iterrows():
        lines.append(
            f"| {r['label']} | {r['n_seeds']} | {pm(r, 'accuracy')} | {pm(r, 'f1_macro')} | {pm(r, 'val_accuracy')} | "
            f"{params_text(r)} | {pm(r, 'train_min', 1, 1)} | {pm(r, 'best_epoch', 1, 1)} | {pm(r, 'ece', 1, 3)} |"
        )
    ranked = main_methods.sort_values("accuracy_mean", ascending=False)
    best, worst = ranked.iloc[0], ranked.iloc[-1]
    lines += [
        "",
        "**Rangfolge nach Test-Accuracy:** " + " > ".join(ranked["label"]),
        "",
        f"- Beste Methode: **{best['label']}** ({pm(best, 'accuracy')} %), schlechteste: **{worst['label']}** "
        f"({pm(worst, 'accuracy')} %).",
        f"- Auf Validation beste Methode (für Station 2 als Teacher): **{label_of(best_group)}**.",
    ]
    vit = main_methods[main_methods.index.str.startswith("ms1_vit")]
    if len(vit) > 1:
        lines.append(
            "- Abstände zur besten ViT-Methode im Vergleich zur Seed-Streuung (nur größere Abstände interpretieren):"
        )
        top = vit.sort_values("accuracy_mean").iloc[-1]
        for _, r in vit.iterrows():
            if r.name == top.name:
                continue
            gap = 100 * (top["accuracy_mean"] - r["accuracy_mean"])
            spread = 100 * max(
                *(x if x == x else 0 for x in (top["accuracy_std"], r["accuracy_std"]))  # NaN with 1 seed
            )
            if min(top["n_seeds"], r["n_seeds"]) < 2:
                verdict = "nur 1 Seed, Streuung unbekannt (mit SEEDS='1 2 3' wiederholen)"
            else:
                verdict = "größer als die Streuung" if gap > 2 * spread else "innerhalb der Streuung"
            lines.append(f"  - {r['label']}: −{gap:.2f} Pp. (Streuung ±{spread:.2f}) → {verdict}")
    lora = main_methods[main_methods.index.str.contains("lora")]
    if len(lora):
        lines += [
            "",
            "## 2. LoRA mit unterschiedlichen Parametern",
            "",
            "| Variante | Rang | alpha | Ziel-Layer | Test-Acc (%) | Trainierbar |",
            "|---|---|---|---|---|---|",
        ]
        for g, r in lora.iterrows():
            first = df[df["group"] == g].iloc[0]
            lines.append(
                f"| {r['label']} | {int(first['lora_rank'])} | {first['lora_alpha']:g} | {first['lora_targets']} | "
                f"{pm(r, 'accuracy')} | {params_text(r)} |"
            )
    if len(aug_methods):
        lines += [
            "",
            "## 3. Augmentierung",
            "",
            "| Lauf | Test-Acc (%) | Macro-F1 (%) | Val-Acc (%) |",
            "|---|---|---|---|",
        ]
        # The main run of the same method uses the default preset "light".
        bases = {re.sub(r"_aug_(none|light|strong)$", "", g) for g in aug_methods.index}
        rows_aug = [
            (f"{main_methods.loc[g, 'label']} [aug light]", main_methods.loc[g])
            for g in bases
            if g in main_methods.index
        ]
        rows_aug += [(r["label"], r) for _, r in aug_methods.iterrows()]
        for label, r in rows_aug:
            lines.append(f"| {label} | {pm(r, 'accuracy')} | {pm(r, 'f1_macro')} | {pm(r, 'val_accuracy')} |")
    if "gflops_mean" in main_methods.columns:
        lines += ["", "## 4. Kosten zur Inferenz", "", "| Methode | GFLOPs | Latenz (ms, Batch 1) |", "|---|---|---|"]
        for _, r in main_methods.iterrows():
            lines.append(f"| {r['label']} | {pm(r, 'gflops', 1, 2)} | {pm(r, 'latency_ms', 1, 1)} |")
        lines.append("\nLoRA wird nach dem Training in die Gewichte gemergt: gleiche Inferenzkosten wie der Basis-ViT.")
    lines += ["", f"## 5. Fehleranalyse: {label_of(best_group)}", ""]
    if f1 is not None:
        lines.append("Schwierigste Rassen (F1): " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in f1.head(5).items()))
        lines.append("")
        lines.append(
            "Leichteste Rassen (F1): " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in f1.tail(5)[::-1].items())
        )
    if conf is not None and len(conf):
        lines += [
            "",
            "Häufigste Verwechslungen (über alle Seeds):",
            "",
            "| Wahr | Vorhergesagt | Anzahl |",
            "|---|---|---|",
        ]
        lines += [f"| {r['true']} | {r['predicted']} | {r['count']} |" for _, r in conf.iterrows()]
    lines += interpretation(main_methods, aug_methods, conf)
    lines += [
        "",
        "## 6. Dateien",
        "",
        "- `efficiency.png`: Accuracy vs. trainierbare Parameter (Kernabbildung)",
        "- `validation_curves.png`: Lernkurven aller Methoden",
        "- `per_breed_f1.png`, `confusions.csv`: Fehleranalyse",
        "- `tables.tex`: Tabellenzeilen für `paper/ms1/sec/4_results.tex`",
        "- `methods.csv`, `runs.csv`: alle Zahlen",
        "- Datenqualität: `results/data_quality/README.md`; pro Lauf: `runs/<name>/README.md`",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))
    print(f"wrote {args.out}")


def interpretation(main_methods, aug_methods, conf) -> list[str]:
    """Data-driven findings with candidate explanations and sources, as material for the discussion."""

    def acc(g):
        return main_methods.loc[g, "accuracy_mean"] if g in main_methods.index else None

    def gap_text(a, b) -> str:
        """``a - b`` in pp, with a verdict against the seed spread (needs >= 2 seeds)."""
        ra, rb = main_methods.loc[a], main_methods.loc[b]
        gap = 100 * (ra["accuracy_mean"] - rb["accuracy_mean"])
        if min(ra["n_seeds"], rb["n_seeds"]) < 2:
            return f"{gap:+.2f} pp (1 Seed, Streuung unbekannt)"
        spread = 100 * max(x if x == x else 0 for x in (ra["accuracy_std"], rb["accuracy_std"]))
        verdict = "größer als die" if abs(gap) > 2 * spread else "innerhalb der"
        return f"{gap:+.2f} pp ({verdict} Seed-Streuung)"

    out = [
        "",
        "## 5b. Befunde und Erklärungsansätze (Material für die Diskussion)",
        "",
        "Aus den Ergebnissen abgeleitet: Befund, möglicher Grund mit Quelle, was im Paper zu begründen ist. "
        "Formulierung und Abwägung sind eure Aufgabe.",
        "",
    ]
    lora = [g for g in main_methods.index if "lora" in g]
    best_lora = max(lora, key=acc) if lora else None
    head, full = "ms1_vit_b16_head", "ms1_vit_b16"
    if best_lora and head in main_methods.index:
        out.append(
            f"- **LoRA vs. Linear Probing:** {main_methods.loc[best_lora, 'label']} gegenüber Linear Probing "
            f"{gap_text(best_lora, head)}. Erklärungsansatz: Linear Probing kann nur die eingefrorenen "
            "ImageNet-Merkmale neu gewichten; LoRA passt zusätzlich die Attention an und kann feine "
            "Rassenunterschiede herausarbeiten (Hu et al. 2022). Ist der Abstand klein, sind die ImageNet-Merkmale "
            "für Hunde schon gut (ImageNet enthält ca. 120 Hunderassen; vgl. Kornblith et al. 2019)."
        )
    if best_lora and full in main_methods.index:
        share = (
            100 * main_methods.loc[best_lora, "trainable_params_mean"] / main_methods.loc[full, "trainable_params_mean"]
        )
        out.append(
            f"- **LoRA vs. volles Fine-Tuning:** {gap_text(best_lora, full)} bei {share:.2f} % der trainierbaren "
            "Parameter. Erklärungsansatz: Die nötige Anpassung hat einen niedrigen Rang (Hu et al. 2022); volles "
            "Fine-Tuning kann auf ca. 2 250 Bildern eher überanpassen. Lernkurven prüfen (validation_curves.png)."
        )
    ranks = [(int(re.search(r"lora_r(\d+)$", g).group(1)), g) for g in lora if re.search(r"lora_r\d+$", g)]
    if len(ranks) > 1:
        ranks.sort()
        trend = " → ".join(f"r={r}: {100 * acc(g):.1f} %" for r, g in ranks)
        out.append(
            f"- **LoRA-Rang:** {trend}. Steigt die Accuracy kaum, reicht ein kleiner Rang (abnehmender Ertrag, "
            "Hypothese A3); entscheidend ist dann, *welche* Layer angepasst werden."
        )
    if "ms1_vit_b16_lora_r8_all" in main_methods.index and "ms1_vit_b16_lora_r8" in main_methods.index:
        out.append(
            f"- **Ziel-Layer:** alle Linear-Layer vs. nur Query/Value (r = 8): "
            f"{gap_text('ms1_vit_b16_lora_r8_all', 'ms1_vit_b16_lora_r8')}. Hu et al. (2022) passen nur Query/Value "
            "an; mehr Layer geben mehr Freiheitsgrade, aber auch mehr Parameter."
        )
    vit = [g for g in main_methods.index if g.startswith("ms1_vit")]
    cnn = [g for g in main_methods.index if g in ("ms1_resnet50", "ms1_convnext_t")]
    if vit and cnn:
        bv, bc = max(vit, key=acc), max(cnn, key=acc)
        out.append(
            f"- **ViT vs. CNN:** bester ViT ({main_methods.loc[bv, 'label']}) gegenüber bestem CNN "
            f"({main_methods.loc[bc, 'label']}) {gap_text(bv, bc)}. Einordnen: unterschiedliches Pretraining "
            "(ViT ImageNet-21k, CNNs ImageNet-1k), induktive Verzerrung der CNNs hilft bei wenig Daten "
            "(Vorlesung 03), ConvNeXt übernimmt Designentscheidungen der Transformer (Liu et al. 2022)."
        )
    if len(aug_methods):
        rows = {re.search(r"_aug_(none|strong)$", g).group(1): r["accuracy_mean"] for g, r in aug_methods.iterrows()}
        base = next(
            (
                main_methods.loc[re.sub(r"_aug_(none|strong)$", "", g), "accuracy_mean"]
                for g in aug_methods.index
                if re.sub(r"_aug_(none|strong)$", "", g) in main_methods.index
            ),
            None,
        )
        if base is not None and rows:
            order = sorted(
                [("none", rows.get("none")), ("light", base), ("strong", rows.get("strong"))],
                key=lambda kv: -1 if kv[1] is None else kv[1],
                reverse=True,
            )
            out.append(
                "- **Augmentierung:** Reihenfolge "
                + " > ".join(f"{k} ({100 * v:.1f} %)" for k, v in order if v is not None)
                + ". Erklärungsansatz: Augmentierung wirkt bei wenig Daten als Regularisierung (Vorlesung 01, "
                "Steiner et al. 2022); zu starke Augmentierung verändert Rassemerkmale (Fellfarbe, Form) und "
                "schadet (Vorlesung 02: „zu viel Augmentierung vermeiden“)."
            )
    for _, r in main_methods.iterrows():
        ece, epoch = r.get("ece_mean"), r.get("best_epoch_mean")
        if ece is not None and ece == ece and ece > 0.3:  # mean confidence far below accuracy
            out.append(
                f"- **Achtung, {r['label']} ist schlecht kalibriert (ECE {ece:.2f}, beste Epoche {epoch:.1f}).** "
                "Liegt die beste Epoche am Ende des Trainings und ist der Validation-Loss noch hoch "
                "(`runs/<name>/history.json`), ist das Modell nicht konvergiert: Lernrate oder Epochenzahl passen "
                "nicht zur Architektur (CNN-Kopf und BatchNorm brauchen meist eine höhere Lernrate als ein ViT). "
                "Dann ist der Vergleich mit dem ViT unfair; Lernrate auf Validation nachjustieren "
                "(`make ms1-cnn-lr`) oder als Limitation nennen."
            )
    if conf is not None and len(conf):
        top = conf.iloc[0]
        out.append(
            f"- **Fehler:** häufigste Verwechslung {top['true']} → {top['predicted']} ({top['count']}×). "
            "Mit Beispielbildern begründen: optisch ähnliche Rassen, Welpen, ungewöhnliche Perspektiven?"
        )
    out += [
        "",
        "**Fragen, die die Diskussion beantworten sollte:** Welche Methode ist am besten und warum? Lohnt sich "
        "LoRA gegenüber Linear Probing und vollem Fine-Tuning (Genauigkeit pro Parameter, Trainingszeit)? Sind die "
        "Unterschiede größer als die Seed-Streuung? Auf welchen Daten würde eine andere Methode gewinnen?",
    ]
    return out


if __name__ == "__main__":
    main()
