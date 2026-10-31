"""Report of ``make why``: why do the models reach more than 90 % on the 25 dog breeds?

Reads the from-scratch runs (``runs/why_*``), the station runs and the outputs of
``scripts/why_analysis.py`` in ``results/why/`` and writes ``results/why/REPORT.md`` with
figures. Each section tests one candidate explanation; missing inputs are skipped.

``scratch_vs_pretrained.png``  same architecture with and without pretraining
``fewshot_curves.png``         linear probe accuracy vs. images per breed (ImageNet backbones, CLIP)
``max_sim_hist.png``           similarity of each test image to its nearest train/val image
``confident_errors.png``       test images the best model gets wrong in every seed

Example::

    python scripts/report_why.py --runs-dir runs --out results/why
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR, RUNS_DIR
from animal_vit.report_tools import aggregate, base_row, figure, md_table, paired_delta, per_breed_f1, pm, save
from animal_vit.why import HARD_SUBSET, IMAGENET_IDS, NOT_IN_IMAGENET

#: Pretrained counterpart (station 1) of each from-scratch group: same architecture, all weights trained.
SCRATCH_PAIRS = {
    "why_vit_b16_scratch": ("ms1_vit_b16", "ViT-B/16"),
    "why_resnet50_scratch": ("ms1_resnet50", "ResNet-50"),
}
#: Groups for the ImageNet-overlap table.
OVERLAP_GROUPS = {
    "ms1_vit_b16_head": "ViT linear probing",
    "ms1_vit_b16_lora_r8": "ViT LoRA r=8",
    "ms1_vit_b16": "ViT full fine-tuning",
    "ms1_resnet50": "ResNet-50",
    "ms1_convnext_t": "ConvNeXt-T",
    "ms2_deit_tiny": "DeiT-Ti (no KD)",
    "ms3_clip_vit_l14_dog": "CLIP ViT-L/14 zero-shot",
}
#: Link of each check to the lectures (docs/CONTEXT.md, section 2), printed under each heading.
LECTURES = {
    1: (
        "Transfer Learning statt Neutraining und die Faustregel „mindestens 1000 Bilder pro Klasse“ "
        "(Vorlesung 02, wir haben ca. 90); ViTs brauchen viele Daten, CNNs haben eine induktive Verzerrung "
        "(Vorlesung 02/03)."
    ),
    2: (
        "Die Wahl der Transfer-Methode hängt von der *Ähnlichkeit zum Originaldatensatz* ab (Vorlesung 02); "
        "hier wird diese Ähnlichkeit gemessen. Fine-Tuning-Tabelle für kleine Datensätze: „nur Head "
        "trainieren“ (Vorlesung 03)."
    ),
    3: (
        "„Hohe Accuracy beweist nicht, dass das Richtige gelernt wurde“ (Zhang et al. 2017, Vorlesung 04) und"
        " saubere Trennung von Train und Test (Vorlesung 02, Praxis-Strategien). Die Suche nutzt Bild-"
        "Embeddings und Ähnlichkeit wie Bildsuche und Siamese Networks (Vorlesung 05)."
    ),
    4: (
        "Linear Probing / „mit der letzten Schicht beginnen“ (Vorlesung 02, Station 1) und Embeddings, in "
        "denen ähnliche Bilder nah beieinander liegen (Similarity Learning, Vorlesung 05)."
    ),
    5: (
        "„Loss und Labels prüfen“ (Vorlesung 02, Praxis-Strategien), Outlier betrachten und Bias beim Labeln "
        "(Vorlesung 04)."
    ),
}
DUPLICATE_THRESHOLD = 0.95
CHANCE = 1 / 25


def runs_of(runs_dir: Path, group: str) -> list[Path]:
    return sorted(p for p in runs_dir.glob(f"{group}_s*") if (p / "metrics.json").exists()) or (
        [runs_dir / group] if (runs_dir / group / "metrics.json").exists() else []
    )


def best_vit_group(results_root: Path) -> str:
    """Station-1 ViT chosen on validation (station 2 teacher), fallback LoRA r=8."""
    teachers = results_root / "ms1" / "best_models.json"
    if teachers.exists():
        return json.loads(teachers.read_text())["vit"]["group"]
    return "ms1_vit_b16_lora_r8"


def main() -> None:  # noqa: C901 - one linear report, one section per check
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "why")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs_dir = args.runs_dir
    findings: list[str] = []

    lines = [
        "# Warum über 90 %? Belege aus Experimenten",
        "",
        "Automatisch erzeugt von `scripts/report_why.py` (`make why`). Testsplit der 25 Hunderassen, "
        "Mittelwert ± Std über Seeds. Jeder Abschnitt prüft eine mögliche Erklärung.",
    ]

    # ------------------------------------------------------------------ 1. pretraining
    lines += [
        "",
        "## 1. Pretraining: dieselbe Architektur ohne vortrainierte Gewichte",
        "",
        f"> **Vorlesungsbezug:** {LECTURES[1]}",
        "",
    ]
    rows = [
        r
        for g in SCRATCH_PAIRS
        for pair in (g, SCRATCH_PAIRS[g][0])
        for p in runs_of(runs_dir, pair)
        if (r := base_row(p))
    ]
    if rows:
        df = pd.DataFrame(rows)
        agg = aggregate(df)
        table, bars = [], []
        for scratch, (pre, label) in SCRATCH_PAIRS.items():
            if scratch not in agg.index or pre not in agg.index:
                continue
            s, p = agg.loc[scratch], agg.loc[pre]
            d = paired_delta(df, scratch, pre)
            table.append(
                [
                    label,
                    pm(p.accuracy_mean, p.accuracy_std),
                    pm(s.accuracy_mean, s.accuracy_std),
                    f"{100 * (p.accuracy_mean - s.accuracy_mean):+.1f} pp",
                    pm(s.f1_macro_mean, s.f1_macro_std),
                    int(s.n_seeds),
                ]
            )
            bars.append((label, p.accuracy_mean, p.accuracy_std, s.accuracy_mean, s.accuracy_std))
            findings.append(
                f"- **Pretraining ({label}):** mit ImageNet-Gewichten {100 * p.accuracy_mean:.1f} %, ohne "
                f"{100 * s.accuracy_mean:.1f} % (Zufall {100 * CHANCE:.0f} %), Differenz "
                f"{100 * (p.accuracy_mean - s.accuracy_mean):.1f} pp"
                + (" (gepaart über Seeds, > 2·SE)" if d and d["significant"] else "")
                + ". Der größte Teil der Accuracy stammt aus dem Pretraining, nicht aus den ca. 2 250 "
                "Trainingsbildern. "
                + (
                    "Beim ViT ist der Einbruch erwartungsgemäß stärker: ohne die induktive Verzerrung der "
                    "Faltung braucht er sehr viele Daten (Dosovitskiy et al. 2021)."
                    if label.startswith("ViT")
                    else "Die CNN profitiert ohne Pretraining von ihrer induktiven Verzerrung (Lokalität, "
                    "Translationsinvarianz; Vorlesung 02/03)."
                )
            )
        lines += md_table(
            ["Architektur", "mit Pretraining (Station 1)", "ohne Pretraining", "Differenz", "Macro-F1 ohne", "Seeds"],
            table,
        )
        lines += [
            "",
            "Gleiche Daten, Splits und Auswertung; ohne Pretraining 30 statt 10 Epochen und eine größere "
            "Lernrate (`configs/why/`), damit das Training eine faire Chance hat.",
        ]
        if bars:
            fig, ax = figure(5.0, 3.6)
            x = np.arange(len(bars))
            ax.bar(x - 0.2, [100 * b[1] for b in bars], 0.4, yerr=[100 * b[2] for b in bars], label="pretrained")
            ax.bar(x + 0.2, [100 * b[3] for b in bars], 0.4, yerr=[100 * b[4] for b in bars], label="from scratch")
            ax.axhline(100 * CHANCE, color="grey", linestyle=":", label="chance (4 %)")
            ax.set_xticks(x, [b[0] for b in bars])
            ax.set(ylabel="test accuracy (%)", ylim=(0, 100), title="Same architecture, with and without pretraining")
            ax.legend(fontsize=8)
            save(fig, args.out / "scratch_vs_pretrained.png")
    else:
        lines += ["(noch keine Läufe, `make why`)"]

    # ------------------------------------------------------------------ 2. prior knowledge
    lines += [
        "",
        "## 2. Vorwissen: 22 der 25 Rassen sind ImageNet-1k-Klassen",
        "",
        f"> **Vorlesungsbezug:** {LECTURES[2]}",
        "",
    ]
    lines += [
        f"In ImageNet-1k (Russakovsky et al. 2015) fehlen nur {', '.join(NOT_IN_IMAGENET)}. Die Checkpoints von "
        "Station 1 sind auf ImageNet-1k trainiert, haben diese Rassen also schon als Klassen gelernt.",
        "",
    ]
    heads = args.out / "imagenet_head.csv"
    if heads.exists():
        h = pd.read_csv(heads)
        lines += md_table(
            [
                "Modell (Original-Kopf, kein Training)",
                "Rassen",
                "Testbilder",
                "Acc 22 Rassen (%)",
                "Top-1 aus 1000 (%)",
                "Top-5 aus 1000 (%)",
            ],
            [
                [
                    r.model,
                    r.breeds,
                    r.test_images,
                    f"{100 * r.accuracy_22way:.2f}",
                    f"{100 * r.accuracy_1000way_top1:.2f}",
                    f"{100 * r.accuracy_1000way_top5:.2f}",
                ]
                for r in h.itertuples()
            ],
        )
        best = h.sort_values("accuracy_22way").iloc[-1]
        # Fair reference: the trained model on exactly these 22 breeds (its 25-way predictions, so it can
        # still confuse them with the 3 extra breeds - a slightly harder task than the 22-way head).
        group = best_vit_group(args.out.parent)
        ft = [pd.read_csv(r / "predictions.csv") for r in runs_of(runs_dir, group) if (r / "predictions.csv").exists()]
        ft_acc = float(np.mean([d[d.true.isin(list(IMAGENET_IDS))].correct.mean() for d in ft])) if ft else None
        if ft_acc is not None:
            lines += [
                "",
                f"Zum Vergleich auf denselben 22 Rassen: das trainierte `{group}` {100 * ft_acc:.2f} % "
                "(25 Klassen zur Auswahl, also eher schwerer als die 22-Klassen-Auswahl des ImageNet-Kopfs).",
            ]
        findings.append(
            f"- **Vorwissen:** Ohne ein einziges Trainingsbild erkennt der ImageNet-Kopf von {best.model} die 22 "
            f"Rassen zu {100 * best.accuracy_22way:.1f} %"
            + (f"; das trainierte Modell erreicht auf denselben Rassen {100 * ft_acc:.1f} %" if ft_acc else "")
            + ". Unser Training lernt die Aufgabe also nicht neu, sondern passt eine schon vorhandene "
            "Unterscheidung an. Das relativiert die hohen Werte von Station 1 und erklärt, warum Linear Probing (nur "
            "ein neuer Kopf) so gut ist. Fairness: Der ImageNet-Kopf wählt nur zwischen 22 Rassen und kennt den "
            "schwierigen American Bulldog nicht."
        )
    else:
        lines += ["(fehlt: `python scripts/why_analysis.py --only heads`)"]
    groups = {g: runs_of(runs_dir, g) for g in OVERLAP_GROUPS}
    rows = []
    for g, runs in groups.items():
        f1 = per_breed_f1(runs) if runs else None
        if f1 is None:
            continue
        inn = [b for b in f1.index if b in IMAGENET_IDS]
        out = [b for b in f1.index if b in NOT_IN_IMAGENET]
        rows.append(
            [
                OVERLAP_GROUPS[g],
                f"{100 * f1[inn].mean():.1f}",
                f"{100 * f1[out].mean():.1f}",
                ", ".join(f"{b} {100 * f1[b]:.0f}" for b in out),
            ]
        )
    if rows:
        lines += ["", "F1 je Rassengruppe (Station-Läufe, Mittel über Seeds):", ""]
        lines += md_table(["Modell", "F1 in ImageNet (22)", "F1 nicht in ImageNet (3)", "einzeln (%)"], rows)
        findings.append(
            "- **Rassen ohne ImageNet-Klasse** sind nur wenig schlechter, und der Abstand kommt vor allem von "
            "American Bulldog (verwechselbar mit Pit Bull). Havanese und Shiba Inu erkennen die Modelle trotzdem "
            "gut: Die Merkmale übertragen sich auf neue, ähnliche Klassen. Einschränkung: nur 3 Rassen."
        )

    # ------------------------------------------------------------------ 3. leakage
    lines += [
        "",
        "## 3. Leakage: Fast-Duplikate zwischen Test und Training",
        "",
        f"> **Vorlesungsbezug:** {LECTURES[3]}",
        "",
    ]
    sims = args.out / "test_max_sim.csv"
    if sims.exists():
        s = pd.read_csv(sims)
        counts = [
            [f"≥ {t}", int((s.max_similarity >= t).sum()), f"{100 * (s.max_similarity >= t).mean():.2f} %"]
            for t in (0.90, 0.95, 0.98)
        ]
        lines += ["Kosinus-Ähnlichkeit (CLIP ViT-B/16) jedes Testbilds zum ähnlichsten Train-/Validation-Bild:", ""]
        lines += md_table(["Ähnlichkeit", "Testbilder", "Anteil"], counts)
        fig, ax = figure(5.0, 3.2)
        ax.hist(s.max_similarity, bins=50)
        ax.axvline(DUPLICATE_THRESHOLD, color="red", linestyle="--", label=f"{DUPLICATE_THRESHOLD}")
        ax.set(
            xlabel="cosine similarity to nearest train/val image",
            ylabel="test images",
            title="Near-duplicate check (CLIP ViT-B/16)",
        )
        ax.legend(fontsize=8)
        save(fig, args.out / "max_sim_hist.png")
        dup = s[s.max_similarity >= DUPLICATE_THRESHOLD]
        best_group = best_vit_group(args.out.parent)
        accs = []
        for run in runs_of(runs_dir, best_group):
            pred = pd.read_csv(run / "predictions.csv")
            if len(pred) == len(s):
                clean = ~pred["index"].isin(dup.test_index)
                accs.append((pred.correct.mean(), pred.correct[clean].mean()))
        if accs:
            a_all, a_clean = np.mean([a for a, _ in accs]), np.mean([c for _, c in accs])
            lines += [
                "",
                f"Bester ViT (`{best_group}`): alle Testbilder {100 * a_all:.2f} %, ohne die "
                f"{len(dup)} Bilder ≥ {DUPLICATE_THRESHOLD}: {100 * a_clean:.2f} %.",
            ]
            findings.append(
                f"- **Leakage:** {len(dup)} Testbilder ({100 * len(dup) / len(s):.1f} %) haben ein sehr ähnliches "
                f"Trainingsbild (≥ {DUPLICATE_THRESHOLD}); ohne sie ändert sich die Accuracy um "
                f"{100 * (a_clean - a_all):+.2f} pp. "
                + (
                    "Die hohen Werte sind also nicht durch Duplikate aufgebläht. "
                    if abs(a_clean - a_all) < 0.005
                    else "Das ist ein messbarer Effekt und gehört in die Limitationen. "
                )
                + "`duplicates.png` ansehen: echte Duplikate oder nur ähnliche Motive (gleicher Hund, andere Pose)?"
            )
        same = (dup.test_breed == dup.neighbour_breed).mean() if len(dup) else float("nan")
        if len(dup):
            lines += [
                "",
                f"Davon gleiche Rasse wie der Nachbar: {100 * same:.0f} %. Abbildungen: `duplicates.png`, "
                "`max_sim_hist.png`.",
            ]
    else:
        lines += ["(fehlt: `python scripts/why_analysis.py --only duplicates`)"]

    # ------------------------------------------------------------------ 4. separability
    lines += [
        "",
        "## 4. Trennbarkeit: wie wenige Bilder reichen?",
        "",
        f"> **Vorlesungsbezug:** {LECTURES[4]}",
        "",
    ]
    fs_path = args.out / "fewshot_backbones.csv"
    curves = {}
    if fs_path.exists():
        fs = pd.read_csv(fs_path)
        for model, part in fs.groupby("model"):
            k = part[part.shots != "all"].assign(shots=lambda d: d.shots.astype(int))
            curves[model] = k.groupby("shots").accuracy.agg(["mean", "std"])
            full = part[part.shots == "all"].accuracy
            curves[model].attrs["all"] = float(full.iloc[0]) if len(full) else None
    for tag, label in (("vit_b16", "CLIP ViT-B/16"), ("vit_l14", "CLIP ViT-L/14")):
        rows = []
        for run in runs_dir.glob(f"ms3_clip_{tag}_fewshot_k*_s*"):
            if (run / "metrics.json").exists():
                k = int(run.name.split("_k")[1].split("_")[0])
                rows.append(
                    {"shots": k, "accuracy": json.loads((run / "metrics.json").read_text())["test_metrics"]["accuracy"]}
                )
        if rows:
            curves[label] = pd.DataFrame(rows).groupby("shots").accuracy.agg(["mean", "std"])
    if curves:
        shots = sorted({k for c in curves.values() for k in c.index})
        lines += md_table(
            ["Merkmale (eingefroren) + Linear Probe", *[f"k={k}" for k in shots], "alle Bilder"],
            [
                [
                    m,
                    *[pm(c.loc[k, "mean"], c.loc[k, "std"]) if k in c.index else "-" for k in shots],
                    f"{100 * c.attrs['all']:.2f}" if c.attrs.get("all") is not None else "-",
                ]
                for m, c in curves.items()
            ],
        )
        fig, ax = figure(5.5, 3.6)
        for m, c in curves.items():
            ax.errorbar(c.index, 100 * c["mean"], yerr=100 * c["std"].fillna(0), marker="o", capsize=3, label=m)
        ax.set_xscale("log", base=2)
        ax.set_xticks(shots, [str(k) for k in shots])
        ax.axhline(100 * CHANCE, color="grey", linestyle=":", label="chance")
        ax.set(
            xlabel="training images per breed (k)", ylabel="test accuracy (%)", title="Linear probe on frozen features"
        )
        ax.legend(fontsize=7)
        save(fig, args.out / "fewshot_curves.png")
        vit = curves.get("ViT-B/16 (ImageNet)")
        if vit is not None and 1 in vit.index:
            findings.append(
                f"- **Trennbarkeit:** Mit eingefrorenen ImageNet-ViT-Merkmalen reicht **ein** Bild pro Rasse für "
                f"{100 * vit.loc[1, 'mean']:.0f} %"
                + (f", 16 Bilder für {100 * vit.loc[16, 'mean']:.0f} %" if 16 in vit.index else "")
                + ". Die Rassen liegen im Merkmalsraum schon weitgehend linear getrennt; die übrigen "
                "Trainingsbilder bringen vergleichsweise wenig dazu. "
                "Vergleich mit CLIP: welches Pretraining liefert die besser trennbaren Merkmale?"
            )
        # Frozen features vs. full fine-tuning of the same backbone (station 1)
        pairs = {
            "ViT-B/16 (ImageNet)": ("ms1_vit_b16", "ViT-B/16"),
            "ResNet-50 (ImageNet)": ("ms1_resnet50", "ResNet-50"),
        }
        cmp = []
        for key, (group, label) in pairs.items():
            frozen = curves.get(key)
            runs = runs_of(runs_dir, group)
            if frozen is not None and frozen.attrs.get("all") is not None and runs:
                full = np.mean([json.loads((r / "metrics.json").read_text())["test_metrics"]["accuracy"] for r in runs])
                cmp.append((label, frozen.attrs["all"], full))
        if cmp:
            lines += [
                "",
                "Eingefrorene Merkmale (Linear Probe, alle Bilder) gegen volles Fine-Tuning aus Station 1:",
                "",
            ]
            lines += md_table(
                ["Backbone", "eingefroren + Linear Probe (%)", "volles Fine-Tuning (%)", "Differenz"],
                [[lab, f"{100 * fr:.2f}", f"{100 * fu:.2f}", f"{100 * (fr - fu):+.2f} pp"] for lab, fr, fu in cmp],
            )
            if all(fr > fu for _, fr, fu in cmp):
                findings.append(
                    "- **Eingefroren schlägt volles Fine-Tuning:** "
                    + ", ".join(f"{lab} {100 * fr:.1f} % gegenüber {100 * fu:.1f} %" for lab, fr, fu in cmp)
                    + ". Volles Fine-Tuning auf ca. 2 250 Bildern verschiebt die guten ImageNet-Merkmale eher, "
                    "als dass "
                    "es sie verbessert (Kumar et al., ICLR 2022: „Fine-Tuning can Distort Pretrained Features“). Das "
                    "stützt die Empfehlung der Vorlesung 03, bei kleinen Datensätzen nur den Kopf zu trainieren. "
                    "Einschränkung: Die Linear Probe ist eine logistische Regression mit auf Validation gewähltem C, "
                    "das Fine-Tuning lief 10 Epochen mit festem Schema; nur ein Lauf je Probe."
                )

    # ------------------------------------------------------------------ 5. remaining error
    lines += [
        "",
        "## 5. Wo liegt der Rest-Fehler?",
        "",
        f"> **Vorlesungsbezug:** {LECTURES[5]}",
        "",
    ]
    best_group = best_vit_group(args.out.parent)
    preds = [
        pd.read_csv(r / "predictions.csv") for r in runs_of(runs_dir, best_group) if (r / "predictions.csv").exists()
    ]
    if preds:
        p = pd.concat(preds, keys=range(len(preds)), names=["seed_i", "row"]).reset_index()
        hard = p.true.isin(HARD_SUBSET)
        errors = p[~p.correct]
        in_hard = errors.true.isin(HARD_SUBSET) & errors.predicted.isin(HARD_SUBSET)
        lines += md_table(
            ["Teilmenge", "Anteil der Testbilder", "Accuracy (%)"],
            [
                [
                    f"{len(HARD_SUBSET)} Bully-Rassen ({', '.join(HARD_SUBSET)})",
                    f"{100 * hard.mean():.0f} %",
                    f"{100 * p.correct[hard].mean():.2f}",
                ],
                ["übrige Rassen", f"{100 * (~hard).mean():.0f} %", f"{100 * p.correct[~hard].mean():.2f}"],
            ],
        )
        lines += [
            "",
            f"Modell: `{best_group}` (auf Validation gewählt), {len(preds)} Seeds. Fehler innerhalb der "
            f"Bully-Rassen: {100 * in_hard.mean():.0f} % aller Fehler.",
        ]
        findings.append(
            f"- **Rest-Fehler:** {100 * hard.mean():.0f} % der Testbilder (Bully-Rassen) verursachen "
            f"{100 * in_hard.mean():.0f} % aller Fehler; dort liegt die Accuracy bei "
            f"{100 * p.correct[hard].mean():.1f} % statt {100 * p.correct[~hard].mean():.1f} %. Diese Rassen "
            "unterscheiden sich vor allem in Größe und Proportionen, die auf einem Foto ohne Maßstab kaum "
            "erkennbar sind; selbst die Rassestandards überschneiden sich (American Pit Bull vs. Staffordshire)."
        )
        always = p.groupby("index").agg(
            wrong=("correct", lambda c: (~c).all()),
            conf=("confidence", "mean"),
            true=("true", "first"),
            predicted=("predicted", lambda x: x.mode()[0]),
        )
        always = always[always.wrong].sort_values("conf", ascending=False)
        always.head(20).to_csv(args.out / "confident_errors.csv")
        lines += [
            "",
            f"In allen Seeds falsch: {len(always)} Testbilder. Die sichersten davon "
            "(`confident_errors.csv`, `confident_errors.png`) von Hand prüfen: Label-Fehler im Datensatz?",
        ]
        findings.append(
            f"- **Label-Rauschen?** {len(always)} Testbilder sind in jedem Seed falsch, viele davon mit hoher "
            "Konfidenz. Bei der Durchsicht von `confident_errors.png` zählen: falsches Label, Mischling, mehrere "
            "Hunde, Hund kaum sichtbar. Ein Teil des Rest-Fehlers ist dann keine Modellschwäche, sondern eine "
            "Obergrenze des Datensatzes."
        )
        try:
            plot_errors(always.head(8), args.out / "confident_errors.png")
        except Exception as err:  # dataset not available on this machine
            print(f"confident_errors.png skipped: {err}")

    lines += [
        "",
        "## 6. Befunde (Material für die Diskussion)",
        "",
        "Aus den Zahlen abgeleitet; Formulierung und Gewichtung sind eure Aufgabe.",
        "",
        *findings,
    ]
    lines += [
        "",
        "**Gesamtbild prüfen:** Ergeben Pretraining (1), Vorwissen (2) und Trennbarkeit (4) zusammen eine "
        "schlüssige Erklärung, und schließt (3) Leakage aus? Was bedeutet das für die Aussagekraft von 96 % "
        "auf einem Datensatz, dessen Klassen im Pretraining vorkamen? Auf welchem Datensatz wäre die Aufgabe "
        "wirklich schwer (Rassen außerhalb von ImageNet, wenige Bilder, andere Domäne)?",
        "",
        "## Dateien",
        "",
        "- `scratch_vs_pretrained.png`, `fewshot_curves.png`, `max_sim_hist.png`, `duplicates.png`, "
        "`confident_errors.png`",
        "- `imagenet_head.csv`, `test_max_sim.csv`, `fewshot_backbones.csv`, `confident_errors.csv`",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out / 'REPORT.md'}")


def plot_errors(errors, path: Path) -> None:
    """Test images wrong in every seed, with true and predicted breed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from animal_vit.data import build_datasets, build_transforms

    datasets, _ = build_datasets("oxford_pets", build_transforms(224, [0.5] * 3, [0.5] * 3, "none"))
    test = datasets["test"]
    n = len(errors)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6), squeeze=False)
    for ax, (idx, r) in zip(axes[0], errors.iterrows()):
        ax.imshow(test.load(test.indices[int(idx)])[0].convert("RGB"))
        ax.set_title(f"true: {r.true}\npred: {r.predicted} ({r.conf:.2f})", fontsize=6)
        ax.axis("off")
    fig.suptitle("Wrong in every seed (most confident first)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
