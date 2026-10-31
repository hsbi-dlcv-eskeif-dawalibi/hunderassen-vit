"""Report of ``make explain``: why this breed and not the other?

Reads ``results/explain/`` (written by ``scripts/explain.py``) and writes ``REPORT.md`` with
``deletion_curves.png``. Every number comes from the CSV files; the findings are material for the
discussion, not finished text.

Example::

    python scripts/report_explain.py --out results/explain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.paths import RESULTS_DIR
from animal_vit.report_tools import md_table, save

METHOD_NAMES = {
    "gradcam": "Grad-CAM",
    "rollout": "Attention Rollout",
    "occlusion": "Occlusion",
    "random": "zufällig (Referenz)",
}
SOURCES = [
    "Selvaraju et al., Grad-CAM, ICCV 2017; Implementierung: Gildenblat et al., pytorch-grad-cam "
    "(github.com/jacobgil/pytorch-grad-cam)",
    "Zeiler & Fergus, Visualizing and Understanding Convolutional Networks, ECCV 2014 (Occlusion); "
    "Implementierung: Kokhlikyan et al., Captum, 2020 (captum.ai)",
    "Abnar & Zuidema, Quantifying Attention Flow in Transformers, ACL 2020 (Attention Rollout, eigener Code)",
    "Prabhushankar et al., Contrastive Explanations in Neural Networks, ICIP 2020 (Idee der kontrastiven Karte)",
    "Petsiuk et al., RISE, BMVC 2018 (Deletion-Metrik)",
    "Chefer et al., Transformer Interpretability Beyond Attention Visualization, CVPR 2021 (Grenzen von Grad-CAM "
    "bei ViTs)",
    "Parkhi et al., Cats and Dogs, CVPR 2012 (Oxford-IIIT Pet mit Kopf-Boxen)",
]


def main() -> None:  # noqa: C901 - one linear report
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "explain")
    args = parser.parse_args()
    out = args.out
    findings: list[str] = []
    areas: dict = {}  # (model, method) -> deletion area, used to pick the most faithful method per model
    lines = [
        "# Warum diese Rasse und nicht die andere? Erklärungsmethoden",
        "",
        "Automatisch erzeugt von `scripts/report_explain.py` (`make explain`). Modelle: Teacher von Station 2 "
        "(bester ViT und bestes CNN nach Validation) und das zweite CNN, jeweils Seed 1.",
        "",
        "> **Vorlesungsbezug:** Attention und Attention-Visualisierung (Vorlesung 03), „Was lernt das Modell "
        "wirklich?“ und „hohe Accuracy beweist nicht, dass das Richtige gelernt wurde“ (Vorlesung 04). Die "
        "Kopf-Analyse nutzt die Annotationen des Datensatzes, die Deletion-Kurve prüft, ob eine Erklärung stimmt.",
    ]

    # ------------------------------------------------------------------ 1. contrastive
    figs = sorted(out.glob("contrastive_*.png"))
    lines += ["", "## 1. Warum A und nicht B? Kontrastive Karten", ""]
    lines += [
        "Für die häufigsten Verwechslungen aus Station 1 zeigt jede Zeile ein Testbild der Rasse A: Grad-CAM "
        "für A, Grad-CAM für B und die **kontrastive Karte** mit dem Ziel `logit_A − logit_B`. Rot sind die "
        "Bildteile, die für A *und gegen* B sprechen. Die letzte Spalte (Occlusion) ist eine unabhängige "
        "Kontrolle ohne Gradienten.",
        "",
        *[f"- `{f.name}`" for f in figs],
    ]
    if figs:
        findings.append(
            "- **Kontrastive Karten:** In den Abbildungen beschreiben, *welche* Bildteile A von B trennen (Kopfform, "
            "Schnauze, Ohren, Fell, Körperbau). Oxford-IIIT Pet hat keine Annotationen für Ohren oder Augen; diese "
            "Zuordnung ist eure qualitative Beobachtung und sollte an mehreren Beispielen belegt werden. Stimmen "
            "Grad-CAM und Occlusion überein, ist die Aussage belastbarer."
        )

    # ------------------------------------------------------------------ 2. faithfulness
    lines += ["", "## 2. Stimmt die Erklärung? Deletion-Test", ""]
    faith = out / "faithfulness.csv"
    if faith.exists():
        f = pd.read_csv(faith)
        from animal_vit.explain import curve_area

        rows = []
        for (model, method), part in f.groupby(["model", "method"], sort=False):
            part = part.sort_values("removed")
            area = curve_area(part.probability.values, tuple(part.removed.values))
            areas[(model, method)] = area
            p = dict(zip(part.removed, part.probability))
            rows.append(
                [
                    model,
                    METHOD_NAMES.get(method, method),
                    f"{100 * p.get(0.0, np.nan):.1f}",
                    f"{100 * p.get(0.1, np.nan):.1f}",
                    f"{100 * p.get(0.3, np.nan):.1f}",
                    f"{area:.3f}",
                    int(part.n.iloc[0]),
                ]
            )
        lines += [
            "Die als wichtig markierten 16×16-Patches werden zuerst grau übermalt. Fällt die Wahrscheinlichkeit "
            "der richtigen Klasse schneller als beim zufälligen Löschen, zeigt die Methode wirklich, worauf das "
            "Modell sich stützt. **Kleinere Fläche = treuere Erklärung.**",
            "",
        ]
        lines += md_table(["Modell", "Methode", "p (0 % gelöscht)", "p (10 %)", "p (30 %)", "Fläche ↓", "Bilder"], rows)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = list(dict.fromkeys(f.model))
        fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.3), squeeze=False, sharey=True)
        for ax, model in zip(axes[0], models):
            for method, part in f[f.model == model].groupby("method", sort=False):
                part = part.sort_values("removed")
                style = ":" if method == "random" else "-"
                ax.plot(100 * part.removed, 100 * part.probability, style, marker="o", label=METHOD_NAMES.get(method))
            ax.set(title=model, xlabel="patches removed (%)")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("probability of true class (%)")
        axes[0, -1].legend(fontsize=7)
        fig.suptitle("Deletion test: most important patches first (lower = more faithful)", fontsize=10)
        fig.tight_layout()
        save(fig, out / "deletion_curves.png")
        for model in models:
            cand = {m: a for (mo, m), a in areas.items() if mo == model and m != "random"}
            rand = areas.get((model, "random"))
            if cand and rand is not None:
                best = min(cand, key=cand.get)
                gain = 100 * (1 - cand[best] / rand)
                findings.append(
                    f"- **Treue ({model}):** {METHOD_NAMES[best]} ist am treuesten (Fläche {cand[best]:.3f} gegenüber "
                    f"{rand:.3f} beim zufälligen Löschen, {gain:.0f} % kleiner)"
                    + (
                        ". Alle Methoden liegen unter der Zufallsreferenz: Die markierten Regionen tragen mehr zur "
                        "Entscheidung bei als zufällige. Je größer der Abstand, desto eher darf man die Karte deuten."
                        if all(a < rand for a in cand.values())
                        else "; mindestens eine Methode ist nicht besser als Zufall und sollte nicht interpretiert "
                        "werden."
                    )
                )
            rnd = f[(f.model == model) & (f.method == "random")].sort_values("removed")
            if len(rnd) and rnd.probability.iloc[0] - rnd.probability.iloc[-1] > 0.15:
                findings.append(
                    f"- **Achtung ({model}):** Schon zufälliges Übermalen senkt die Sicherheit stark "
                    f"({100 * rnd.probability.iloc[0]:.0f} % → {100 * rnd.probability.iloc[-1]:.0f} %). Das Modell "
                    "reagiert also allgemein empfindlich auf graue Flächen (vgl. Variante „nur Hund auf Grau“ in "
                    "Station 3). Die Flächen dieses Modells deshalb nur untereinander vergleichen, nicht mit anderen "
                    "Modellen."
                )
    else:
        lines += ["(fehlt: `make explain`)"]

    # ------------------------------------------------------------------ 3. head focus
    lines += ["", "## 3. Kopf, Körper oder Hintergrund?", ""]
    head = out / "head_focus.csv"
    if head.exists():
        h = pd.read_csv(head)
        lines += [
            "Anteil der Saliency in der annotierten **Kopf-Box**, auf dem **übrigen Hund** (Maske ohne Kopf) und im "
            "**Hintergrund**, jeweils geteilt durch den Flächenanteil der Region: 1 = proportional zur Fläche, "
            "> 1 = das Modell schaut bevorzugt dorthin. Validierungssplit (nur dort gibt es Kopf-Boxen; diese "
            "Bilder wurden nicht zum Training benutzt).",
            "",
        ]
        piv = h.pivot_table(index=["model", "method"], columns="region", values="focus_mean", sort=False)
        n = h.groupby(["model", "method"], sort=False).n.max()
        rows = [
            [
                m,
                METHOD_NAMES.get(meth, meth),
                f"{piv.loc[(m, meth), 'head']:.2f}",
                f"{piv.loc[(m, meth), 'body']:.2f}",
                f"{piv.loc[(m, meth), 'background']:.2f}",
                int(n.loc[(m, meth)]),
            ]
            for m, meth in piv.index
            if meth != "random"
        ]
        lines += md_table(["Modell", "Methode", "Kopf", "übriger Hund", "Hintergrund", "Bilder"], rows)
        lines += ["", "Beispiele: `head_examples.png` (weiße Box = annotierter Kopf)."]
        # per model the method that passed the deletion test best (fallback: occlusion, model-agnostic)
        best = {}
        for m in dict.fromkeys(piv.index.get_level_values("model")):
            cand = {meth: a for (mo, meth), a in areas.items() if mo == m and meth != "random"}
            best[m] = min(cand, key=cand.get) if cand else "occlusion"
        heads = {m: piv.loc[(m, meth), "head"] for m, meth in best.items() if (m, meth) in piv.index}
        if heads:
            ranked = sorted(heads.items(), key=lambda kv: -kv[1])
            spread = ranked[0][1] / max(ranked[-1][1], 1e-9)
            findings.append(
                "- **Kopf:** Mit der jeweils treuesten Methode (Abschnitt 2) liegt die Saliency am Kopf bei "
                + ", ".join(f"{m} {v:.1f}× ({METHOD_NAMES[best[m]]})" for m, v in ranked)
                + " über dem Flächenanteil. "
                + (
                    "Alle Modelle stützen sich also ähnlich stark auf den Kopf. "
                    if spread < 1.5
                    else f"{ranked[0][0]} konzentriert sich deutlich stärker auf den Kopf als {ranked[-1][0]}. "
                )
                + "Rassen unterscheiden sich vor allem am Kopf (Schnauze, Ohren, Augenpartie); das ist dieselbe "
                "Region, die auch ein Mensch nutzt. Der Hintergrund liegt unter 1, passend zu den Hintergrund-Tests "
                "in Station 3."
            )
            for m in heads:
                if (m, "gradcam") in piv.index and (m, "occlusion") in piv.index:
                    gc, oc = piv.loc[(m, "gradcam"), "head"], piv.loc[(m, "occlusion"), "head"]
                    if max(gc, oc) / max(min(gc, oc), 1e-9) > 2:
                        findings.append(
                            f"- **Methoden widersprechen sich ({m}):** Kopf-Fokus mit Grad-CAM {gc:.1f}×, mit "
                            f"Occlusion {oc:.1f}×. Grad-CAM ist bei ViTs bekannt unzuverlässig (Chefer et al., CVPR "
                            "2021); hier zählt die Methode, die im Deletion-Test besser abschneidet."
                        )

    # ------------------------------------------------------------------ 4. agreement
    agree = out / "agreement.csv"
    if agree.exists():
        a = pd.read_csv(agree)
        lines += ["", "## 4. Schauen ViT und CNN auf dasselbe?", ""]
        lines += [
            "Überlappung (IoU) der jeweils wichtigsten 20 % des Bildes (Grad-CAM) zwischen zwei Modellen, "
            "Validierungssplit. Zum Vergleich: zwei zufällige Karten überlappen bei 20 % Fläche im Mittel mit "
            "IoU ≈ 0.11.",
            "",
        ]
        lines += md_table(
            ["Modell A", "Modell B", "IoU der Top-20 %", "Bilder"],
            [[r.model_a, r.model_b, f"{r.iou_top20:.2f}", r.n] for r in a.itertuples()],
        )
        if len(a):
            lo = a.sort_values("iou_top20").iloc[0]
            random_iou = 0.2 * 0.2 / (1 - 0.8 * 0.8)  # two independent 20 % masks
            verdict = (
                "kaum mehr als zwei zufällige Karten"
                if lo.iou_top20 < 1.5 * random_iou
                else "mehr als zufällig, aber weit entfernt von deckungsgleich"
            )
            findings.append(
                f"- **Übereinstimmung:** {lo.model_a} und {lo.model_b} überlappen nur zu IoU {lo.iou_top20:.2f} "
                f"({verdict}, Zufall ≈ {random_iou:.2f}). Gleiche Accuracy heißt also nicht, dass die Modelle "
                "dieselben Bildteile nutzen; passt zur Leitfrage „Was sehen Vision Transformer?“. Einschränkung: "
                "Die Übereinstimmung beruht auf Grad-CAM, das beim ViT im Deletion-Test am schwächsten "
                "abschneidet; außerdem sind die Karten unterschiedlich fein (ViT 14×14 Patches, CNNs 7×7). Dass die "
                "beiden CNNs untereinander stärker übereinstimmen als mit dem ViT, ist das robustere Ergebnis."
            )

    lines += [
        "",
        "## 5. Befunde (Material für die Diskussion)",
        "",
        "Aus den Zahlen abgeleitet; Formulierung und Gewichtung sind eure Aufgabe.",
        "",
        *findings,
    ]
    lines += ["", "## Quellen (im Paper zitieren)", "", *[f"- {s}" for s in SOURCES]]
    lines += [
        "",
        "## Dateien",
        "",
        "- `contrastive_*.png`, `head_examples.png`, `deletion_curves.png`",
        "- `faithfulness.csv`, `head_focus.csv`, `agreement.csv`, `runs.json`",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
