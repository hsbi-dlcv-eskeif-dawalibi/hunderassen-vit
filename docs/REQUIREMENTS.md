# Anforderungen des Moduls → Umsetzung (Nachweis)

Jede Anforderung aus Vorlesung 01 (Folien 13, 14, 16, 18–20) und den Hinweisen vom 15.10.2026 mit der
Stelle, an der sie umgesetzt ist, und der Datei, in der das Ergebnis steht. Alles wird mit **einem
Befehl** erzeugt: `make all` (bzw. `bash scripts/runpod.sh all`).

Legende: ✅ umgesetzt · ⚠️ umgesetzt, Entscheidung/Begründung im Paper nötig · 👤 nur durch das Team möglich

---

## Station 1 „Adaptation" (30 %)

| Anforderung | Status | Umsetzung | Ergebnis |
|---|---|---|---|
| Geeigneten Datensatz auswählen | ✅ | Oxford-IIIT Pet, 25 Hunderassen (`data.species: dogs`) | `docs/CONTEXT.md` §4 |
| Datenqualität prüfen, Bereinigung, Labeling, Formatierung | ✅ | `make data-report`: Bilder je Rasse, Farbmodi, Größen, exakte Duplikate (auch über Splits und mit unterschiedlichem Label), Anteil des Hundes; RGB-Konvertierung im Loader | `results/data_quality/README.md`, `examples.png` |
| ViT mit Linear Probing (nur Kopf) | ✅ | `configs/ms1/vit_b16_head.yaml` | `results/ms1/REPORT.md` |
| ViT mit LoRA, unterschiedliche Parameter | ✅ | Rang 4 / 8 / 16 und Query/Value vs. alle Layer (`vit_b16_lora_*.yaml`) | `REPORT.md` §2 |
| CNN trainieren | ✅ | ResNet-50, ConvNeXt-T (gleiches Protokoll) | `REPORT.md`, `efficiency.png` |
| Modelle vergleichen | ✅ | Tabelle mit Mittelwert ± Std über 3 Seeds, Accuracy vs. trainierbare Parameter, Lernkurven, Kosten | `tables.tex`, `efficiency.png`, `validation_curves.png` |
| Ergebnisse diskutieren | 👤 | Report liefert Rangfolge, Abstände vs. Seed-Streuung, Fehleranalyse; Diskussion schreibt das Team | `REPORT.md`, `per_breed_f1.png`, `confusions.csv` |
| „Etwas größeres Modell" in Station 1 | ✅ | ViT-B/16 (86 M) | – |
| Augmentierung machen | ✅ | `light` in allen Läufen, Vergleich none / light / strong (`make ms1-augment`) | `REPORT.md` |

## Über die Stationen hinaus: Warum über 90 %?

| Frage | Status | Umsetzung | Ergebnis |
|---|---|---|---|
| Woher kommen die hohen Werte? | ✅ | `make why`: ohne Pretraining, ImageNet-Köpfe ohne Training, Fast-Duplikate, Few-Shot auf eingefrorenen Merkmalen, Rest-Fehler; jede Prüfung mit Vorlesungsbezug (`docs/CONTEXT.md` §6) | `results/why/REPORT.md`, 5 Abbildungen |
| Warum diese Rasse und nicht die andere? | ✅ | `make explain`: kontrastive Grad-CAM-Karten, Occlusion, Attention Rollout, Kopf-Fokus, Deletion-Test, Übereinstimmung ViT/CNN; Bibliotheken pytorch-grad-cam und Captum (zitiert) | `results/explain/REPORT.md` |
| Faire Hyperparameter für die CNNs | ✅ | Lernrate von ResNet-50 und ConvNeXt-T auf Validation gewählt (`make ms1-cnn-lr`) | `results/ms1/lr_sweep_*.md` |

## Abgaben

| Anforderung | Status | Umsetzung |
|---|---|---|
| Station 1: PDF, max. 6 Seiten, CVPR-Template | ✅ / 👤 | Vorlage `paper/ms1/` (Tabellen/Abbildungen automatisch), Text schreibt das Team; `make paper` |
| Git-Link | ✅ | `make tag MS=ms1`, Link steht im Paper |
| Aufbau nach Folie 16 (Abstract 100–150 Wörter, Einleitung mit Beiträgen, ≥ 5 Quellen, Methode, Ergebnisse, Diskussion) | ✅ / 👤 | Abschnitte und Seitenbudget in `paper/ms1/` und `docs/milestones/` |
| Literatur auch zur Art der Bilder, Datensatz als Quelle | ✅ | Oxford-IIIT Pet, Stanford Dogs in `main.bib` |
| PDF auf Deutsch oder Englisch | ✅ | beide Paper auf Deutsch (babel ngerman, Dezimalkomma) |

## Allgemeine Hinweise (15.10.2026)

| Hinweis | Status | Umsetzung |
|---|---|---|
| Quellcode dokumentiert (sonst keine Punkte) | ✅ | Docstrings in jedem Modul/Skript, `docs/TECHNICAL.md`, kommentierte Configs |
| Modelle und Bilder nicht hochladen | ✅ | `.gitignore` (`data/`, `runs/`, Gewichte), `make pack` ohne Gewichte |
| 224 × 224, Standard-ViT von Hugging Face | ✅ | `google/vit-base-patch16-224` |
| Nur eine Tierart | ✅ | nur Hunde |
| 5–10 Klassen empfohlen | ⚠️ | 25 Rassen; Begründung im Paper (ca. 200 Bilder pro Rasse, schwierigere feinkörnige Aufgabe). Falls gewünscht vorab mit dem Dozenten abstimmen |
| Ergebnisse und Diskussion besonders wichtig | ✅ / 👤 | Reports prüfen Hypothesen automatisch gegen die Seed-Streuung; Begründungen schreibt das Team |
| Ressourcen können entzogen werden | ✅ | alle Läufe fortsetzbar (fertige werden übersprungen) |

## Regeln (Folie 13)

| Regel | Status | Hinweis |
|---|---|---|
| Max. 2 Personen | ✅ | Wael Eskeif, Tarek Dawalibi |
| Externe Quellen kenntlich machen | ✅ | Quellen in Docstrings, `docs/CONTEXT.md` §10, `main.bib` |
| Quellen angeben | ✅ | Literaturverzeichnisse beider Paper, Quellen im Code und in `docs/` |
| „Kein Code aus externen Quellen, z. B. generative KI-Werkzeuge" | ⚠️ 👤 | Mit KI-Unterstützung entstandene Commits sind mit `Co-Authored-By` gekennzeichnet. **Vor der Abgabe mit dem Dozenten klären**, ob und wie das zulässig ist |

---

## Was ein Befehl erzeugt (`make all`)

```
results/
├── data_quality/      README.md, examples.png, class_counts.png, per_breed.csv
├── ms1/               REPORT.md, tables.tex, efficiency.png, validation_curves.png, per_breed_f1.png, ...
├── why/               REPORT.md, 5 Abbildungen, CSV-Dateien
├── explain/           REPORT.md, Erklärungskarten, CSV-Dateien
└── summary.md         alle Läufe, Mittelwert ± Std
results_bundle_<Datum>.tar.gz   alles ohne Modellgewichte (Download von RunPod)
```

Danach lokal: `make paper` (LaTeX).
