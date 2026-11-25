<div align="center">

# 🐕 Hunderassen-ViT

**Vision Transformer auf 25 Hunderassen: anpassen und erklären**

Projektarbeit im Modul *Deep Learning for Computer Vision* · M.Sc. Informatik · Hochschule Bielefeld (HSBI) · WS 2026/27

[![CI](https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit/actions/workflows/ci.yml/badge.svg)](https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.48%2B-FFD21E)
![Dataset](https://img.shields.io/badge/data-Oxford--IIIT%20Pet%20(25%20Hunderassen)-6A5ACD)
![Lizenz](https://img.shields.io/badge/Lizenz-MIT-blue)
![Code style](https://img.shields.io/badge/code%20style-ruff-D7FF64?logo=ruff&logoColor=black)
![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen?logo=pytest&logoColor=white)
![Seeds](https://img.shields.io/badge/Seeds-3%20je%20Konfiguration-informational)
![Paper](https://img.shields.io/badge/Paper-CVPR%202026%20Template-B31B1B?logo=latex&logoColor=white)
![Git LFS](https://img.shields.io/badge/Abbildungen-Git%20LFS-F64935?logo=git&logoColor=white)

[Highlights](#-highlights) · [Ergebnisse](#-ergebnisse-station-1) · [Schnellstart](#-schnellstart) · [Struktur](#-struktur) · [Dokumentation](#-dokumentation) · [Team](#-team) · [Mitwirken](#-mitwirken) · [Lizenz](#-lizenz) · [Zitieren](#-zitieren)

</div>

---

## 💡 Worum geht es?

Wie effizient lässt sich ein auf ImageNet vortrainierter **Vision Transformer** an eine feinkörnige Aufgabe
mit wenigen Daten anpassen? Wir untersuchen das auf den **25 Hunderassen von
[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)** (≈ 2 250 Trainingsbilder, Segmentierungsmasken).

**Station 1 „Adaptation“** (dieser Stand, Tag `ms1-v2`): Linear Probing · LoRA (Rang 4/8/16, Query/Value vs.
alle Layer) · volles Fine-Tuning des ViT-B/16 im Vergleich mit vortrainierten CNNs (ResNet-50, ConvNeXt-T).
Zusätzlich: Warum liegen alle Modelle über 90 %? (`make why`) und welche Bildteile tragen die Entscheidung?
(`make explain`). Die weiteren Stationen folgen in späteren Ständen dieses Repositories.

Alle Modelle sind vortrainiert, jede Konfiguration läuft mit **3 Seeds**, ausgewählt wird nur auf dem
Validierungssplit, und Unterschiede werden nur interpretiert, wenn sie größer als die Seed-Streuung sind.

---

## ✨ Highlights

| | |
|:--|:--|
| 🔁 **Ein Befehl, alles reproduziert** | `make all` erzeugt Läufe, Reports, Tabellen und Abbildungen; abgebrochene GPU-Sitzungen setzen einfach fort. `make smoke` prüft die ganze Pipeline in Minuten. |
| 📏 **Saubere Statistik** | 3 Seeds je Konfiguration, Modellauswahl nur auf Validation, Test genau einmal. Unterschiede gelten nur, wenn die gepaarte Differenz über zwei Standardfehlern liegt. |
| 🧩 **LoRA selbst implementiert** | Eigene Adapter in [`lora.py`](src/animal_vit/lora.py) (Hu et al. 2022), getestet und nach dem Training in die Gewichte eingerechnet: gleiche Inferenzkosten wie der Basis-ViT. |
| 🔍 **Erklärungen, die geprüft werden** | Grad-CAM, Occlusion und kontrastive Karten („warum A und nicht B?“), bewertet mit dem Deletion-Test statt nur hübscher Heatmaps. |
| 🕵️ **Gegenprobe auf Leakage** | Fast-Duplikat-Suche mit CLIP zwischen Test und Training, Läufe ohne Pretraining und ImageNet-Köpfe ohne Training erklären die > 90 %. |
| 📝 **Reports schreiben sich selbst** | `scripts/report_*.py` erzeugen Tabellen, Abbildungen und Befunde direkt aus den Läufen, dazu die LaTeX-Tabellen fürs Paper. |
| ✅ **Software-Qualität** | 62 Tests auf der CPU, GitHub Actions bei jedem Push, Ruff und Pre-Commit, Docstrings für jede öffentliche Funktion. |

---

## 📊 Ergebnisse (Station 1)

Testsplit (2 486 Bilder), Mittelwert ± Standardabweichung über 3 Seeds.

| Methode | Test-Accuracy | trainierbare Parameter |
|:--|--:|--:|
| ViT-B/16 + LoRA r=8 | **96,21 ± 0,24 %** | 0,31 M |
| ViT-B/16 + LoRA r=4 | 96,11 ± 0,21 % | 0,17 M |
| ViT-B/16 Linear Probing | 96,03 ± 0,12 % | 0,02 M |
| ViT-B/16 volles Fine-Tuning | 95,39 ± 0,39 % | 85,8 M |
| ConvNeXt-T (vortrainiert) | 94,78 ± 0,46 % | 27,8 M |
| ResNet-50 (vortrainiert) | 94,22 ± 0,85 % | 23,6 M |

**Kernaussagen**
- Ein neuer linearer Kopf genügt: Linear Probing ist mit 0,02 % der Parameter so gut wie LoRA, volles
  Fine-Tuning ist leicht schlechter.
- Warum so hoch? 22 der 25 Rassen sind ImageNet-Klassen: Die ImageNet-Köpfe erkennen sie **ohne Training**
  zu 96–97 %, ohne Pretraining schaffen ViT bzw. ResNet nur 11 bzw. 31 %.
- Alle Modelle stützen sich auf den Kopf des Hundes (Saliency etwa dreimal so stark wie nach Flächenanteil).

<p align="center">
  <img src="results/paper/effizienz.png" alt="Test-Accuracy gegen trainierbare Parameter" width="46%">
  &nbsp;
  <img src="results/paper/gradcam.png" alt="Saliency-Karten von ViT, ConvNeXt-T und ResNet-50 mit annotierter Kopf-Box" width="50%">
</p>
<p align="center"><sub>Links: Accuracy gegen trainierbare Parameter (log). Rechts: Worauf die Modelle schauen; weiße Box = annotierter Kopf.</sub></p>

Details mit Tabellen, Abbildungen und geprüften Hypothesen: [Report Station 1](results/ms1/REPORT.md) ·
[Warum > 90 %](results/why/REPORT.md) · [Erklärungen](results/explain/REPORT.md) ·
[Datenqualität](results/data_quality/README.md) · Paper: `make paper`.

---

## 🚀 Schnellstart

```bash
git clone https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit.git && cd hunderassen-vit
make setup        # einmal: .venv, PyTorch (GPU falls vorhanden), Paket, Datensatz, Tests
make smoke        # komplette Pipeline auf einer Mini-Teilmenge (Minuten)
make all          # Station 1 + why + explain + Reports (GPU, etwa eine Stunde)
```

| Ziel | Befehl |
|:--|:--|
| Station 1 | `make station1` |
| Zusatzanalysen | `make why` · `make explain` |
| Paper bauen | `make paper` (latexmk oder tectonic) |
| Qualität | `make test` · `make lint` |
| auf einer Miet-GPU | `bash scripts/runpod.sh`, siehe [RunPod-Anleitung](docs/RUNPOD.md) |
| ein Experiment | `python scripts/train.py configs/ms1/vit_b16_lora_r8.yaml --set training.seed=2` |

Abgebrochene Läufe einfach neu starten: Fertige Läufe werden übersprungen.

---

## 🗂️ Struktur

```
hunderassen-vit/
├── src/animal_vit/   Paket: Daten, Modelle, LoRA, Training, Analysen, Metriken
├── scripts/          CLI: train, report_ms1, why_analysis, explain, benchmark, setup
├── configs/          ein YAML pro Experiment: ms1/ why/ extra/
├── results/          Reports, Abbildungen und Tabellen (Bilder über Git LFS)
├── paper/ms1/        Paper Station 1 (LaTeX, CVPR-Template, Deutsch)
├── tests/            pytest auf der CPU, ohne Downloads
└── docs/             Kontext, Architektur, Technik, Anforderungen, Gliederung, RunPod
```

---

## 📚 Dokumentation

| Dokument | Inhalt |
|:--|:--|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module, Datenfluss, Ablauf, wo man was ändert |
| [CONTEXT.md](docs/CONTEXT.md) | Modul, Vorlesungsinhalte, Forschungsfragen, Hypothesen, Methodik |
| [TECHNICAL.md](docs/TECHNICAL.md) | Installation, Config-Schema, CLI, Troubleshooting |
| [REQUIREMENTS.md](docs/REQUIREMENTS.md) | Anforderung des Moduls → Umsetzung → Ergebnisdatei |
| [milestones/](docs/milestones) | Gliederung und Checkliste der Abgabe |
| [RUNPOD.md](docs/RUNPOD.md) | alle Experimente auf einer gemieteten GPU |
| [AGENTS.md](AGENTS.md) | Regeln für KI-Assistenten (Claude Code, Codex, Copilot, Cursor) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Arbeitsweise, Commits, Code- und Experimentregeln |

---

## 👥 Team

<table align="center">
  <tr>
    <td align="center"><a href="https://github.com/weski17"><img src="https://github.com/weski17.png" width="90" alt=""><br><b>Wael Eskeif</b></a><br><sub>M.Sc. Informatik, HSBI</sub></td>
    <td align="center"><a href="https://github.com/Tareq1337"><img src="https://github.com/Tareq1337.png" width="90" alt=""><br><b>Mohammad Tarek Dawalibi</b></a><br><sub>M.Sc. Informatik, HSBI</sub></td>
  </tr>
</table>

---

## 🤝 Mitwirken

Beiträge folgen den Regeln in [`CONTRIBUTING.md`](CONTRIBUTING.md), kurz:

- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `exp:` …), klein und thematisch.
- Gemeinsamer Code nach `src/animal_vit/`, Experimente als YAML in `configs/`, Skripte bleiben dünne CLI-Hüllen.
- Vor jedem Push: `make test` und `make lint` grün, neue Funktionen mit Docstring und Test.
- Keine Daten, Gewichte oder Zugangsdaten im Repository; Zahlen in Paper und Folien nur aus `results/`.

---

## 📄 Lizenz

- **Code:** [MIT](LICENSE) © 2026 Wael Eskeif, Mohammad Tarek Dawalibi
- **Datensatz:** Oxford-IIIT Pet (Parkhi et al., CVPR 2012), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Der Datensatz ist nicht im Repository und wird per `make data` geladen.
- **Vortrainierte Modelle:** jeweils unter der Lizenz ihres Hugging-Face-Checkpoints.

---

## 📖 Zitieren

Metadaten in [`CITATION.cff`](CITATION.cff) (GitHub zeigt dafür „Cite this repository“ an). BibTeX:

```bibtex
@software{eskeif_dawalibi_2026_hunderassen_vit,
  author  = {Eskeif, Wael and Dawalibi, Mohammad Tarek},
  title   = {Hunderassen-ViT: adapting Vision Transformers to 25 dog breeds},
  year    = {2026},
  version = {ms1-v2},
  url     = {https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit},
  license = {MIT}
}
```

---

## 🙏 Danksagung

Prof. Dr.-Ing. Jan Rexilius für das Modul *Deep Learning for Computer Vision*, der Visual Geometry Group
(Oxford) für Oxford-IIIT Pet und den Projekten hinter **PyTorch**, **Hugging Face Transformers**,
**scikit-learn**, **pytorch-grad-cam** und **Captum**.

<div align="center">
<sub>Hochschule Bielefeld · Campus Minden · Deep Learning for Computer Vision · Prof. Dr.-Ing. Jan Rexilius</sub>
</div>
