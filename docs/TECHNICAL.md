# Technische Dokumentation

Setup, Architektur, Konfiguration und Bedienung des Codes. Den fachlichen Hintergrund
(Forschungsfrage, Experimentplan, Methodik) beschreibt [`CONTEXT.md`](CONTEXT.md).

---

## Inhalt

1. [Voraussetzungen](#1-voraussetzungen)
2. [Installation](#2-installation)
3. [Datensatz](#3-datensatz)
4. [Projektstruktur](#4-projektstruktur)
5. [Architektur des Pakets `animal_vit`](#5-architektur-des-pakets-animal_vit)
6. [Experimente konfigurieren](#6-experimente-konfigurieren)
7. [Training](#7-training)
8. [Analysen](#8-analysen)
9. [Ausgabeformate](#9-ausgabeformate)
10. [Tests und Codequalität](#10-tests-und-codequalität)
11. [Umgebungsvariablen](#11-umgebungsvariablen)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Voraussetzungen

| Komponente | Hinweis |
|---|---|
| Python | ≥ 3.10 |
| GPU | NVIDIA + CUDA empfohlen (ViT-B/16, Batch 32, fp16: ca. 10 GB VRAM). Apple Silicon (MPS) und CPU reichen für `make debug` und die Tests. |
| Git LFS | für die Abbildungen und TensorBoard-Events im Archiv |
| Speicher | ca. 1 GB Datensatz, ca. 350 MB pro ViT-B-Lauf |

## 2. Installation

**RunPod (gemietete GPU):** eigene Schritt-für-Schritt-Anleitung in [`RUNPOD.md`](RUNPOD.md)
(`bash scripts/runpod.sh` startet alle drei Stationen).

**Ein Befehl** (Rechner, AI-Cluster oder Colab):

```bash
git clone https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit.git && cd hunderassen-vit
make setup        # .venv, PyTorch (CUDA wenn GPU da), Projekt, Datensatz, Tests
make station1     # alles für Station 1 (fortsetzbar), siehe Abschnitt 7
```

`scripts/setup.sh` legt `.venv` an, installiert PyTorch passend zur Hardware (NVIDIA → CUDA,
macOS → MPS, sonst CPU), das Projekt mit Dev-Tools, lädt Oxford-IIIT Pet und führt die Tests aus.
Das Makefile nutzt danach automatisch `.venv/bin/python`.
Ohne virtuelle Umgebung (Colab, conda, Cluster-Modul): `NO_VENV=1 make setup`.

**Colab:**

```bash
!git clone https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit.git
%cd hunderassen-vit
!NO_VENV=1 make setup
!make station1 SEEDS="1 2 3"
```

Colab kann die GPU jederzeit entziehen: `make station1` einfach erneut starten, fertige Läufe
(`runs/<name>/metrics.json` vorhanden) werden übersprungen. `runs/` am besten auf Google Drive legen
(`ANIMAL_RUNS_DIR=/content/drive/MyDrive/hunderassen_runs`).

**Manuell:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # ggf. passendes CUDA-Wheel, https://pytorch.org
pip install -e ".[dev]"
make data
```

Hinweis: `transformers` ist auf < 5 begrenzt (Version 5 entfernt u. a. `prune_heads`).

## 3. Datensatz

**Oxford-IIIT Pet** wird über torchvision geladen, mit Klassenlabels und Trimaps:

```bash
make data            # = python scripts/prepare_data.py  →  data/oxford_pets/
```

| Split | Herkunft | Größe |
|---|---|---|
| `train` | offizielles `trainval` minus Validation | ca. 2 250 |
| `validation` | 10 % je Klasse aus `trainval` (stratifiziert, `data.split_seed`) | ca. 250 |
| `test` | offizielles `test` | ca. 2 490 |

Das Projekt nutzt nur die **25 Hunderassen** (`data.species: dogs`, Standard). Die Labels werden
auf 0–24 neu nummeriert. Cue-Conflict-Bilder müssen mit derselben Auswahl erzeugt werden
(`make_cue_conflict.py --species dogs`), weil sie Klassennummern speichern.

Der Validierungssplit hängt **nur** von `data.split_seed` ab, nicht vom Trainings-Seed. Alle Läufe
werden also auf denselben Bildern ausgewählt.

Für Tests ohne Download gibt es den generischen Ordner-Datensatz `data.name: image_folder`
(`<dir>/{train,validation,test}/<klasse>/`, Pfad über `IMAGE_FOLDER_DIR`).

## 4. Projektstruktur

```
hunderassen-vit/
├── src/animal_vit/          # Paket (siehe Abschnitt 5)
├── scripts/                 # CLI: train, prepare_data, evaluate_variants, make_cue_conflict,
│                            #      evaluate_shape_bias, saliency_maps, benchmark, summarize,
│                            #      data_report, report_ms1, setup.sh
├── configs/                 # debug.yaml, ms1/, why/, extra/
├── docs/
│   ├── CONTEXT.md           # Projektkontext
│   ├── TECHNICAL.md         # dieses Dokument
│   ├── milestones/          # Gliederung und Checkliste der Abgabe (MS1)
├── results/                 # committete Ergebnisse (summary.md, Abbildungen)
├── tests/                   # pytest (Unit + End-to-End mit Mini-Modellen)
├── paper/ms1/               # MS1-Paper im CVPR-2026-Template (LaTeX), `make paper`
├── data/      (ignoriert)   # Datensätze, Cue-Conflict-Bilder
└── runs/      (ignoriert)   # Trainingsläufe
```

## 5. Architektur des Pakets `animal_vit`

```
configs/*.yaml ─► config.ExperimentConfig ─► training.run_experiment
                                                  │
     ┌──────────────┬──────────────┬──────────────┼───────────────┬─────────────────┐
     ▼              ▼              ▼              ▼               ▼                 ▼
  data.*        models.*      components.*    attention.*   ExperimentTrainer   export_evaluation
  (Pets+Masken, (Auto*-Klassen (Pos-Embedding,  (Temperatur,  (CE + Distillation  (metrics.json,
   Splits,       für CNN/ViT/   Conv-Stem)       Pruning,      + Entropie-Terme)   report, CM, model/)
   Transforms)   Swin)                           Entropie)

 runs/<name>/ ─► training.load_run ─► variants · style_transfer · saliency · efficiency ─► summarize
```

| Modul | Verantwortung |
|---|---|
| `paths.py` | Repo-, Daten- und Run-Pfade, Umgebungsvariablen |
| `config.py` | Dataclasses: `model`, `modifications`, `distillation`, `data`, `training`; YAML + `--set`; Tippfehlerschutz |
| `data.py` | Oxford-IIIT Pet / generischer Ordner-Datensatz (Tests), stratifizierter Split, Transforms (`none`/`light`/`strong`), Masken, `collate_fn` |
| `models.py` | Laden beliebiger HF-Bildklassifikatoren, `model_family`, `set_trainable` |
| `lora.py` | **Station 1:** LoRA-Adapter (`apply_lora`), Mergen nach dem Training (`merge_lora`) |
| `reporting.py` | ausführliches Protokoll je Lauf (`README.md`, Lernkurven, Vorhersagen, Datenzahlen) |
| `data_quality.py` | Datenqualitäts-Check (Farbmodi, Größen, Duplikate, Anteil des Tiers) für `scripts/data_report.py` |
| `components.py` | **MS2:** Positions-Embeddings (`learned`/`none`/`shuffled`), Patch Embedding (`linear`/`conv_stem`) |
| `attention.py` | Attention-Temperatur (über Query-Skalierung), Head Pruning, Attention-Entropie |
| `training.py` | `ExperimentTrainer` (inkl. **Distillation**), `run_experiment`, `load_run` |
| `variants.py` | Testbild-Varianten: foreground, background, mixed_random, grayscale, patch_shuffle |
| `style_transfer.py` | Gatys-Style-Transfer (VGG-19) für Cue-Conflict-Bilder, `shape_bias` |
| `saliency.py` | Attention Rollout, Grad-CAM, `foreground_focus` |
| `efficiency.py` | Parameter, FLOPs (`torch.utils.flop_counter`), Latenz |
| `evaluation.py` | Geräteauswahl, Batch-Inferenz, Testsplit eines Laufs |
| `clip_zeroshot.py` | **Station 3:** CLIP Zero-Shot, Prompt-Sets, Linear Probe, Few-Shot |
| `why.py`, `explain.py` | Zusatzblöcke: ImageNet-Abgleich, Fast-Duplikate, Erklärungsmethoden |
| `report_tools.py` | gemeinsame Report-Helfer (gepaarte Differenzen, Tabellen, Abbildungen) |
| `metrics.py` | Accuracy, Macro/Weighted P/R/F1, Cross Entropy, Prediction Entropy, ECE |
| `plotting.py` | Confusion Matrix |

**Designentscheidungen**
- *Eine Pipeline für alle Modelle:* CNNs (ResNet, ConvNeXt) und Transformer (ViT, DeiT, Swin) laufen
  über `AutoModelForImageClassification` und denselben `Trainer`, mit gleichem Protokoll für alle.
- *Läufe sind wiederherstellbar:* `load_run` baut die Architektur inklusive aller Eingriffe aus
  `config.json` neu und lädt die Gewichte. Die Analysen brauchen also nur den Run-Ordner.
- *Lazy Imports:* Configs, Metriken und Varianten sind ohne GPU-Stack testbar.
- *Testset genau einmal:* Auswahl auf `validation`, `test` nach dem Training.

## 6. Experimente konfigurieren

```yaml
name: my_experiment               # Ordner unter runs/
output_dir: null                  # Standard: runs/<name>

model:
  checkpoint: google/vit-base-patch16-224   # jeder HF-Bildklassifikator oder lokaler Pfad
  attn_implementation: null       # "eager" für Attention-Gewichte
  hidden_dropout_prob: null

modifications:                    # MS2-Eingriffe (ViT/DeiT)
  positional_embedding: learned   # learned | none | shuffled
  patch_embedding: linear         # linear | conv_stem
  attention_temperature: 1.0
  prune_heads_random: 0
  trainable: all                  # all | head | lora | last_n:<k>
  lora:                           # nur bei trainable: lora (Station 1)
    rank: 8                       # Rang r von B·A
    alpha: 16.0                   # Skalierung alpha/r
    dropout: 0.1
    targets: qv                   # qv | all | Liste von Modulnamen-Endungen

distillation:                     # MS3
  teacher: null                   # z. B. runs/ms1_convnext_t_s1
  alpha: 0.5                      # L = alpha*CE + (1-alpha)*T^2*KL
  temperature: 4.0

data:
  name: oxford_pets               # oxford_pets | image_folder (Tests)
  species: dogs                   # dogs (25 Rassen, Projekt) | cats (12) | all (37)
  augmentation: light             # none | light | strong
  val_fraction: 0.1
  split_seed: 0
  max_samples: null               # Debug-Teilmenge
  download: true

training:
  strategy: cosine                # fixed | linear | cosine | constant
  epochs: 10
  learning_rate: 5.0e-5           # mit Punkt schreiben (YAML)
  batch_size: 32
  seed: 123
  early_stopping_patience: 3
  metric_for_best_model: accuracy
  fp16: true
  extra: {weight_decay: 0.05, warmup_ratio: 0.1}   # weitere TrainingArguments
```

Overrides: `python scripts/train.py configs/ms1/vit_b16.yaml --set training.seed=2 name=ms1_vit_b16_s2`

## 7. Training

```bash
make debug                  # Smoke-Test (64 Bilder/Split, 1 Epoche)
make station1               # Station 1 komplett: data, data-report, ms1, ms1-augment, ms1-benchmark,
                            #   report-ms1, summary, paper (fertige Läufe werden übersprungen)
make ms1                    # ViT (Linear Probing, LoRA r4/r8/r16/all, voll), ResNet-50, ConvNeXt-T × 3 Seeds
make ms1-vit                # nur die ViT-Anpassungsmethoden
make ms1-augment            # Augmentierung none / strong auf LoRA r=8 (light = Hauptlauf)
make report-ms1             # results/ms1/REPORT.md, tables.tex, Abbildungen
make why                    # warum > 90 %?: ohne Pretraining, ImageNet-Köpfe, Fast-Duplikate, Few-Shot, Rest-Fehler
make explain                # Erklärungskarten: Grad-CAM, Occlusion, kontrastive Karten, Kopf-Fokus, Deletion-Test
make all                    # alles von Meilenstein 1: Station 1, best-models, why, explain, summary, Archiv
make smoke                  # alles auf einer Mini-Teilmenge (1 Seed, 1 Epoche) -> runs_smoke/ (Funktionstest)
make paper                  # Paper Station 1 (LaTeX, CVPR-Template; latexmk oder tectonic)
make tag MS=ms1             # Abgabestand taggen (Git-Link im Paper)
make extra                  # optional: Positions-Embeddings, Conv-Stem, Swin-T (configs/extra)
make ms1 SEEDS="1"          # nur ein Seed
tensorboard --logdir runs   # Live-Kurven
```

## 8. Analysen

| Befehl | Ergebnis (pro Lauf) |
|---|---|
| `python scripts/evaluate_variants.py runs/ms1_*` | `variants.csv`: Accuracy je Variante |
| `make cue-conflict` (einmal) | `data/cue_conflict/*.png` + `index.csv` |
| `python scripts/evaluate_shape_bias.py runs/ms1_*` | `shape_bias.json` |
| `python scripts/saliency_maps.py runs/ms1_vit_b16_s1 --method rollout gradcam` | `saliency.json`, `saliency_<method>.png` |
| `python scripts/benchmark.py runs/*_s1` | `efficiency.json` |
| `make analysis` | alles oben für alle Läufe |
| `make summary` | `results/summary.{csv,md}`: Mittelwert ± Std je Konfiguration  |
| `make pack` | `results_bundle_<Datum>.tar.gz`: alle Ergebnisse ohne Modellgewichte (Download von RunPod) |

## 9. Ausgabeformate

`runs/<name>/` enthält:

| Datei | Inhalt |
|---|---|
| `config.json`, `class_names.json` | vollständige Konfiguration und Klassen (für `load_run`) |
| `model/` | Gewichte (`model.safetensors`) + Processor |
| `metrics.json` | Testmetriken (`accuracy`, `f1_macro`, `ece`, …), beste Validierungsmetrik, Parameter |
| `classification_report.csv`, `confusion_matrix.png` | pro Klasse |
| `README.md` | Ein-Seiten-Zusammenfassung des Laufs (Setup, Kosten, beste Epoche, Testmetriken, Epochen-Tabelle) |
| `learning_curves.png`, `history.json` | Loss und Validation-Accuracy je Epoche, kompletter Trainings-Log |
| `predictions.csv` | Testvorhersage je Bild (wahr, vorhergesagt, Konfidenz) |
| `data_stats.json`, `environment.json` | Bilder je Split/Klasse; GPU und Bibliotheksversionen |
| `tensorboard/`, `checkpoints/` | Trainingsverlauf, letzter/bester Checkpoint |
| `variants.csv`, `shape_bias.json`, `saliency.json`, `efficiency.json` | Analysen |

## 10. Tests und Codequalität

```bash
make test     # 46 Tests, CPU, wenige Sekunden, ohne Downloads
make lint     # ruff check + format --check
```

`tests/test_pipeline.py` trainiert Mini-ViTs auf einem künstlichen Datensatz und deckt die echten
Codepfade ab: Training, Speichern/Laden, Distillation, Conv-Stem, Positions-Embeddings, Temperatur,
Saliency, Varianten, FLOPs. Pre-Commit prüft Ruff, YAML, große Dateien und leert Notebook-Ausgaben.

## 11. Umgebungsvariablen

| Variable | Standard | Zweck |
|---|---|---|
| `DATA_ROOT` | `data/` | Wurzel aller Datensätze (`oxford_pets/`, `cue_conflict/`) |
| `IMAGE_FOLDER_DIR` | `data/image_folder` | generischer Ordner-Datensatz (Tests) |
| `ANIMAL_RUNS_DIR` | `runs/` | Ausgabe der Läufe |
| `HF_HOME` | `~/.cache/huggingface` | Cache der vortrainierten Modelle |
| `ANIMAL_NUM_WORKERS` | `4` | DataLoader-Worker (0 bei Problemen auf macOS, 8 auf RunPod) |

## 12. Troubleshooting

| Problem | Lösung |
|---|---|
| `git-lfs: command not found` | `brew install git-lfs && git lfs install` |
| `CUDA out of memory` | `--set training.batch_size=16` |
| fp16-Fehler auf Mac/CPU | `--set training.fp16=false` |
| `view size is not compatible …` auf Apple-GPU | PyTorch-MPS-Bug im Backward-Pass (volles ViT-Fine-Tuning, Head Pruning). Auf NVIDIA/RunPod tritt er nicht auf. Auf dem Mac auf der CPU trainieren: `--set training.extra.use_cpu=true` (`make smoke` macht das automatisch) |
| DataLoader hängt auf macOS | `--set training.extra.dataloader_num_workers=0` |
| `NotImplementedError: component swaps support ViT/DeiT-style models only` | Positions-/Patch-Eingriffe nur für ViT/DeiT |
| Teacher-Fehler „different classes" | Teacher und Student müssen auf demselben Datensatz trainiert sein |
| `KeyError: Unknown config key(s)` | Tippfehler in der YAML |
