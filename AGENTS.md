# AGENTS.md: Anleitung für KI-Assistenten

Diese Datei richtet sich an KI-Assistenten (Claude Code, Codex, Copilot, Cursor …), die in diesem
Repository arbeiten. Menschen finden dasselbe ausführlicher in [`README.md`](README.md) und
[`docs/`](docs).

## Projekt in einem Satz

ViTs und CNNs auf den **25 Hunderassen von Oxford-IIIT Pet**. Dieser Stand ist **Meilenstein 1**
(Station 1 „Adaptation“: Linear Probing, LoRA, volles Fine-Tuning vs. vortrainierte CNNs) mit den
Zusatzblöcken `make why` (warum > 90 %?) und `make explain` (Erklärungskarten). Weitere Stationen folgen
in späteren Ständen. Modul *Deep Learning for Computer Vision*, HSBI, WS 2026/27.

## Vor jeder Aufgabe lesen

| Datei | Wozu |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module, Datenfluss, Ablauf, wo man was ändert |
| [`docs/CONTEXT.md`](docs/CONTEXT.md) | Modul, Vorlesungen, Forschungsfragen, Hypothesen, Methodik |
| [`docs/TECHNICAL.md`](docs/TECHNICAL.md) | Setup, Config-Schema, CLI, Troubleshooting |
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Anforderung des Moduls → Umsetzung → Ergebnisdatei (bei Änderungen pflegen) |
| [`docs/milestones/`](docs/milestones) | Gliederungen und Checklisten der Abgaben |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | verbindliche Repo-Regeln |

## Befehle

```bash
make setup                 # einmal: venv, PyTorch, Paket, Datensatz, Tests
make test                  # pytest auf der CPU, ohne Downloads (vor jedem Commit)
make lint                  # ruff check + format --check (vor jedem Commit)
make smoke                 # ganze Pipeline auf Mini-Teilmenge (Minuten)
make all                   # Station 1 + why + explain + Reports (GPU, etwa eine Stunde)
make station1|why|explain
make paper                 # Paper bauen (latexmk oder tectonic)
python scripts/train.py configs/<...>.yaml --set key=value
```

## Arbeitsweise

- Mit dem Team **auf Deutsch** kommunizieren; Code, Docstrings und Commit-Messages **auf Englisch**.
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `style:`, `exp:`),
  klein und thematisch. **Direkt auf `main` committen und pushen, keine Pull Requests.**
- Gemeinsamer Code → `src/animal_vit/`, Experimente → YAML in `configs/`, CLI → `scripts/`.
  Keine Logik in Notebooks. Pfade nur über `animal_vit.paths`.
- Schwere Abhängigkeiten (torch, transformers) **innerhalb** der Funktionen importieren, damit Tests und
  Hilfsmodule leicht bleiben.
- Neue Funktion → Test in `tests/` (ohne Downloads, Mini-Modelle aus `tests/conftest.py`).
- Reports (`scripts/report_*.py`) erzeugen Zahlen **und** Texte. Aussagen nur, wenn der Unterschied die
  Seed-Streuung übersteigt (siehe `report_tools.paired_delta`, `compare_gains`).

## Wissenschaftliche Regeln

- Modellauswahl **nur auf `validation`**, `test` einmal am Ende. Hyperparameter (z. B. CNN-Lernrate)
  ebenfalls auf Validation wählen.
- 3 Seeds, Mittelwert ± Std, Accuracy **und** Macro-F1.
- CNNs immer **vortrainiert** anpassen (Vorgabe des Dozenten); Training ohne Pretraining nur als
  Vergleich (`make why`).
- Zahlen in Paper, Folien und Gliederungen nur aus `results/` übernehmen, nie erfinden.

## Grenzen

- Git-Historie nicht umschreiben, keine Commits oder Autorenangaben entfernen.
- Keine Datensätze, Modellgewichte oder Zugangsdaten committen (`data/`, `runs/` sind ignoriert).
- Externe Quellen im Code, in den Docs und in den Papern angeben (Modulregel).
- GPU-Läufe nicht als erfolgreich melden, ohne die Ausgabe gesehen zu haben.
- Abbildungen (`*.png`, `*.pdf`) liegen in Git LFS; ohne `git lfs` sind es nur Platzhalter-Textdateien.
