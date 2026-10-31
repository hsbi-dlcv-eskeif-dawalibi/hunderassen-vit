# Regeln für dieses Repository

Diese Regeln gelten für alle Beiträge, von Menschen wie von KI-Assistenten.
Ziel: Ein Repository, das jederzeit lauffähig, nachvollziehbar und abgabefertig ist.

---

## 1. Branches

Es wird **direkt auf `main`** committet und gepusht, ohne Pull Requests. `main` soll trotzdem
jederzeit lauffähig sein: vor jedem Push die Checkliste in Abschnitt 8 durchgehen.

Branches sind optional, z. B. `exp/<name>` für kurzlebige Experimente, die nicht nach `main` sollen.

## 2. Commits: Conventional Commits

Format:

```
<typ>(<optionaler bereich>): <kurze beschreibung im imperativ>

<optional: warum und was, in ganzen sätzen>
```

| Typ | Wann |
|---|---|
| `feat` | neue Funktion, neues Experiment, neue Config |
| `fix` | Fehlerbehebung (beschreiben, **was** falsch war) |
| `docs` | nur Dokumentation |
| `refactor` | Codeumbau ohne Verhaltensänderung |
| `test` | Tests hinzufügen oder ändern |
| `style` | Formatierung, keine Logikänderung |
| `chore` | Tooling, Abhängigkeiten, Konfiguration, Aufräumen |
| `exp` | Ergebnisse eines Experiments in `results/` aufnehmen |

Beispiele:

```
feat(attention): add real attention-entropy regularisation
fix(cnn): keep a deep copy of the best weights instead of live references
docs: document dataset imbalance in CONTEXT.md
exp(ms1): add LoRA rank ablation results (3 seeds)
```

Regeln:
- **Ein Commit = eine logische Änderung.** Lieber mehrere kleine Commits.
- Beschreibung auf Englisch, Imperativ, ohne Punkt am Ende, maximal ca. 72 Zeichen.
- Keine Sammel-Commits wie `update`, `fix stuff` oder `wip`.

## 3. Code

- **Gemeinsamer Code gehört nach `src/animal_vit/`**, nicht in Notebooks oder Skripte.
  Wenn Code zum zweiten Mal kopiert wird, gehört er ins Paket.
- **Skripte** in `scripts/` sind dünne CLI-Hüllen: Argumente parsen, Paketfunktion aufrufen.
- **Keine hartcodierten Pfade.** Verwende `animal_vit.paths` (`DATA_DIR`, `RESULTS_DIR`, ...).
- **Keine Magic Numbers in Notebooks.** Experimentparameter gehören in eine YAML-Datei unter `configs/`.
- **Docstrings** für jedes Modul und jede öffentliche Funktion: was, warum, Besonderheiten.
  Kommentare erklären das *Warum*, nicht das *Was*.
- Stil: [Ruff](https://docs.astral.sh/ruff/) mit Zeilenlänge 120, wird per Pre-Commit erzwungen.
- Type Hints für öffentliche Funktionen.
- Sprache im Code (Namen, Docstrings, Kommentare): **Englisch**. Dokumentation in `docs/`: **Deutsch**.

## 4. Experimente

1. **Config anlegen** unter `configs/msN/<name>.yaml` und darin kurz kommentieren, was getestet wird
   und warum.
2. Erst **`make debug`** bzw. `--set data.max_samples=256`, dann den echten Lauf.
3. **Mindestens 3 Seeds** für jede Aussage, die in eine Abgabe kommt. Berichte Mittelwert ± Std.
4. **Modellauswahl nur auf `validation`.** Das Testset wird pro finaler Konfiguration genau einmal
   ausgewertet und **nie** zum Tunen benutzt.
5. Berichte immer **Accuracy und Macro-F1**. Der Testsplit ist unbalanciert.
6. Läufe mit `make analysis` auswerten und mit `make summary` zusammenfassen. `results/summary.md`
   und verwendete Abbildungen (`results/figures/`) committen. **Keine Checkpoints, keine `runs/`.**
7. Zahlen in die Gliederung des Meilensteins (`docs/milestones/`) und in die Folien übertragen.

## 5. Was nicht ins Repository gehört

- Datensätze (`data/`), Modellgewichte (`*.pt`, `*.pth`, `*.safetensors`), `runs/`
- `.ipynb_checkpoints/`, `__pycache__/`, IDE-Ordner
- Zugangsdaten, Tokens, `kaggle.json`
- Notebook-Ausgaben (werden von `nbstripout` entfernt)
- Dateien > 2 MB ohne guten Grund (Abbildungen laufen über Git LFS)

## 6. Notebooks

- Neue Notebooks nur für **Exploration und Visualisierung**. Sie importieren aus `animal_vit` und
  enthalten keine eigene Trainingslogik.
- Benennung: `msN_<thema>.ipynb`.

## 7. Quellen und Redlichkeit

Die Modulregeln (Vorlesung 01, Folie 13) gelten:
- **Keinen Code anderer Gruppen verwenden oder teilen.**
- **Externe Quellen kenntlich machen**, also Paper, Webseiten, Bücher und Code-Snippets im Code
  (Kommentar mit Link) und in Präsentationen.
- Die Git-Historie wird **nicht umgeschrieben** (kein `push --force` auf `main`, kein Entfernen
  fremder Commits). Wer was beigetragen hat, bleibt nachvollziehbar.

## 8. Checkliste vor dem Push auf `main`

- [ ] `make test` ist grün
- [ ] `make lint` ist grün
- [ ] neue Funktionen haben Docstrings
- [ ] neue Experimente haben eine kommentierte Config
- [ ] `docs/CONTEXT.md`, `docs/TECHNICAL.md` bzw. `docs/milestones/` sind aktualisiert, falls nötig
- [ ] keine Daten, Gewichte oder Secrets im Diff
