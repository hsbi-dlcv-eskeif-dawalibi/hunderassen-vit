# Training auf RunPod

Schritt-für-Schritt-Anleitung, um alle Experimente (Station 1–3) auf einer gemieteten GPU bei
[RunPod](https://console.runpod.io) laufen zu lassen. Die Ergebnisse holt ihr danach als kleines
Archiv auf den eigenen Rechner.

> **Kurzfassung:** Pod mit RTX 4090 und 60-GB-Volume starten → Repo nach `/workspace/hunderassen-vit` klonen →
> `bash scripts/runpod.sh` → nach einigen Stunden `results_bundle_*.tar.gz` herunterladen → **Pod stoppen**.

---

## 1. Pod erstellen

In der Konsole unter **Pods → Deploy**:

| Einstellung | Empfehlung | Warum |
|---|---|---|
| **GPU** | **RTX 4090 (24 GB)** | genug Speicher für ViT-B/16 und CLIP ViT-B/16, günstig. Alternativen: RTX A5000/A6000, L40S |
| Cloud | Community Cloud (günstiger) oder Secure Cloud | beide funktionieren |
| **Template** | **RunPod PyTorch 2.4** (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`) oder neuer | PyTorch mit CUDA ist schon installiert |
| Container Disk | 20 GB | System, Pakete |
| **Volume Disk** | **60 GB**, eingehängt unter `/workspace` | Datensatz, Modelle, Läufe. Bleibt beim Stoppen erhalten |
| Ports | Jupyter (8888) eingeschaltet lassen | Dateien bequem herunterladen |

Auf **Deploy** klicken und warten, bis der Pod „Running" zeigt. Dann **Connect → Start Web Terminal**
(oder JupyterLab → Terminal).

## 2. Code auf den Pod holen

Das Repository ist **privat**, deshalb braucht der Pod einmal Zugriff. **Gebt euren Token nicht in einem
Chat ein**, sondern nur direkt im Terminal des Pods.

**Variante A: GitHub-Token (empfohlen)**
1. Auf GitHub: *Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token*. Repository access nur `hsbi-dlcv-eskeif-dawalibi/hunderassen-vit`, Berechtigung *Contents: Read-only*,
   Ablauf z. B. 7 Tage.
2. Im Pod-Terminal:
   ```bash
   cd /workspace
   git clone https://github.com/hsbi-dlcv-eskeif-dawalibi/hunderassen-vit.git
   ```
   Benutzername: euer GitHub-Name, Passwort: der Token.

**Variante B: GitHub CLI** (Login per Code im Browser, kein Token kopieren)
```bash
(type -p gh >/dev/null || (apt-get update && apt-get install -y gh)) && gh auth login
cd /workspace && gh repo clone hsbi-dlcv-eskeif-dawalibi/hunderassen-vit
```

> Die Abbildungen im Archiv liegen in Git LFS und werden für das Training nicht gebraucht.
> Ein `git lfs pull` ist auf dem Pod nicht nötig.

## 3. Alles starten

```bash
cd /workspace/hunderassen-vit
bash scripts/runpod.sh
```

Das Skript
1. installiert beim ersten Mal das Projekt (nutzt das PyTorch des Templates), lädt Oxford-IIIT Pet
   und lässt die Tests laufen (`scripts/setup.sh`),
2. startet `make all` in einer **tmux-Sitzung**. Das Training läuft also weiter, wenn ihr den
   Browser schließt,
3. schreibt ein Log nach `logs/`.

**Ganz zuerst: Funktionstest** (ca. 15–30 min, prüft die komplette Kette auf einer winzigen Teilmenge,
Ergebnisse in `runs_smoke/`):
```bash
bash scripts/runpod.sh smoke
```

**Dann ein Probelauf mit einem Seed** (empfohlen, ca. 2–4 h):
```bash
bash scripts/runpod.sh all SEEDS=1
```
Danach dasselbe ohne `SEEDS=1`: Fertige Läufe werden übersprungen, es kommen nur Seed 2 und 3 dazu.

**Einzelne Stationen:**
```bash
bash scripts/runpod.sh station1
```

## 4. Fortschritt beobachten

```bash
tmux attach -t train          # Live-Ausgabe; verlassen mit Ctrl+b, dann d
tail -f logs/*.log            # alternativ das Log
watch -n 5 nvidia-smi         # GPU-Auslastung
ls runs/                      # fertige Läufe (je Lauf ein Ordner mit metrics.json)
```

## 5. Was `make all` macht

| Schritt | Make-Ziel | Läufe | Ergebnis |
|---|---|---|---|
| Datensatz + Datenqualität | `data`, `data-report` | – | `results/data_quality/` |
| **Station 1 „Adaptation"** | `ms1`, `ms1-augment`, `ms1-benchmark`, `report-ms1` | ViT-B/16: Linear Probing, LoRA r4/r8/r16/r8-all, volles Fine-Tuning; ResNet-50; ConvNeXt-T; Augmentierung none/strong (je 3 Seeds) | `results/ms1/REPORT.md`, Tabellen, Abbildungen |
| **Warum über 90 %?** | `why` | ViT-B/16 und ResNet-50 ohne Pretraining (je 3 Seeds, 30 Epochen); ImageNet-Köpfe ohne Training; Fast-Duplikate; Few-Shot auf eingefrorenen Merkmalen; Rest-Fehler | `results/why/REPORT.md` + 5 Abbildungen |
| **Warum diese Rasse?** | `explain` | Grad-CAM, Occlusion, kontrastive Karten, Kopf-Fokus, Deletion-Test für die Teacher und das zweite CNN (Seed 1) | `results/explain/REPORT.md` + Abbildungen |
| Zusammenfassung | `summary` | alle | `results/summary.md` (Mittelwert ± Std je Konfiguration) |
| Archiv | `pack` | – | `results_bundle_<Datum>.tar.gz` (ohne Modellgewichte) |

## 6. Dauer und Kosten (Schätzung)

| Teil | Läufe | RTX 4090 |
|---|---|---|
| Station 1 | 30 Trainingsläufe | ca. 1 h (RTX 5090: ca. 30 min) |
| `why` und `explain` | 6 Läufe ohne Pretraining + Analysen | ca. 20–40 min |
| **Gesamt** | | **ca. 1,5–3 h** (mit `SEEDS=1` etwa ein Drittel) |

Bei etwa 0,6–0,8 US-$ pro Stunde für eine RTX 4090 kostet das insgesamt rund **1–3 US-$**, dazu kommt
eine kleine Gebühr für das Volume, solange es existiert. Die aktuellen Preise zeigt die Konsole.

## 7. Ergebnisse herunterladen

Am Ende von `make all` liegt im Repo-Ordner `results_bundle_<Datum>.tar.gz` (Metriken, Reports,
Abbildungen, Vorhersagen, **keine** Modellgewichte, typischerweise < 100 MB). Neu erzeugen mit `make pack`.

- **JupyterLab:** Dateibrowser → `hunderassen-vit/results_bundle_….tar.gz` → Rechtsklick → *Download*
- **runpodctl** (auf dem Pod vorinstalliert):
  ```bash
  runpodctl send results_bundle_*.tar.gz      # zeigt einen Code an
  ```
  Auf dem Mac (`brew install runpod/runpodctl/runpodctl`):
  ```bash
  runpodctl receive <code>
  ```

Lokal im Repository entpacken:
```bash
tar -xzf results_bundle_*.tar.gz
```
`runs/` bleibt lokal (ist in `.gitignore`), committet werden nur `results/` und die Abbildungen, die ins
Paper kommen.

## 8. Pod stoppen (wichtig!)

- **Stop:** GPU wird freigegeben, keine GPU-Kosten mehr. Das Volume (`/workspace`) bleibt erhalten,
  ihr könnt später weitermachen (`bash scripts/runpod.sh` überspringt fertige Läufe).
- **Terminate:** löscht den Pod **und das Volume**. Erst machen, wenn das Ergebnis-Archiv
  heruntergeladen ist.

Die trainierten Modelle liegen in `runs/` auf dem Volume. Das Volume erst löschen, wenn alles fertig ist.

## 9. Probleme

| Problem | Lösung |
|---|---|
| Abbildungen in `results/` sind kleine Textdateien (`version https://git-lfs…`) | Git LFS fehlt auf dem Pod: `apt-get install -y git-lfs && git lfs install && git lfs pull`. Sonst landen Platzhalter statt PNGs im Ergebnis-Archiv |
| `CUDA out of memory` | `bash scripts/runpod.sh all SEEDS=1` mit kleinerer Batch Size: in der Config `training.batch_size: 16` |
| `DataLoader worker … killed` / `Bus error` (zu wenig Shared Memory) | `ANIMAL_NUM_WORKERS=2 bash scripts/runpod.sh` |
| Download der Modelle langsam / Rate Limit | einmalig; sie liegen danach in `/workspace/.cache/huggingface` |
| tmux-Sitzung „train" läuft schon | `tmux attach -t train` oder beenden: `tmux kill-session -t train` |
| Pod wurde unterbrochen (Community Cloud) | Pod neu starten, `cd /workspace/hunderassen-vit && bash scripts/runpod.sh` |
| `git pull` für neuen Code | `cd /workspace/hunderassen-vit && git pull` (Token/gh wie in Schritt 2) |
