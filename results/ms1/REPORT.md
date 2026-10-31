# Station 1 - Ergebnisbericht

Automatisch erzeugt von `scripts/report_ms1.py`. Alle Werte: Testsplit, Mittelwert ± Standardabweichung
über die Seeds. Auswahl des besten Checkpoints und der besten Methode nur auf Validation.

## 1. Anpassungsmethoden (Kern von Station 1)

| Methode | Seeds | Test-Acc (%) | Macro-F1 (%) | Val-Acc (%) | Trainierbare Parameter | Trainingszeit (min) | Beste Epoche | ECE |
|---|---|---|---|---|---|---|---|---|
| ViT linear probing | 3 | 96.03 ± 0.12 | 95.92 ± 0.13 | 97.07 ± 0.23 | 0.019 M (0.02 %) | 0.3 ± 0.1 | 3.7 ± 1.2 | 0.012 ± 0.005 |
| ViT LoRA r=4 | 3 | 96.11 ± 0.21 | 96.00 ± 0.21 | 97.87 ± 0.83 | 0.167 M (0.19 %) | 0.5 ± 0.1 | 6.7 ± 2.1 | 0.011 ± 0.002 |
| ViT LoRA r=8 | 3 | 96.21 ± 0.24 | 96.12 ± 0.25 | 97.73 ± 0.83 | 0.314 M (0.36 %) | 0.5 ± 0.2 | 5.3 ± 2.9 | 0.017 ± 0.002 |
| ViT LoRA r=16 | 3 | 95.96 ± 0.15 | 95.86 ± 0.16 | 97.73 ± 0.46 | 0.609 M (0.70 %) | 0.6 ± 0.1 | 6.7 ± 1.5 | 0.019 ± 0.004 |
| ViT LoRA r=8 (all layers) | 3 | 95.82 ± 0.18 | 95.69 ± 0.19 | 97.87 ± 0.83 | 1.346 M (1.54 %) | 0.6 ± 0.2 | 5.7 ± 3.1 | 0.019 ± 0.008 |
| ViT full fine-tuning | 3 | 95.39 ± 0.39 | 95.24 ± 0.42 | 96.80 ± 0.40 | 85.818 M (100.00 %) | 0.7 ± 0.1 | 6.3 ± 1.2 | 0.014 ± 0.001 |
| ResNet-50 (CNN) | 3 | 94.22 ± 0.85 | 93.92 ± 0.99 | 97.47 ± 0.46 | 23.559 M (100.00 %) | 0.5 ± 0.1 | 5.0 ± 1.0 | 0.032 ± 0.011 |
| ConvNeXt-T (CNN) | 3 | 94.78 ± 0.46 | 94.63 ± 0.46 | 97.47 ± 0.46 | 27.839 M (100.00 %) | 1.0 ± 0.1 | 6.0 ± 1.0 | 0.017 ± 0.003 |

**Rangfolge nach Test-Accuracy:** ViT LoRA r=8 > ViT LoRA r=4 > ViT linear probing > ViT LoRA r=16 > ViT LoRA r=8 (all layers) > ViT full fine-tuning > ConvNeXt-T (CNN) > ResNet-50 (CNN)

- Beste Methode: **ViT LoRA r=8** (96.21 ± 0.24 %), schlechteste: **ResNet-50 (CNN)** (94.22 ± 0.85 %).
- Auf Validation beste Methode (für Station 2 als Teacher): **ViT LoRA r=4**.
- Abstände zur besten ViT-Methode im Vergleich zur Seed-Streuung (nur größere Abstände interpretieren):
  - ViT linear probing: −0.17 Pp. (Streuung ±0.24) → innerhalb der Streuung
  - ViT LoRA r=4: −0.09 Pp. (Streuung ±0.24) → innerhalb der Streuung
  - ViT LoRA r=16: −0.24 Pp. (Streuung ±0.24) → innerhalb der Streuung
  - ViT LoRA r=8 (all layers): −0.39 Pp. (Streuung ±0.24) → innerhalb der Streuung
  - ViT full fine-tuning: −0.82 Pp. (Streuung ±0.39) → größer als die Streuung

## 2. LoRA mit unterschiedlichen Parametern

| Variante | Rang | alpha | Ziel-Layer | Test-Acc (%) | Trainierbar |
|---|---|---|---|---|---|
| ViT LoRA r=4 | 4 | 8 | qv | 96.11 ± 0.21 | 0.167 M (0.19 %) |
| ViT LoRA r=8 | 8 | 16 | qv | 96.21 ± 0.24 | 0.314 M (0.36 %) |
| ViT LoRA r=16 | 16 | 32 | qv | 95.96 ± 0.15 | 0.609 M (0.70 %) |
| ViT LoRA r=8 (all layers) | 8 | 16 | all | 95.82 ± 0.18 | 1.346 M (1.54 %) |

## 3. Augmentierung

| Lauf | Test-Acc (%) | Macro-F1 (%) | Val-Acc (%) |
|---|---|---|---|
| ViT LoRA r=8 [aug light] | 96.21 ± 0.24 | 96.12 ± 0.25 | 97.73 ± 0.83 |
| ViT LoRA r=8 [aug none] | 95.99 ± 0.16 | 95.88 ± 0.18 | 97.07 ± 0.61 |
| ViT LoRA r=8 [aug strong] | 95.91 ± 0.36 | 95.79 ± 0.36 | 96.80 ± 1.06 |

## 4. Kosten zur Inferenz

| Methode | GFLOPs | Latenz (ms, Batch 1) |
|---|---|---|
| ViT linear probing | 35.13 ± 0.00 | 3.7 ± 0.0 |
| ViT LoRA r=4 | 35.13 ± 0.00 | 3.4 ± 0.0 |
| ViT LoRA r=8 | 35.13 ± 0.00 | 3.6 ± 0.0 |
| ViT LoRA r=16 | 35.13 ± 0.00 | 3.6 ± 0.0 |
| ViT LoRA r=8 (all layers) | 35.13 ± 0.00 | 4.6 ± 0.0 |
| ViT full fine-tuning | 35.13 ± 0.00 | 3.6 ± 0.0 |
| ResNet-50 (CNN) | 8.17 ± 0.00 | 3.4 ± 0.0 |
| ConvNeXt-T (CNN) | 8.91 ± 0.00 | 2.5 ± 0.0 |

LoRA wird nach dem Training in die Gewichte gemergt: gleiche Inferenzkosten wie der Basis-ViT.

## 5. Fehleranalyse: ViT LoRA r=4

Schwierigste Rassen (F1): American Pit Bull Terrier 77.2 %, Staffordshire Bull Terrier 78.2 %, American Bulldog 88.7 %, Boxer 93.0 %, Miniature Pinscher 94.6 %

Leichteste Rassen (F1): Scottish Terrier 100.0 %, Leonberger 99.8 %, English Setter 99.8 %, Newfoundland 99.7 %, Keeshond 99.3 %

Häufigste Verwechslungen (über alle Seeds):

| Wahr | Vorhergesagt | Anzahl |
|---|---|---|
| American Pit Bull Terrier | Staffordshire Bull Terrier | 62 |
| Staffordshire Bull Terrier | American Pit Bull Terrier | 27 |
| American Pit Bull Terrier | American Bulldog | 24 |
| Staffordshire Bull Terrier | American Bulldog | 16 |
| Miniature Pinscher | Chihuahua | 16 |
| Basset Hound | Beagle | 15 |
| Boxer | American Bulldog | 15 |
| Wheaten Terrier | Havanese | 12 |
| Chihuahua | Miniature Pinscher | 10 |
| American Bulldog | Boxer | 10 |

## 5b. Befunde und Erklärungsansätze (Material für die Diskussion)

Aus den Ergebnissen abgeleitet: Befund, möglicher Grund mit Quelle, was im Paper zu begründen ist. Formulierung und Abwägung sind eure Aufgabe.

- **LoRA vs. Linear Probing:** ViT LoRA r=8 gegenüber Linear Probing +0.17 pp (innerhalb der Seed-Streuung). Erklärungsansatz: Linear Probing kann nur die eingefrorenen ImageNet-Merkmale neu gewichten; LoRA passt zusätzlich die Attention an und kann feine Rassenunterschiede herausarbeiten (Hu et al. 2022). Ist der Abstand klein, sind die ImageNet-Merkmale für Hunde schon gut (ImageNet enthält ca. 120 Hunderassen; vgl. Kornblith et al. 2019).
- **LoRA vs. volles Fine-Tuning:** +0.82 pp (größer als die Seed-Streuung) bei 0.37 % der trainierbaren Parameter. Erklärungsansatz: Die nötige Anpassung hat einen niedrigen Rang (Hu et al. 2022); volles Fine-Tuning kann auf ca. 2 250 Bildern eher überanpassen. Lernkurven prüfen (validation_curves.png).
- **LoRA-Rang:** r=4: 96.1 % → r=8: 96.2 % → r=16: 96.0 %. Steigt die Accuracy kaum, reicht ein kleiner Rang (abnehmender Ertrag, Hypothese A3); entscheidend ist dann, *welche* Layer angepasst werden.
- **Ziel-Layer:** alle Linear-Layer vs. nur Query/Value (r = 8): -0.39 pp (innerhalb der Seed-Streuung). Hu et al. (2022) passen nur Query/Value an; mehr Layer geben mehr Freiheitsgrade, aber auch mehr Parameter.
- **ViT vs. CNN:** bester ViT (ViT LoRA r=8) gegenüber bestem CNN (ConvNeXt-T (CNN)) +1.42 pp (größer als die Seed-Streuung). Einordnen: unterschiedliches Pretraining (ViT ImageNet-21k, CNNs ImageNet-1k), induktive Verzerrung der CNNs hilft bei wenig Daten (Vorlesung 03), ConvNeXt übernimmt Designentscheidungen der Transformer (Liu et al. 2022).
- **Augmentierung:** Reihenfolge light (96.2 %) > none (96.0 %) > strong (95.9 %). Erklärungsansatz: Augmentierung wirkt bei wenig Daten als Regularisierung (Vorlesung 01, Steiner et al. 2022); zu starke Augmentierung verändert Rassemerkmale (Fellfarbe, Form) und schadet (Vorlesung 02: „zu viel Augmentierung vermeiden“).
- **Fehler:** häufigste Verwechslung American Pit Bull Terrier → Staffordshire Bull Terrier (62×). Mit Beispielbildern begründen: optisch ähnliche Rassen, Welpen, ungewöhnliche Perspektiven?

**Fragen, die die Diskussion beantworten sollte:** Welche Methode ist am besten und warum? Lohnt sich LoRA gegenüber Linear Probing und vollem Fine-Tuning (Genauigkeit pro Parameter, Trainingszeit)? Sind die Unterschiede größer als die Seed-Streuung? Auf welchen Daten würde eine andere Methode gewinnen?

## 6. Dateien

- `efficiency.png`: Accuracy vs. trainierbare Parameter (Kernabbildung)
- `validation_curves.png`: Lernkurven aller Methoden
- `per_breed_f1.png`, `confusions.csv`: Fehleranalyse
- `tables.tex`: Tabellenzeilen für `paper/ms1/sec/4_results.tex`
- `methods.csv`, `runs.csv`: alle Zahlen
- Datenqualität: `results/data_quality/README.md`; pro Lauf: `runs/<name>/README.md`
