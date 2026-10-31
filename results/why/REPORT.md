# Warum über 90 %? Belege aus Experimenten

Automatisch erzeugt von `scripts/report_why.py` (`make why`). Testsplit der 25 Hunderassen, Mittelwert ± Std über Seeds. Jeder Abschnitt prüft eine mögliche Erklärung.

## 1. Pretraining: dieselbe Architektur ohne vortrainierte Gewichte

> **Vorlesungsbezug:** Transfer Learning statt Neutraining und die Faustregel „mindestens 1000 Bilder pro Klasse“ (Vorlesung 02, wir haben ca. 90); ViTs brauchen viele Daten, CNNs haben eine induktive Verzerrung (Vorlesung 02/03).

| Architektur | mit Pretraining (Station 1) | ohne Pretraining | Differenz | Macro-F1 ohne | Seeds |
|---|---|---|---|---|---|
| ViT-B/16 | 95.39 ± 0.39 | 11.18 ± 2.61 | +84.2 pp | 9.79 ± 2.83 | 3 |
| ResNet-50 | 94.22 ± 0.85 | 31.07 ± 0.41 | +63.2 pp | 29.79 ± 0.30 | 3 |

Gleiche Daten, Splits und Auswertung; ohne Pretraining 30 statt 10 Epochen und eine größere Lernrate (`configs/why/`), damit das Training eine faire Chance hat.

## 2. Vorwissen: 22 der 25 Rassen sind ImageNet-1k-Klassen

> **Vorlesungsbezug:** Die Wahl der Transfer-Methode hängt von der *Ähnlichkeit zum Originaldatensatz* ab (Vorlesung 02); hier wird diese Ähnlichkeit gemessen. Fine-Tuning-Tabelle für kleine Datensätze: „nur Head trainieren“ (Vorlesung 03).

In ImageNet-1k (Russakovsky et al. 2015) fehlen nur American Bulldog, Havanese, Shiba Inu. Die Checkpoints von Station 1 sind auf ImageNet-1k trainiert, haben diese Rassen also schon als Klassen gelernt.

| Modell (Original-Kopf, kein Training) | Rassen | Testbilder | Acc 22 Rassen (%) | Top-1 aus 1000 (%) | Top-5 aus 1000 (%) |
|---|---|---|---|---|---|
| ViT-B/16 | 22 | 2186 | 96.20 | 88.29 | 98.86 |
| ResNet-50 | 22 | 2186 | 96.75 | 89.39 | 98.72 |
| ConvNeXt-T | 22 | 2186 | 96.52 | 90.30 | 99.27 |

Zum Vergleich auf denselben 22 Rassen: das trainierte `ms1_vit_b16_lora_r4` 95.88 % (25 Klassen zur Auswahl, also eher schwerer als die 22-Klassen-Auswahl des ImageNet-Kopfs).

F1 je Rassengruppe (Station-Läufe, Mittel über Seeds):

| Modell | F1 in ImageNet (22) | F1 nicht in ImageNet (3) | einzeln (%) |
|---|---|---|---|
| ViT linear probing | 96.2 | 93.8 | American Bulldog 86, Havanese 97, Shiba Inu 99 |
| ViT LoRA r=8 | 96.3 | 94.6 | American Bulldog 87, Havanese 97, Shiba Inu 99 |
| ViT full fine-tuning | 95.5 | 93.0 | American Bulldog 83, Havanese 97, Shiba Inu 99 |
| ResNet-50 | 94.1 | 92.4 | American Bulldog 86, Havanese 95, Shiba Inu 96 |
| ConvNeXt-T | 94.8 | 93.3 | American Bulldog 86, Havanese 96, Shiba Inu 98 |
| DeiT-Ti (no KD) | 90.2 | 87.1 | American Bulldog 77, Havanese 90, Shiba Inu 94 |
| CLIP ViT-L/14 zero-shot | 96.0 | 96.0 | American Bulldog 89, Havanese 98, Shiba Inu 100 |

## 3. Leakage: Fast-Duplikate zwischen Test und Training

> **Vorlesungsbezug:** „Hohe Accuracy beweist nicht, dass das Richtige gelernt wurde“ (Zhang et al. 2017, Vorlesung 04) und saubere Trennung von Train und Test (Vorlesung 02, Praxis-Strategien). Die Suche nutzt Bild-Embeddings und Ähnlichkeit wie Bildsuche und Siamese Networks (Vorlesung 05).

Kosinus-Ähnlichkeit (CLIP ViT-B/16) jedes Testbilds zum ähnlichsten Train-/Validation-Bild:

| Ähnlichkeit | Testbilder | Anteil |
|---|---|---|
| ≥ 0.9 | 1735 | 69.79 % |
| ≥ 0.95 | 215 | 8.65 % |
| ≥ 0.98 | 21 | 0.84 % |

Bester ViT (`ms1_vit_b16_lora_r4`): alle Testbilder 96.11 %, ohne die 215 Bilder ≥ 0.95: 95.77 %.

Davon gleiche Rasse wie der Nachbar: 99 %. Abbildungen: `duplicates.png`, `max_sim_hist.png`.

## 4. Trennbarkeit: wie wenige Bilder reichen?

> **Vorlesungsbezug:** Linear Probing / „mit der letzten Schicht beginnen“ (Vorlesung 02, Station 1) und Embeddings, in denen ähnliche Bilder nah beieinander liegen (Similarity Learning, Vorlesung 05).

| Merkmale (eingefroren) + Linear Probe | k=1 | k=2 | k=4 | k=8 | k=16 | alle Bilder |
|---|---|---|---|---|---|---|
| ResNet-50 (ImageNet) | 87.41 ± 1.91 | 92.81 ± 1.27 | 94.22 ± 0.76 | 94.58 ± 0.56 | 94.95 ± 0.47 | 96.06 |
| ViT-B/16 (ImageNet) | 84.66 ± 2.63 | 92.83 ± 0.72 | 94.50 ± 0.44 | 94.21 ± 0.50 | 94.85 ± 0.18 | 95.94 |
| CLIP ViT-B/16 | 41.08 ± 1.48 | 59.55 ± 0.76 | 73.32 ± 2.33 | 81.78 ± 1.31 | 88.19 ± 0.92 | - |
| CLIP ViT-L/14 | 57.72 ± 0.56 | 73.05 ± 1.13 | 84.73 ± 1.16 | 90.96 ± 0.35 | 93.55 ± 0.16 | - |

Eingefrorene Merkmale (Linear Probe, alle Bilder) gegen volles Fine-Tuning aus Station 1:

| Backbone | eingefroren + Linear Probe (%) | volles Fine-Tuning (%) | Differenz |
|---|---|---|---|
| ViT-B/16 | 95.94 | 95.39 | +0.55 pp |
| ResNet-50 | 96.06 | 94.22 | +1.84 pp |

## 5. Wo liegt der Rest-Fehler?

> **Vorlesungsbezug:** „Loss und Labels prüfen“ (Vorlesung 02, Praxis-Strategien), Outlier betrachten und Bias beim Labeln (Vorlesung 04).

| Teilmenge | Anteil der Testbilder | Accuracy (%) |
|---|---|---|
| 4 Bully-Rassen (American Bulldog, American Pit Bull Terrier, Staffordshire Bull Terrier, Boxer) | 16 % | 84.88 |
| übrige Rassen | 84 % | 98.19 |

Modell: `ms1_vit_b16_lora_r4` (auf Validation gewählt), 3 Seeds. Fehler innerhalb der Bully-Rassen: 59 % aller Fehler.

In allen Seeds falsch: 71 Testbilder. Die sichersten davon (`confident_errors.csv`, `confident_errors.png`) von Hand prüfen: Label-Fehler im Datensatz?

## 6. Befunde (Material für die Diskussion)

Aus den Zahlen abgeleitet; Formulierung und Gewichtung sind eure Aufgabe.

- **Pretraining (ViT-B/16):** mit ImageNet-Gewichten 95.4 %, ohne 11.2 % (Zufall 4 %), Differenz 84.2 pp (gepaart über Seeds, > 2·SE). Der größte Teil der Accuracy stammt aus dem Pretraining, nicht aus den ca. 2 250 Trainingsbildern. Beim ViT ist der Einbruch erwartungsgemäß stärker: ohne die induktive Verzerrung der Faltung braucht er sehr viele Daten (Dosovitskiy et al. 2021).
- **Pretraining (ResNet-50):** mit ImageNet-Gewichten 94.2 %, ohne 31.1 % (Zufall 4 %), Differenz 63.2 pp (gepaart über Seeds, > 2·SE). Der größte Teil der Accuracy stammt aus dem Pretraining, nicht aus den ca. 2 250 Trainingsbildern. Die CNN profitiert ohne Pretraining von ihrer induktiven Verzerrung (Lokalität, Translationsinvarianz; Vorlesung 02/03).
- **Vorwissen:** Ohne ein einziges Trainingsbild erkennt der ImageNet-Kopf von ResNet-50 die 22 Rassen zu 96.8 %; das trainierte Modell erreicht auf denselben Rassen 95.9 %. Unser Training lernt die Aufgabe also nicht neu, sondern passt eine schon vorhandene Unterscheidung an. Das relativiert die hohen Werte von Station 1 und erklärt, warum Linear Probing (nur ein neuer Kopf) so gut ist. Fairness: Der ImageNet-Kopf wählt nur zwischen 22 Rassen und kennt den schwierigen American Bulldog nicht.
- **Rassen ohne ImageNet-Klasse** sind nur wenig schlechter, und der Abstand kommt vor allem von American Bulldog (verwechselbar mit Pit Bull). Havanese und Shiba Inu erkennen die Modelle trotzdem gut: Die Merkmale übertragen sich auf neue, ähnliche Klassen. Einschränkung: nur 3 Rassen.
- **Leakage:** 215 Testbilder (8.6 %) haben ein sehr ähnliches Trainingsbild (≥ 0.95); ohne sie ändert sich die Accuracy um -0.34 pp. Die hohen Werte sind also nicht durch Duplikate aufgebläht. `duplicates.png` ansehen: echte Duplikate oder nur ähnliche Motive (gleicher Hund, andere Pose)?
- **Trennbarkeit:** Mit eingefrorenen ImageNet-ViT-Merkmalen reicht **ein** Bild pro Rasse für 85 %, 16 Bilder für 95 %. Die Rassen liegen im Merkmalsraum schon weitgehend linear getrennt; die übrigen Trainingsbilder bringen vergleichsweise wenig dazu. Vergleich mit CLIP: welches Pretraining liefert die besser trennbaren Merkmale?
- **Eingefroren schlägt volles Fine-Tuning:** ViT-B/16 95.9 % gegenüber 95.4 %, ResNet-50 96.1 % gegenüber 94.2 %. Volles Fine-Tuning auf ca. 2 250 Bildern verschiebt die guten ImageNet-Merkmale eher, als dass es sie verbessert (Kumar et al., ICLR 2022: „Fine-Tuning can Distort Pretrained Features“). Das stützt die Empfehlung der Vorlesung 03, bei kleinen Datensätzen nur den Kopf zu trainieren. Einschränkung: Die Linear Probe ist eine logistische Regression mit auf Validation gewähltem C, das Fine-Tuning lief 10 Epochen mit festem Schema; nur ein Lauf je Probe.
- **Rest-Fehler:** 16 % der Testbilder (Bully-Rassen) verursachen 59 % aller Fehler; dort liegt die Accuracy bei 84.9 % statt 98.2 %. Diese Rassen unterscheiden sich vor allem in Größe und Proportionen, die auf einem Foto ohne Maßstab kaum erkennbar sind; selbst die Rassestandards überschneiden sich (American Pit Bull vs. Staffordshire).
- **Label-Rauschen?** 71 Testbilder sind in jedem Seed falsch, viele davon mit hoher Konfidenz. Bei der Durchsicht von `confident_errors.png` zählen: falsches Label, Mischling, mehrere Hunde, Hund kaum sichtbar. Ein Teil des Rest-Fehlers ist dann keine Modellschwäche, sondern eine Obergrenze des Datensatzes.

**Gesamtbild prüfen:** Ergeben Pretraining (1), Vorwissen (2) und Trennbarkeit (4) zusammen eine schlüssige Erklärung, und schließt (3) Leakage aus? Was bedeutet das für die Aussagekraft von 96 % auf einem Datensatz, dessen Klassen im Pretraining vorkamen? Auf welchem Datensatz wäre die Aufgabe wirklich schwer (Rassen außerhalb von ImageNet, wenige Bilder, andere Domäne)?

## Dateien

- `scratch_vs_pretrained.png`, `fewshot_curves.png`, `max_sim_hist.png`, `duplicates.png`, `confident_errors.png`
- `imagenet_head.csv`, `test_max_sim.csv`, `fewshot_backbones.csv`, `confident_errors.csv`
