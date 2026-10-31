# Station 1 „Adaptation" (Paper-Gliederung)

**Leitfrage (Folie 14):** Wie effizient lässt sich ein vortrainierter ViT an eine neue Aufgabe anpassen?
**Arbeitstitel:** *How Efficiently Can a Pretrained ViT Be Adapted? Linear Probing, LoRA and CNNs on 25 Dog Breeds*
**Format:** CVPR-Template, max. 6 Seiten (ohne Referenzen und Anhang) · **Sprache:** Deutsch oder Englisch ·
**Autoren:** Wael Eskeif, Tarek Dawalibi

> **LaTeX-Vorlage:** [`paper/ms1/`](../../paper/ms1) (CVPR2026-v1), bauen mit `make paper`.
> **Alles ausführen:** `make setup` (einmal), dann `make station1`.
> **Zahlen** nur aus `results/ms1/REPORT.md` (bzw. `tables.tex`) und `results/data_quality/README.md`,
> nie erfundene Werte. Platzhalter im Paper sind rot als `\TODO{}` markiert.

---

## Aufgabe laut Folie 18 → wo im Paper

| Anforderung | Umsetzung | Abschnitt |
|---|---|---|
| Auswahl eines geeigneten Datensatzes | 25 Hunderassen von Oxford-IIIT Pet | 3.1 |
| Hinweis: Datenqualität prüfen, Bereinigung, Labeling, Formatierung | `make data-report` | 3.1 |
| ViT: Linear Probing (nur Kopf nachtrainieren) | `vit_b16_head.yaml` | 3.3, 4.1 |
| ViT: LoRA mit unterschiedlichen Parametern | Rang 4 / 8 / 16, Query/Value vs. alle Layer | 3.3, 4.2 |
| CNN: Training einer CNN-Architektur | ResNet-50, ConvNeXt-T | 3.2, 4.1 |
| Vergleich der Modelle | Tabelle + `efficiency.png` | 4 |
| Diskutieren Sie Ihre Ergebnisse | | 5 |

Zusätzlich: volles Fine-Tuning als Obergrenze, Augmentierungs-Vergleich (Professor empfiehlt Augmentierung).

---

## Seitenbudget (Folie 16)

| Abschnitt | Vorgabe | Plan |
|---|---|---|
| Abstract | 100–150 Wörter | 150 Wörter |
| 1 Einleitung (mit konkreten Beiträgen) | ½–¾ Seite | ½ Seite |
| 2 Stand der Forschung (mind. 5 Quellen) | ¾–1 Seite | ¾ Seite |
| 3 Methode (Datensatz, Modelle, Parameter) | 1–1½ Seiten | 1½ Seiten |
| 4 Ergebnisse | 1–1½ Seiten | 1¼ Seiten |
| 5 Diskussion | 1–1½ Seiten | 1½ Seiten |
| Referenzen, Anhang | zählen nicht | – |

---

## Gliederung

### Abstract
Problem (Anpassung großer vortrainierter Modelle ist teuer) → Vorgehen (Linear Probing, LoRA mit
verschiedenen Rängen/Layern, volles Fine-Tuning, zwei CNNs; 25 Hunderassen; 3 Seeds) → Ergebnis
(LoRA r=8 96,21 ± 0,24 % bei 0,36 % trainierbaren Parametern, Linear Probing 96,03 ± 0,12 % bei 0,02 %
gleichauf; volles Fine-Tuning 95,39 %, schlechteste Methode ResNet-50 94,22 %). Ausformuliert in
`paper/ms1/sec/0_abstract.tex`.

### 1 Einleitung
- Motivation: große vortrainierte ViTs (Vorlesung 03); volles Fine-Tuning teuer, Linear Probing evtl. zu
  schwach für feinkörnige Klassen; LoRA als Mittelweg.
- Feinkörnige Aufgabe: nur eine Tierart (Hinweis des Dozenten), Rassen sehen sich ähnlich.
- Hypothesen A1–A4 (`docs/CONTEXT.md`, Abschnitt 3).
- **Konkrete Beiträge:** (1) Vergleich Linear Probing / LoRA / volles Fine-Tuning / CNN bei gleichem
  Protokoll, (2) LoRA-Ablation (Rang, Ziel-Layer), (3) Kosten-Nutzen-Analyse (Accuracy vs. trainierbare
  Parameter, Trainingszeit), (4) Datenqualitäts-Analyse, (5) dokumentierter Code.

### 2 Stand der Forschung (mind. 5 Quellen, nur relevante, Konferenz/Journal prüfen)
- ViT (Dosovitskiy 2021), ResNet (He 2016), ConvNeXt (Liu 2022)
- Transfer Learning: Kornblith 2019 (Linear vs. Fine-Tuning, u. a. auf Pets), Kumar 2022, AugReg (Steiner 2022)
- LoRA (Hu 2022)
- Art der Bilder: Oxford-IIIT Pet (Parkhi 2012), Stanford Dogs (Khosla 2011)
- Augmentierung: RandAugment (Cubuk 2020), Random Erasing (Zhong 2020)

### 3 Methode
3.1 **Datensatz:** 25 Hunderassen, Splits, Begründung (eine Art, Hunde wegen doppelter Datenmenge,
25 > 5–10 Klassen wegen ausreichend Bildern pro Klasse). Datenqualität aus `results/data_quality/`:
Bilder je Rasse, Farbmodi, Bildgrößen, Duplikate, Anteil des Hundes am Bild. Augmentierung.
**Abb. 1:** `results/data_quality/examples.png`.
3.2 **Modelle und Training:** ViT-B/16, ResNet-50, ConvNeXt-T; AdamW, wd 0.05, Cosine + Warmup,
10 Epochen, 224 px, 3 Seeds, Early Stopping und Auswahl auf Validation.
3.3 **Anpassungsmethoden:** Linear Probing; LoRA mit Formel `h = Wx + (alpha/r) B A x`, B = 0 zu Beginn,
Ränge 4/8/16, Query/Value vs. alle Linear-Layer, Mergen nach dem Training; volles Fine-Tuning.
Lernraten je Methode begründen.

### 4 Ergebnisse
- **Tab. 1:** alle Methoden: Acc, Macro-F1, trainierbare Parameter (absolut und %), Trainingszeit
  (aus `results/ms1/tables.tex`)
- **Abb. 2:** `efficiency.png`: Accuracy vs. trainierbare Parameter
- **Tab. 2:** LoRA-Varianten (Rang, Ziel-Layer)
- **Tab. 3:** Augmentierung none / light / strong
- **Abb. 3:** `validation_curves.png` (Konvergenz)
- Fehleranalyse: `per_breed_f1.png`, häufigste Verwechslungen

### 5 Diskussion (dem Professor besonders wichtig)
- Welche Methode ist die beste, welche die schlechteste, und **warum**?
- Lohnt sich LoRA gegenüber Linear Probing? Was bringt ein höherer Rang? (A1–A3)
- ViT vs. CNN (A4): Unterschied größer als Seed-Streuung?
- Auf welchen Daten würde eine andere Methode gewinnen (Datenmenge, Ähnlichkeit zu ImageNet, Vorlesung 02/03)?
- Augmentierung: hilft sie bei ca. 2 250 Bildern? „Zu viel" Augmentierung (Vorlesung 02)?
- Fehleranalyse: welche Rassen werden verwechselt und warum (Aussehen, Datenqualität)?
- Limitationen: ein Datensatz, Lernraten nicht für jede Methode optimiert, 3 Seeds.
- Schluss: Ausblick auf die folgenden Stationen.

### Referenzen und Anhang (zählen nicht zu den 6 Seiten)
Alle Quellen inkl. Datensatz; Anhang z. B. Hyperparameter-Tabelle, Datenqualitäts-Tabelle je Rasse.

---

## Checkliste vor Abgabe

- [ ] `make station1` vollständig gelaufen, `results/ms1/REPORT.md` und `results/data_quality/README.md` vorhanden
- [ ] Aufbau und Seitenzahl nach Folie 16, max. 6 Seiten im CVPR-Template (ohne Referenzen, Anhang)
- [ ] mind. 5 wissenschaftliche Quellen, nur relevante, Konferenz/Journal geprüft, Datensatz als Referenz
- [ ] Linear Probing, LoRA (mehrere Parameter) und CNN enthalten und verglichen
- [ ] alle Zahlen als Mittelwert ± Std über 3 Seeds, Auswahl nur auf Validation (im Text erwähnt)
- [ ] Diskussion beantwortet „Was bedeutet das?" und „Warum ist A besser als B?"
- [ ] jede Abbildung mit Achsenbeschriftung und Bildunterschrift
- [ ] **Quellcode dokumentiert** (sonst keine Punkte)
- [ ] keine Modelle und keine Bilder im Repo / in ILIAS
- [ ] Git-Tag `ms1` gesetzt, Link im Paper
