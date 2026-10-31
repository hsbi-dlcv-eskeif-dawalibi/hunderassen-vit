# Projektkontext

> **Zweck:** Vollständiger Kontext unseres Projekts, also Modul und Regeln, Vorlesungsinhalte,
> Forschungsfrage, Datensatz, Modelle, Experimentplan je Meilenstein, Methodik, Risiken und
> Quellen. Das Dokument dient dem Team, der Prüfungsvorbereitung und KI-Assistenten
> (siehe [`CLAUDE.md`](../CLAUDE.md)) als Wissensquelle.
>
> Bedienung des Codes: [`TECHNICAL.md`](TECHNICAL.md) · Abgaben je Meilenstein: [`milestones/`](milestones)

---

## Inhalt

1. [Modul und Rahmenbedingungen](#1-modul-und-rahmenbedingungen)
2. [Vorlesungsinhalte und Bezug zum Projekt](#2-vorlesungsinhalte-und-bezug-zum-projekt)
3. [Forschungsfrage und Hypothesen](#3-forschungsfrage-und-hypothesen)
4. [Datensatz](#4-datensatz)
5. [Modelle](#5-modelle)
6. [Experimentplan je Meilenstein](#6-experimentplan-je-meilenstein)
7. [Metriken und Methodik](#7-metriken-und-methodik)
8. [Zeitplan, Rechenbudget und Risiken](#8-zeitplan-rechenbudget-und-risiken)
9. [Glossar](#9-glossar)
10. [Quellen](#10-quellen)

---

## 1. Modul und Rahmenbedingungen

| | |
|---|---|
| **Modul** | Deep Learning for Computer Vision |
| **Studiengang** | M.Sc. Informatik, Hochschule Bielefeld (Campus Minden) |
| **Dozent** | Prof. Dr.-Ing. Jan Rexilius |
| **Umfang** | 2 SWS Vorlesung + Praktikum, 10 CP |
| **Prüfungsform** | Parcoursprüfung mit 3 benoteten Stationen |
| **Projektthema (vorgegeben)** | *„Verständnis und praktische Anwendung von ViT-Architekturen"* |
| **Team** | Wael Eskeif, Tarek Dawalibi |

### Aufgaben der drei Stationen (Vorlesung 01, Stand 15.10.2026, Folien 14 und 18–20)

| Station | Leitfrage laut Folie | Aufgabe laut Folie | Anteil | Abgabe |
|---|---|---|---|---|
| **1 „Adaptation"** | Wie effizient lässt sich ein vortrainierter ViT an eine neue Aufgabe anpassen? | Datensatz wählen; ViT mit **Linear Probing** (nur Kopf nachtrainieren); ViT mit **LoRA** (Low-Rank Adaptation mit unterschiedlichen Parametern); **CNN** trainieren; Modelle vergleichen; Ergebnisse diskutieren. Hinweis: Datenqualität prüfen, Datenbereinigung soweit nötig, Labeling, Formatierung | 30 % | PDF (max. 6 Seiten, CVPR-Template) + Git-Link |
| **2 „Compression"** | Wie stark lässt sich ein ViT mit Knowledge Distillation verkleinern, ohne viel Genauigkeit zu verlieren? | Modell aus Station 1 als **Teacher**; deutlich kleinerer ViT als **Student**; Student mit und ohne Distillation trainieren; mit Station 1 vergleichen | 30 % | PDF (max. 6 Seiten, CVPR-Template) + Git-Link |
| **3 „Generalization"** | Wie gut kann CLIP einen ViT ohne aufgabenspezifisches Training ersetzen? | CNN, ViT und **CLIP** vergleichen (Zero-Shot, Prompts); mit Station 1 und 2 vergleichen | 40 % | Vortrag (max. 10 min) + Folien (PDF) + Git-Link |

> Dieser Stand enthält **Station 1** (Abschnitt 6); sie läuft mit einem Befehl: `make all` (lokal/Cluster)
> bzw. `bash scripts/runpod.sh` (RunPod, siehe [`RUNPOD.md`](RUNPOD.md)). Die weiteren Stationen folgen.

**Hinweise aus der Vorlesung am 15.10.2026 (Notizen):** Quellcode muss dokumentiert sein (sonst keine
Punkte); Modelle und Bilder nicht mit hochladen; PDF auf Deutsch oder Englisch; Literatur soll sich
auch auf die Art der Bilder beziehen; der Datensatz ist eine Quelle; Ergebnisse und Diskussion sind
besonders wichtig („Was bedeuten die Ergebnisse? Warum ist eine Methode besser?"), schlechte oder
mittlere Performance ist nicht schlimm, wenn sie gut erklärt ist; Augmentierung machen; 224×224;
Standard-ViT von Hugging Face; in Station 1 ein etwas größeres Modell nehmen (Teacher für Station 2);
5–10 Klassen empfohlen, Klassen müssen nicht gleich verteilt sein; nur eine Tierart, damit es
schwerer wird. Rechenressourcen: „Your AI Cluster" (Ansprechpartnerin Frau Behrens) oder Colab
(Ressourcen können entzogen werden, daher sind alle Läufe fortsetzbar).

- CVPR-Template: <https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)>
- Trainingsbilder **nicht** in ILIAS hochladen. Station 3: **ohne Präsentation vor Ort keine Punkte.**

### Regeln (Folie 13)

- Teams mit höchstens zwei Personen. Austausch mit anderen Gruppen über Ideen und Bugs ist erlaubt.
- **Kein Source Code anderer Gruppen, kein Code aus externen Quellen** (Folie: „z. B. Webseiten,
  generative KI-Werkzeuge wie ChatGPT"). Laut Rücksprache (Wael) zielt das aus Sicherheitsgründen auf
  ChatGPT. Der Einsatz von KI-Werkzeugen ist im Anhang des Papers offengelegt.
- **Externe Quellen kenntlich machen**, also Paper, Webseiten und Bücher im Code und in Präsentationen.

---

## 2. Vorlesungsinhalte und Bezug zum Projekt

Kurzfassung der fünf Vorlesungen. Zu jedem Thema steht, wo es in unserem Projekt vorkommt.

### 2.1 Vorlesung 01: Organisation, Grundlagen neuronaler Netze, Hyperparameter

**Einordnung:** KI ⊃ Maschinelles Lernen ⊃ Deep Learning ⊃ CNN/Transformer, angewendet auf
Computer Vision. Prozessmodelle für Data-Science-Projekte: CRISP-DM und TDSP.

**Grundlagen**
- Feature-Vektoren und Feature-Space, Bilder als mehrdimensionale Vektoren, One-Hot-Encoding.
- Lineare Regression → Perzeptron (gewichtete Summe + Aktivierungsfunktion) → Multilayer Perceptron
  (Fully-Connected-/Dense-Layer, `y = f(Wx + b)`).
- **Aktivierungsfunktionen:** linear, Sigmoid und tanh (Vanishing Gradients), **ReLU** (Standard in
  tiefen Netzen), Swish/SiLU, **Softmax** für die Ausgabeschicht.

**Regularisierung (gegen Overfitting)**
- L1/L2-Regularisierung (L2 = **Weight Decay**), **Dropout**, **Batch Normalization** (mit
  gleitendem Mittelwert für die Inferenz).
- **Augmentierung:** Bilder plausibel transformieren, damit das Modell invariant wird. Fortgeschrittene
  Verfahren: **CutOut, MixUp, CutMix, AugMix**.

**Loss-Funktionen:** MSE/MAE (Regression), Binary Cross Entropy, **Categorical Cross Entropy**
(Multi-Klassen-Klassifikation, also unser Fall).

**Lernrate und Optimierung**
- Lernrate zu klein: langsamer Fortschritt. Zu groß: Oszillation oder Instabilität.
- **Learning Rate Decay:** konstant, Step Decay, **Exponential Decay**, Training Schedules.
- Gradientenabstieg: Batch, **SGD**, **Mini-Batch** (bessere Konvergenz, größere Lernraten).

> **In unserem Projekt:** Cross Entropy, AdamW mit Weight Decay 0.05, Cosine-Schedule mit Warmup,
> Augmentierungs-Presets `none` / `light` (Standard: Random Resized Crop + horizontales Spiegeln) /
> `strong` (zusätzlich RandAugment und Random Erasing = CutOut). Kein vertikales Spiegeln und keine
> Farbtonänderung, weil Orientierung und Fellfarbe Rassemerkmale sind; Station 1 vergleicht die drei
> Presets. Jede Zahl ist ein Mittelwert über 3 Seeds, ausgewählt wird nur auf dem Validierungssplit.

### 2.2 Vorlesung 02: CNN, Architekturen, Transfer Learning

**Convolution statt Fully Connected:** Ein FC-Netz auf einem 1000×1000-RGB-Bild hätte etwa 3 Mrd.
Gewichte. Convolutions teilen Gewichte über das ganze Bild und sind dadurch
translationsinvariant. Wichtige Begriffe: Filter/Kernel, Feature Maps, Padding (valid/same),
Stride, 1×1-Convolution (Kanäle ändern, Nichtlinearität), Pooling (Max/Average), Receptive Field
(vergrößert durch Tiefe, Stride/Pooling, Dilated Convolutions).

**Architekturen (chronologisch)**

| Netz | Kernidee |
|---|---|
| LeNet | Ziffernerkennung, ca. 61.7k Parameter |
| AlexNet | Erster großer CNN-Erfolg: ReLU, Dropout, Augmentierung, ca. 60 Mio. Parameter |
| VGG | „Größer ist besser", nur 3×3-Filter |
| GoogLeNet/Inception | Mehrere Filtergrößen parallel (1×1, 3×3, 5×5), Global Average Pooling statt FC |
| ResNet | **Residual Blocks** `f(x) = x + g(x)`, Bottleneck-Blöcke, sehr tiefe Netze (152 Layer) |
| MobileNet v1/v2 | **Depthwise Separable Convolution** (8–9× günstiger), Inverted Residuals, Linear Bottlenecks |
| EfficientNet | Skalierung von Tiefe, Breite und Auflösung per NAS gefunden |

**Komplexität reduzieren:** Quantisierung, **Pruning** (Weight, Neuron, Filter), **Knowledge
Distillation** (Hinton et al. 2014), FC-Layer entfernen, Kernel- und Kanalzahl reduzieren.

**Praxis-Strategien:** Loss und Labels prüfen, Metriken visualisieren, gleiche Vorverarbeitung für
Train und Test, Daten balancieren, mindestens 1000 Bilder pro Klasse. Modellauswahl: erst
Overfitting auf wenigen Daten, dann Lernrate bestimmen, dann mehrere Modelle kurz trainieren,
das beste länger.

**Transfer Learning:** Ein vortrainiertes Netz anpassen statt neu trainieren. Die Wahl der Methode
hängt von der **Datensatzgröße** und der **Ähnlichkeit zum Originaldatensatz** ab. Vorgehen: mit
dem letzten Layer beginnen und schrittweise weitere Layer auftauen, oder Fine-Tuning mit sehr
kleiner Lernrate.

> **In unserem Projekt:** ResNet-50 (Residual Blocks) und ConvNeXt-T als CNN-Referenzen in MS1,
> Transfer Learning aller Modelle von ImageNet, Fine-Tuning-Strategien `head` / `last_n:k` / `all`,
> **Knowledge Distillation** ConvNeXt → DeiT in MS3, Parameter, FLOPs und Latenz (`benchmark.py`).

### 2.3 Vorlesung 03: Transformer und Vision Transformer (Kern des Projekts)

**Transformer (Vaswani et al., NIPS 2017):** Input Embedding (Tokenisierung → Vektoren),
**Positional Encoding** (Transformer sehen Mengen, keine Sequenzen), **Attention**, Feed-Forward-Netz,
Layer Normalization, Residual Connections. Varianten: Encoder-only (BERT, Klassifikation),
Decoder-only (GPT), Encoder-Decoder.

**Scaled Dot-Product Attention**

```
Q = X·W_Q,  K = X·W_K,  V = X·W_V          (gelernte Gewichtsmatrizen)
A = softmax( Q·Kᵀ / √d_k )                 (attention weight matrix)
out = A·V
```

Multi-Head Attention berechnet das parallel in mehreren „Köpfen".

**Vom Bild zum Token**
- *Self-Attention in CNNs* (Non-Local Networks, CVPR 2018): globaler Kontext, aber teuer.
- *Pixel = Token* (iGPT, ICML 2020): 224×224 ergibt 50 176 Tokens, extrem ineffizient.
- ***Patch = Token* (ViT):** 224×224 mit 16×16-Patches ergibt **196 Tokens**.

**Vision Transformer (ViT)**
1. Patch Flatten: 16×16×3 → 1×768.
2. **Linear Projection** auf Embedding-Dimension D.
3. **Position Embedding** (lernbar oder fest) + **[CLS]-Token**.
4. Transformer Encoder (L Blöcke).
5. **MLP Head** auf dem [CLS]-Output → Logits.

Varianten: B(ase), L(arge), H(uge). ViTs brauchen **viele Daten** (ImageNet-1k < 21k < JFT-300M).

**Training von ViTs (AugReg-Studie):** Augmentierung (Mixup, RandAugment) und Regularisierung
(Dropout, Stochastic Depth, Weight Decay) sind **bei begrenzten Daten entscheidend**. Größere
Patches können helfen. Bei wenig Daten: mehr Epochen + AugReg.

**Fine-Tuning-Strategie nach Datenmenge (Folie „ViT – Finetuning")**

| Datenmenge | Strategie |
|---|---|
| klein (< 10k) | nur Head trainieren / Layer mit sehr kleiner LR |
| mittel (10k–50k) | erste Blöcke einfrieren, Rest trainieren |
| groß (> 50k) | Full Fine-Tuning |

Anpassbar sind Classifier Head, Lernrate und Trainingseinstellungen. **Nicht einfach veränderbar
sind Patch-Größe und Architektur.** Das ist relevant für das Patch-Size-Experiment in MS2!

> Die 25 Hunderassen von Oxford-IIIT Pet haben etwa 2 250 Trainingsbilder, also *klein*. Laut Folie wäre „nur Head trainieren" angemessen.
> Genau das vergleichen wir in MS1 mit teilweisem (letzte 4 Blöcke) und vollem Fine-Tuning.

**ViT verbessern**
- **Distillation / DeiT** (Touvron et al., ICML 2021): Ein großes Teacher-Modell trainiert ein
  kleineres Student-Modell. `L = α·L_student + (1−α)·L_distill` mit temperaturskaliertem Softmax
  und KL-Divergenz. **DeiT = ViT + Distillation Token + CNN-Teacher.**
- **Swin Transformer:** hierarchisch, 4×4-Patches, **(Shifted) Window Attention**: Attention nur
  innerhalb lokaler Fenster, Fenster werden verschoben. Die 2D-Struktur bleibt wie bei CNNs erhalten.
- **MobileViT:** Kombination aus MobileNetV2-Blöcken und Transformer, für mobile Geräte.
- **ConvNeXt** (CVPR 2022): ResNet-50 schrittweise „modernisiert" (Swin-ähnliche Stufen,
  4×4-Stem, Inverted Bottleneck, 7×7-Kernel, GELU, LayerNorm). Ein reines CNN auf
  Transformer-Niveau.

> **In unserem Projekt:** ViT-B/16 in Station 1 (Linear Probing, LoRA, volles Fine-Tuning). In Station 2
> wird er per **Distillation** in **DeiT-Ti/S** komprimiert, Station 3 vergleicht mit **CLIP** (ViT-Encoder,
> Bild-Text-Training). **Attention Rollout** visualisiert die Attention (`make explain`). Komponententausch
> (Positional Encoding, Conv-Stem) und Swin sind optional (`configs/extra/`).

### 2.3b Parameter-effizientes Fine-Tuning: LoRA (Station 1, Folie 18)

Laut Planung (Folie 17) folgt der Vorlesungsblock „ViT, SSL, VLM, Fine-Tuning" noch. Das Wichtigste:

- **Linear Probing:** Backbone einfrieren, nur einen neuen linearen Kopf trainieren. Billig, aber die
  Merkmale bleiben die von ImageNet (Vorlesung 02: „Starte mit der letzten Schicht").
- **LoRA** (Hu et al., ICLR 2022): Die vortrainierte Gewichtsmatrix `W` bleibt eingefroren, gelernt wird
  nur eine Änderung niedrigen Rangs `ΔW = B·A` mit `A ∈ R^(r×d_in)`, `B ∈ R^(d_out×r)`, `r ≪ d`:
  `h = W·x + (alpha/r)·B·A·x`. `B` startet bei 0, das Training beginnt also exakt beim vortrainierten
  Modell. Bei ViT-B/16 mit r = 8 auf Query/Value sind das ca. 0,3 M statt 86 M Parameter.
  Nach dem Training wird `B·A` in `W` addiert („mergen"): keine zusätzlichen Inferenzkosten.
- **Volles Fine-Tuning:** alle Gewichte, laut Vorlesung 03 erst bei großen Datensätzen (> 50k) empfohlen.

> **In unserem Projekt:** eigene, dokumentierte LoRA-Implementierung in `src/animal_vit/lora.py`
> (getestet in `tests/test_lora.py`), Vergleich von Rang und Ziel-Layern.

### 2.4 Vorlesung 04: Detektion, Segmentierung, Bias und Fairness

**Objektdetektion = Klassifikation + Lokalisierung** (Bounding Box `x, y, w, h`)
- Sliding Window → **Selective Search** (hierarchische Gruppierung nach Farbe, Textur, Größe und
  Form) → Region Proposals.
- **IoU** (Intersection over Union / Jaccard), **Non-Maximum Suppression**, **Anchor Boxes**.
- Two-Stage: **R-CNN** (CNN-Features + SVM pro Klasse, langsam) → **Fast R-CNN** (RoI Pooling) →
  **Faster R-CNN** (Region Proposal Network mit 3 Skalen × 3 Seitenverhältnissen). Erweiterungen:
  Mask R-CNN, DensePose.
- One-Stage: **YOLO** (S×S-Gitter, Ankerboxen pro Zelle, Klassenwahrscheinlichkeiten, NMS). Viel
  schneller, etwas ungenauer.

**Segmentierung:** semantisch (jedes Pixel bekommt eine Klasse), instanzbasiert, panoptisch.
**Fully Convolutional Networks** mit Down- und Upsampling (Unpooling, Max-Unpooling,
**Transposed Convolutions**), **U-Net**.

**Bias und Fairness**
- Bias entsteht in der ganzen Kette: menschlicher Bias → Daten sammeln → Labeln → Trainieren →
  Evaluieren → Deployen → Monitoring.
- Häufige Arten: **Selection Bias** (Daten entsprechen nicht der echten Verteilung), Reporting Bias,
  Automation Bias, Implicit Bias.
- Beispiele: Prototypen („gelbe Bananen"), Geschlechterbias („Arzt"/„CEO"), geografische Verteilung
  von ImageNet und Open Images.
- **„Neuronale Netze können alles lernen":** 100 % Train-Accuracy auch bei zufälligen Labels
  (Zhang et al., ICLR 2017). Hohe Accuracy beweist also nicht, dass das Richtige gelernt wurde.
- Gegenmaßnahmen: mehr bzw. ausgewogenere Daten, Experten einbeziehen, Outlier betrachten,
  Performance über die Zeit beobachten.

> **In unserem Projekt:** Die Kernfrage *„Was lernt das Modell wirklich?"* wird mit den
> **Segmentierungsmasken** von Oxford-IIIT Pet beantwortet. Die Hintergrund-Varianten testen auf
> Shortcut Learning bzw. Selection Bias (manche Rassen werden typischerweise in bestimmter Umgebung
> fotografiert, z. B. Bernhardiner im Schnee, Neufundländer am Wasser), der
> Foreground-Focus misst, ob die Saliency auf dem Tier liegt.

### 2.5 Vorlesung 05: Style Transfer, Autoencoder, GAN, Siamese Networks

**Neural Style Transfer:** Ein neues Bild `x` minimiert `L_style(x) + L_content(x)`.
- **Content Loss:** Differenz der Feature Maps eines gewählten Layers.
- **Style Loss:** Differenz der **Gram-Matrizen** `G_km = Σ f_k · f_m` über mehrere Layer, also
  welche Merkmale gemeinsam auftreten (Textur).

**Autoencoder (unüberwacht):** Encoder → niedrigdimensionaler Latent Space → Decoder rekonstruiert.
Anwendungen: Entrauschen, Anomalieerkennung, Kolorieren, Bildsuche. Beispiele als FC- und
Convolutional Autoencoder (`Conv2DTranspose`).

**GANs** (Goodfellow et al., NIPS 2014): Generator G erzeugt Fakes, Diskriminator D unterscheidet
echt und falsch. Min-Max-Spiel `min_G max_D log D(x) + log(1 − D(G(z)))`. Es gibt sehr viele
Varianten.

**Verification vs. Recognition:** 1:1-Abgleich vs. 1:N-Klassifikation. Klassifikation skaliert
schlecht bei neuen Identitäten, daher **Similarity Learning** mit **Siamese Networks** (geteilte
Gewichte, Embeddings, Distanz). **Contrastive Loss** (ähnliche Paare zusammen, unähnliche bis
Margin m auseinander) und **Triplet Loss** (Anker, Positiv, Negativ mit Margin α,
**Hard Negative Mining**, offline/online Triplet-Bildung).

> **In unserem Projekt:** **Neural Style Transfer** (Gatys, Gram-Matrizen) erzeugt Cue-Conflict-Bilder:
> die Form der einen Rasse mit der Textur einer anderen. Daraus ergibt sich der **Shape Bias** jedes
> Modells.

---

---

## 3. Forschungsfrage und Hypothesen

### Leitfrage Station 1 (Folie 14)

> **Wie effizient lässt sich ein vortrainierter ViT an eine neue Aufgabe anpassen?**
> Konkret: Wie viel Accuracy erreichen Linear Probing und LoRA mit unterschiedlichen Parametern auf
> 25 Hunderassen im Vergleich zum vollen Fine-Tuning und zu CNNs, und was kosten sie
> (trainierbare Parameter, Trainingszeit)?

**Hypothesen Station 1**

| # | Hypothese | Messgröße |
|---|---|---|
| **A1** | LoRA erreicht mit < 1 % trainierbaren Parametern fast die Accuracy des vollen Fine-Tunings (Abstand < 2 Prozentpunkte). | Test-Acc, trainierbare Parameter |
| **A2** | Linear Probing ist bei feinkörnigen Rassen deutlich schlechter als LoRA, weil die eingefrorenen ImageNet-Merkmale Rassen-Details nicht genug trennen. | Test-Acc, Macro-F1 |
| **A3** | Ein höherer LoRA-Rang bringt ab r = 8 kaum noch Verbesserung (abnehmender Ertrag); Anpassen aller Linear-Layer bringt mehr als ein höherer Rang. | Acc je Rang / Ziel-Layer |
| **A4** | Der beste ViT und die CNNs liegen in der Accuracy nah beieinander (< 2 Prozentpunkte). | Test-Acc |

### Zusatzanalysen: Was sehen die Modelle? (optional, `make analysis`)

Moderne ViTs und CNNs erreichen ähnliche Genauigkeiten. Die Accuracy allein
sagt aber nicht, *worauf* ein Modell seine Entscheidung stützt (Vorlesung 04: „Neuronale Netze
können alles lernen"). Wir messen das mit drei Werkzeugen, die direkt aus den Vorlesungen kommen:

| Werkzeug | Misst | Vorlesung |
|---|---|---|
| **Hintergrund-Varianten** (Segmentierungsmasken) | Verlässt sich das Modell auf das Tier oder auf den Kontext? | 04 (Bias, Segmentierung) |
| **Cue-Conflict-Bilder** (Style Transfer) | Entscheidet das Modell nach Form oder nach Textur? | 05 (Style Transfer) |
| **Saliency** (Attention Rollout, Grad-CAM) + Foreground Focus | Wohin schaut das Modell? | 03 (Attention) |

### Hypothesen der Zusatzanalysen (alte Planung)

| # | Hypothese | Geprüft in |
|---|---|---|
| **H1** | ViTs und CNNs erreichen auf Oxford-IIIT Pet eine vergleichbare Accuracy (Differenz < 2 Prozentpunkte). | MS1 |
| **H2** | ViTs haben einen **höheren Shape Bias** und sind robuster gegen Patch-Shuffle als CNNs. | MS1 |
| **H3** | CNNs reagieren stärker auf Hintergrund-Austausch (`mixed_random`) als ViTs. | MS1 |
| **H4** | Ohne Positions-Embeddings sinkt die Accuracy des ViT nur wenig, sein Shape Bias aber deutlich. Ohne räumliche Ordnung bleibt nur Textur. | MS2 |
| **H5** | Ein Conv-Stem macht den ViT CNN-ähnlicher: geringerer Shape Bias, stärkere Hintergrundabhängigkeit. | MS2 |
| **H6** | Swin (lokale Fenster-Attention) verhält sich in den Analysen zwischen ViT und CNN. | MS3 |
| **H7** | Ein von ConvNeXt destillierter DeiT übernimmt nicht nur Accuracy, sondern auch das „Sehverhalten" des Teachers (Shape Bias, Foreground Focus). | MS3 |

Die Hypothesen sind bewusst so formuliert, dass sie widerlegt werden können. Ein „nein" ist ein
genauso gültiges Ergebnis wie ein „ja".

---

## 4. Datensatz

**Oxford-IIIT Pet** (Parkhi et al., CVPR 2012) · <https://www.robots.ox.ac.uk/~vgg/data/pets/> · Lizenz **CC BY-SA 4.0**

| Eigenschaft | Wert |
|---|---|
| Klassen | **25 Hunderassen** (`data.species: dogs`). Der Datensatz enthält zusätzlich 12 Katzenrassen, die wir nicht nutzen. |
| Bilder | ca. 4 980 Hundebilder (ca. 200 pro Rasse), Gesamtdatensatz 7 349 |
| Offizielle Splits (nur Hunde) | `trainval` ca. 2 500 · `test` ca. 2 490 |
| Unsere Splits | `train` ca. 2 250 · `validation` ca. 250 (10 % je Klasse aus trainval, stratifiziert, `split_seed=0`) · `test` ca. 2 490 |
| Annotationen | Klassenlabel + **Trimap** je Bild (1 = Tier, 2 = Hintergrund, 3 = Rand) |
| Download | `make data` (torchvision, ca. 800 MB) |

**Warum dieser Datensatz und nur Hunde?**
1. **Masken für jedes Bild.** Damit sind Hintergrund-Tests und Foreground Focus *quantitativ*
   messbar, nicht nur als Beispielbilder.
2. **Feinkörnig.** Wir nutzen nur eine Tierart (Hinweis des Dozenten): Hunderassen sehen sich
   ähnlich, die Aufgabe wird dadurch schwerer, und Hund vs. Katze kann nicht als Abkürzung dienen.
   Rassen unterscheiden sich durch Form (Ohren, Schnauze, Körperbau) *und* Textur (Fellmuster,
   -farbe). Ideal für die Frage Shape vs. Texture.
   Hunde statt Katzen, weil sie mit ca. 4 980 Bildern doppelt so viele Daten haben.
   Die 25 Klassen liegen über den empfohlenen 5–10 Klassen. Das ist im Paper zu begründen
   (feinkörnige Aufgabe, ausreichend Bilder pro Klasse).
3. **Klein (< 10k Bilder).** Genau der Fall, in dem die Vorlesung vor ViTs ohne gutes Pretraining
   warnt. Fine-Tuning-Strategien werden damit relevant.
4. **Offen lizenziert und reproduzierbar**: ein Befehl, gleiche Splits für alle.

**Datenqualität (MS1-Aufgabe „Prüfen Sie die Qualität Ihrer Daten")**
- Klassen sind nahezu balanciert (ca. 100 Testbilder pro Klasse).
- Bekannte Eigenheiten: wenige Graustufen-/CMYK-Bilder (werden nach RGB konvertiert), einige Bilder
  mit mehreren Tieren, Trimaps an feinen Strukturen (Fell, Schnurrhaare) ungenau. Der Randbereich
  zählt bei uns zum Tier.
- **Möglicher Selection Bias:** Manche Rassen werden typischerweise in bestimmter Umgebung
  fotografiert (Schnee, Wasser, Wiese, Ausstellung). Genau das prüft die Variante `background`.

---

## 5. Modelle

Alle Modelle sind ImageNet-vortrainiert und werden mit derselben Pipeline (Hugging Face `Trainer`)
feinjustiert.

| Modell | Checkpoint | Typ | Parameter | Rolle |
|---|---|---|---|---|
| ResNet-50 | `microsoft/resnet-50` | CNN, Residual Blocks | 25 M | MS1: klassisches CNN |
| ConvNeXt-T | `facebook/convnext-tiny-224` | modernisiertes CNN | 28 M | MS1: starkes CNN; MS3: **Teacher** |
| ViT-B/16 | `google/vit-base-patch16-224` | ViT, globale Attention | 86 M | Station 1: Linear Probing, LoRA, volles Fine-Tuning; Station 2: **Teacher** (größeres Modell, wie empfohlen) |
| DeiT-Ti/16 | `facebook/deit-tiny-patch16-224` | kleiner ViT | 5,7 M | Station 2: **Student** (15× kleiner als der Teacher) |
| DeiT-S/16 | `facebook/deit-small-patch16-224` | kleiner ViT | 22 M | Station 2: **Student** (4× kleiner) |
| CLIP ViT-B/32 · B/16 · L/14 | `openai/clip-vit-{base-patch32,base-patch16,large-patch14}` | Bild-Text-Modell | 88 M · 86 M · 304 M (Bild-Encoder) | Station 3: Zero-Shot ohne Training |
| Swin-T | `microsoft/swin-tiny-patch4-window7-224` | hierarchischer ViT | 28 M | optional (`configs/extra`) |

**Begründung der Station-2-Students:** DeiT nutzt dieselbe Architektur wie der Teacher (ViT, Patch 16,
224 px), nur schmaler. Unterschiede gehen also auf die Größe und die Distillation zurück, nicht auf
eine andere Architektur. DeiT wurde zudem genau für dateneffizientes Training und Distillation
entwickelt (Touvron et al. 2021).

**Begründung der CLIP-Modelle:** ViT-B/16 hat denselben Bild-Encoder-Typ und dieselbe Größe wie der
Station-1-ViT. Das ist der direkteste Vergleich „mit vs. ohne aufgabenspezifisches Training". B/32 und
L/14 zeigen den Einfluss von Patch-Größe und Modellgröße.

---

## 6. Experimentplan je Meilenstein

Jede Konfiguration läuft mit **3 Seeds** (`SEEDS="1 2 3"`). Alle Kommandos stehen im
[`Makefile`](../Makefile), alle Parameter in [`configs/`](../configs).

### Station 1: Adaptation (`make station1`)

**ViT-B/16: Anpassungsmethoden** (`make ms1-vit`)

| Methode | Config | Trainiert | Lernrate |
|---|---|---|---|
| Linear Probing | `configs/ms1/vit_b16_head.yaml` | nur Klassifikationskopf (ca. 0,02 M) | 1e-3 |
| LoRA r = 4 | `configs/ms1/vit_b16_lora_r4.yaml` | Rang-4-Updates in Query/Value + Kopf | 5e-4 |
| LoRA r = 8 | `configs/ms1/vit_b16_lora_r8.yaml` | Rang 8, Query/Value + Kopf | 5e-4 |
| LoRA r = 16 | `configs/ms1/vit_b16_lora_r16.yaml` | Rang 16, Query/Value + Kopf | 5e-4 |
| LoRA r = 8, alle Layer | `configs/ms1/vit_b16_lora_r8_all.yaml` | Rang 8 in q, k, v, Attention-Output, beiden MLP-Layern + Kopf | 5e-4 |
| Volles Fine-Tuning (Referenz) | `configs/ms1/vit_b16.yaml` | alle 86 M Parameter | 5e-5 |

„Unterschiedliche Parameter" bei LoRA (Folie 18): Rang r (4, 8, 16) und die angepassten Layer
(Query/Value wie im LoRA-Paper vs. alle Linear-Layer). alpha = 2r, also gleiche Skalierung alpha/r.
LoRA wird nach dem Training in die Gewichte gemergt (`src/animal_vit/lora.py`): gleiche
Inferenzkosten wie der Basis-ViT, und das Modell lässt sich in Station 2 direkt als Teacher laden.

**CNN** (Folie: „Training einer CNN-Architektur", volles Fine-Tuning, gleiches Protokoll)

| Modell | Config |
|---|---|
| ResNet-50 | `configs/ms1/resnet50.yaml` |
| ConvNeXt-T | `configs/ms1/convnext_t.yaml` |

**Augmentierung** (`make ms1-augment`): LoRA r = 8 mit `none` / `light` (Standard) / `strong`
(RandAugment + Random Erasing). Zeigt, ob und wie viel Augmentierung bei ca. 2 250 Trainingsbildern hilft.

**Datenqualität** (`make data-report`): Bilder je Rasse, Farbmodi, Bildgrößen, Duplikate,
Flächenanteil des Hundes → `results/data_quality/README.md`.

**Bericht zum Schreiben** (`make report-ms1`): `results/ms1/REPORT.md` mit allen Tabellen, Rangfolge,
Abständen im Vergleich zur Seed-Streuung, LoRA-Vergleich, Augmentierung, Kosten und Fehleranalyse;
dazu `efficiency.png` (Accuracy vs. trainierbare Parameter), `validation_curves.png`,
`per_breed_f1.png` und `tables.tex` für das Paper. Jeder Lauf hat zusätzlich `runs/<name>/README.md`.

### Zusatz: Warum über 90 %? (`make why`, Teil von `make all`)

Die Modelle erreichen auf 25 Rassen 90–96 %. Der Block prüft fünf mögliche Erklärungen mit Experimenten
und schreibt `results/why/REPORT.md`. Jede Prüfung ist an eine Vorlesung angebunden:

| # | Prüfung | Umsetzung | Vorlesung |
|---|---|---|---|
| 1 | Beitrag des Pretrainings | ViT-B/16 und ResNet-50 ohne vortrainierte Gewichte (`configs/why/`, `model.pretrained: false`), 3 Seeds | 02 Transfer Learning, „≥ 1000 Bilder/Klasse“; 03 ViTs brauchen viele Daten |
| 2 | Vorwissen aus ImageNet | 22 der 25 Rassen sind ImageNet-1k-Klassen; Original-Köpfe der Checkpoints **ohne Training** auf diesen Rassen; F1 der 3 Rassen ohne ImageNet-Klasse | 02 Ähnlichkeit zum Originaldatensatz; 03 Fine-Tuning nach Datenmenge |
| 3 | Leakage | Fast-Duplikate Test ↔ Train/Validation über CLIP-Embeddings, Accuracy ohne sie | 02 Praxis-Strategien; 04 „hohe Accuracy beweist nichts“; 05 Embeddings/Bildsuche |
| 4 | Trennbarkeit der Merkmale | Linear Probe auf eingefrorenen ImageNet-Merkmalen mit 1–16 Bildern pro Rasse, neben CLIP | 02 „mit der letzten Schicht beginnen“; 05 Similarity Learning |
| 5 | Rest-Fehler | Bully-Rassen als schwierige Teilmenge, Bilder, die in allen Seeds falsch sind (Label-Rauschen?) | 02 „Labels prüfen“; 04 Outlier, Labeling-Bias |

### Zusatz: Warum diese Rasse und nicht die andere? (`make explain`, Teil von `make all`)

Erklärungsmethoden für die Teacher von Station 2 (bester ViT und bestes CNN) und das zweite CNN,
Ergebnis `results/explain/REPORT.md`. Bezug: Attention-Visualisierung (Vorlesung 03), „Was lernt das
Modell wirklich?“ (Vorlesung 04).

| Analyse | Methode | Quelle / Bibliothek |
|---|---|---|
| Kontrastive Karten für die häufigsten Verwechslungen (A gegen B) | Grad-CAM mit Ziel `logit_A − logit_B` | Selvaraju et al. 2017, pytorch-grad-cam; Prabhushankar et al. 2020 |
| Unabhängige Kontrolle ohne Gradienten | Occlusion (graues Fenster) | Zeiler & Fergus 2014, Captum |
| ViT-Attention | Attention Rollout (eigener Code) | Abnar & Zuidema 2020 |
| Kopf, übriger Hund oder Hintergrund? | Saliency-Anteil in der annotierten Kopf-Box (nur trainval → Validierungssplit) | Parkhi et al. 2012 |
| Stimmt die Erklärung? | Deletion-Test gegen zufälliges Löschen | Petsiuk et al. 2018 |
| Schauen ViT und CNN auf dasselbe? | IoU der wichtigsten 20 % | eigene Auswertung |

Externe Bibliotheken (pytorch-grad-cam, Captum) und KI-Unterstützung werden im Paper genannt (Folie 13, vorher mit
dem Dozenten klären).

### Optionale Zusatzanalysen (`make extra`, `make analysis`)

Positions-Embeddings entfernen/mischen, Conv-Stem und Swin-T (`configs/extra/`) sowie Shape Bias und
Saliency stammen aus der früheren Planung. Sie sind lauffähig, gehören aber zu keiner Station.

---

## 7. Metriken und Methodik

| Metrik | Definition | Quelle |
|---|---|---|
| Accuracy, **Macro-F1** | Standard; Macro-F1 gewichtet alle Rassen gleich | `metrics.json` |
| **ECE** | Expected Calibration Error (15 Bins): Abweichung Konfidenz vs. Trefferquote | `metrics.json` |
| Accuracy je Variante | `original`, `foreground`, `background`, `mixed_random`, `grayscale`, `patch_shuffle_4/8` | `variants.csv` |
| **Background Gap** | `acc(original) − acc(mixed_random)`: wie viel der Kontext ausmacht | aus `variants.csv` |
| **Shape Bias** | Form-Entscheidungen / (Form- + Textur-Entscheidungen) auf Cue-Conflict-Bildern | `shape_bias.json` |
| **Foreground Focus** | Saliency-Masse auf dem Tier / Flächenanteil des Tiers (1 = proportional, > 1 = fokussiert) | `saliency.json` |
| Kosten | Parameter, GFLOPs (1 Bild, 224 px), Latenz (Median, Batch 1 und 32) | `efficiency.json` |

**Regeln für saubere Ergebnisse:**
1. **Modellauswahl nur auf `validation`.** `test` wird pro Lauf genau einmal ausgewertet.
2. **3 Seeds** pro Konfiguration, berichtet wird Mittelwert ± Standardabweichung.
3. Unterschiede nur interpretieren, wenn sie größer als die Seed-Streuung sind. Bei ca. 2 490 Testbildern
   liegt der Standardfehler der Accuracy bei etwa ±0,6 Prozentpunkten.
4. **Gleiches Budget für alle:** gleiche Epochen, gleicher Optimizer (AdamW, wd 0.05, Cosine mit 10 %
   Warmup), gleiche Auflösung (224 px). Ausnahme ist der Conv-Stem (neue Schichten brauchen mehr
   Epochen, das wird so berichtet).
5. Varianten und Cue-Conflict-Bilder sind geseedet, alle Modelle sehen also identische Eingaben.

---

## 8. Zeitplan, Rechenbudget und Risiken

### Rechenbudget (Schätzung, eine moderne GPU)

| Schritt | Läufe | Zeit (ca.) |
|---|---|---|
| Station 1 | 8 Methoden × 3 Seeds + Augmentierung 2 × 3 Seeds | ca. 5–7 h (LoRA und Linear Probing sind schneller als volles Fine-Tuning) |
| Station 2 | 30 Läufe (kleine Students, je 15 Epochen) + 9 Sweep-Läufe | ca. 2–3 h |
| Station 3 | nur Inferenz (CLIP) + Linear Probe + Varianten | ca. 0,5–1 h |
| Optional: Zusatzanalysen | `make extra analysis cue-conflict` | ca. 4–5 h |

### Zeitplan

| Phase | Inhalt | Ergebnis |
|---|---|---|
| Woche 1 | Setup, `make data`, `make debug`, MS1-Trainings | MS1-Tabellen |
| Woche 2 | Analysen MS1, Paper MS1 | **Abgabe MS1** |
| Woche 3 | MS2-Trainings und Analysen | MS2-Tabellen |
| Woche 4 | Paper MS2 | **Abgabe MS2** |
| Woche 5 | MS3-Trainings, Distillation, Gesamtvergleich | MS3-Abbildungen |
| Woche 6 | Folien, Probevortrag | **Vortrag MS3** |

### Risiken (Projekt-Checkliste, Vorlesung 01)

| Risiko | Wahrscheinlichkeit | Auswirkung | Maßnahme |
|---|---|---|---|
| Kein/zu wenig GPU-Zugang | mittel | hoch | früh klären; zur Not 1 Seed + DeiT-S/ResNet-50 statt ViT-B |
| Unterschiede liegen im Rauschen | mittel | mittel | 3 Seeds, Effektgrößen berichten, „kein Unterschied" ist auch ein Ergebnis |
| Style Transfer zu langsam | niedrig | mittel | `--per-class 5`, `--steps 200` |
| Conv-Stem konvergiert nicht | mittel | mittel | höhere LR für den Stem bzw. längeres Training, Ergebnis trotzdem berichten |
| Trimaps ungenau | niedrig | niedrig | Rand zählt zum Tier, Effekt in der Diskussion erwähnen |
| PyTorch-MPS-Bug beim Training auf Apple-GPUs (volles ViT-Fine-Tuning) | niedrig | niedrig | auf CUDA (RunPod) trainieren; `make smoke` nutzt auf dem Mac automatisch die CPU |

---

## 9. Glossar

| Begriff | Erklärung |
|---|---|
| **Patch Embedding** | Lineare Projektion eines 16×16×3-Bildausschnitts auf die Embedding-Dimension |
| **Conv-Stem** | Kleiner Faltungsblock statt der Patch-Projektion (CNN-artige Eingangsstufe) |
| **Positions-Embedding** | Gelernter Vektor pro Patch-Position. Ohne ihn ist der Transformer reihenfolgeblind |
| **[CLS]-Token** | Zusätzliches Token, dessen Ausgabe klassifiziert wird |
| **Attention Rollout** | Multiplikation der Attention-Matrizen aller Schichten. Zeigt, welche Patches in den [CLS]-Token fließen |
| **Grad-CAM** | Gradientengewichtete Aktivierungskarte der letzten Feature-Map |
| **Cue Conflict** | Bild mit Form der Klasse A und Textur der Klasse B |
| **Shape Bias** | Anteil der Form-Entscheidungen auf Cue-Conflict-Bildern |
| **Foreground Focus** | Saliency-Masse auf dem Tier relativ zu seiner Fläche |
| **Knowledge Distillation** | Student lernt zusätzlich die weichen Vorhersagen eines Teachers |
| **Swin** | Hierarchischer ViT mit Attention in (verschobenen) lokalen Fenstern |
| **ECE** | Expected Calibration Error, misst wie gut die Konfidenz zur Trefferquote passt |

---

## 10. Quellen

**Vorlesung:** „Deep Learning for Computer Vision", Prof. Dr.-Ing. Jan Rexilius, HSBI, WS 2026/27, Foliensätze 01–05.

**Datensatz:** Parkhi, Vedaldi, Zisserman, Jawahar. *Cats and Dogs.* CVPR 2012.

**Modelle**
- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- Touvron et al. *Training data-efficient image transformers & distillation through attention.* ICML 2021.
- Liu et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Liu et al. *A ConvNet for the 2020s.* CVPR 2022.
- He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.

**Methoden**
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Kornblith, Shlens, Le. *Do Better ImageNet Models Transfer Better?* CVPR 2019.
- Kumar et al. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022.
- Cubuk et al. *RandAugment: Practical Automated Data Augmentation with a Reduced Search Space.* CVPR Workshops 2020.
- Zhong et al. *Random Erasing Data Augmentation.* AAAI 2020.
- Khosla et al. *Novel Dataset for Fine-Grained Image Categorization: Stanford Dogs.* CVPR Workshop FGVC 2011.
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015.
- Xiao et al. *Early Convolutions Help Transformers See Better.* NeurIPS 2021.
- Gatys, Ecker, Bethge. *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.
- Geirhos et al. *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness.* ICLR 2019.
- Xiao, Engstrom, Ilyas, Madry. *Noise or Signal: The Role of Image Backgrounds in Object Recognition.* ICLR 2021.
- Naseer et al. *Intriguing Properties of Vision Transformers.* NeurIPS 2021.
- Abnar & Zuidema. *Quantifying Attention Flow in Transformers.* ACL 2020.
- Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017.
- Steiner et al. *How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers.* TMLR 2022.

**Software:** PyTorch, torchvision, Hugging Face `transformers`, scikit-learn.
