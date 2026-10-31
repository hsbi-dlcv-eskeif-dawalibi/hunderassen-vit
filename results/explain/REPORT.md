# Warum diese Rasse und nicht die andere? Erklärungsmethoden

Automatisch erzeugt von `scripts/report_explain.py` (`make explain`). Modelle: Teacher von Station 2 (bester ViT und bestes CNN nach Validation) und das zweite CNN, jeweils Seed 1.

> **Vorlesungsbezug:** Attention und Attention-Visualisierung (Vorlesung 03), „Was lernt das Modell wirklich?“ und „hohe Accuracy beweist nicht, dass das Richtige gelernt wurde“ (Vorlesung 04). Die Kopf-Analyse nutzt die Annotationen des Datensatzes, die Deletion-Kurve prüft, ob eine Erklärung stimmt.

## 1. Warum A und nicht B? Kontrastive Karten

Für die häufigsten Verwechslungen aus Station 1 zeigt jede Zeile ein Testbild der Rasse A: Grad-CAM für A, Grad-CAM für B und die **kontrastive Karte** mit dem Ziel `logit_A − logit_B`. Rot sind die Bildteile, die für A *und gegen* B sprechen. Die letzte Spalte (Occlusion) ist eine unabhängige Kontrolle ohne Gradienten.

- `contrastive_convnext_t.png`
- `contrastive_resnet_50.png`
- `contrastive_vit_lora_r_4.png`

## 2. Stimmt die Erklärung? Deletion-Test

Die als wichtig markierten 16×16-Patches werden zuerst grau übermalt. Fällt die Wahrscheinlichkeit der richtigen Klasse schneller als beim zufälligen Löschen, zeigt die Methode wirklich, worauf das Modell sich stützt. **Kleinere Fläche = treuere Erklärung.**

| Modell | Methode | p (0 % gelöscht) | p (10 %) | p (30 %) | Fläche ↓ | Bilder |
|---|---|---|---|---|---|---|
| ViT LoRA r=4 | Grad-CAM | 93.9 | 87.9 | 75.5 | 0.780 | 300 |
| ViT LoRA r=4 | Attention Rollout | 93.9 | 80.1 | 59.0 | 0.648 | 300 |
| ViT LoRA r=4 | Occlusion | 93.9 | 71.3 | 58.6 | 0.638 | 300 |
| ViT LoRA r=4 | zufällig (Referenz) | 93.9 | 93.7 | 91.6 | 0.916 | 300 |
| ConvNeXt-T | Grad-CAM | 93.3 | 84.0 | 67.7 | 0.729 | 300 |
| ConvNeXt-T | Occlusion | 93.3 | 68.3 | 52.6 | 0.590 | 300 |
| ConvNeXt-T | zufällig (Referenz) | 93.3 | 92.4 | 89.0 | 0.885 | 300 |
| ResNet-50 | Grad-CAM | 87.3 | 9.5 | 4.5 | 0.118 | 300 |
| ResNet-50 | Occlusion | 87.3 | 39.2 | 22.6 | 0.291 | 300 |
| ResNet-50 | zufällig (Referenz) | 87.3 | 86.6 | 60.8 | 0.626 | 300 |

## 3. Kopf, Körper oder Hintergrund?

Anteil der Saliency in der annotierten **Kopf-Box**, auf dem **übrigen Hund** (Maske ohne Kopf) und im **Hintergrund**, jeweils geteilt durch den Flächenanteil der Region: 1 = proportional zur Fläche, > 1 = das Modell schaut bevorzugt dorthin. Validierungssplit (nur dort gibt es Kopf-Boxen; diese Bilder wurden nicht zum Training benutzt).

| Modell | Methode | Kopf | übriger Hund | Hintergrund | Bilder |
|---|---|---|---|---|---|
| ViT LoRA r=4 | Grad-CAM | 1.29 | 1.14 | 0.93 | 250 |
| ViT LoRA r=4 | Attention Rollout | 1.29 | 1.07 | 0.93 | 250 |
| ViT LoRA r=4 | Occlusion | 3.38 | 1.01 | 0.49 | 250 |
| ConvNeXt-T | Grad-CAM | 2.20 | 1.02 | 0.78 | 250 |
| ConvNeXt-T | Occlusion | 3.66 | 0.93 | 0.48 | 250 |
| ResNet-50 | Grad-CAM | 3.08 | 1.20 | 0.23 | 250 |
| ResNet-50 | Occlusion | 3.13 | 1.25 | 0.49 | 250 |

Beispiele: `head_examples.png` (weiße Box = annotierter Kopf).

## 4. Schauen ViT und CNN auf dasselbe?

Überlappung (IoU) der jeweils wichtigsten 20 % des Bildes (Grad-CAM) zwischen zwei Modellen, Validierungssplit. Zum Vergleich: zwei zufällige Karten überlappen bei 20 % Fläche im Mittel mit IoU ≈ 0.11.

| Modell A | Modell B | IoU der Top-20 % | Bilder |
|---|---|---|---|
| ViT LoRA r=4 | ConvNeXt-T | 0.15 | 250 |
| ViT LoRA r=4 | ResNet-50 | 0.16 | 250 |
| ConvNeXt-T | ResNet-50 | 0.28 | 250 |

## 5. Befunde (Material für die Diskussion)

Aus den Zahlen abgeleitet; Formulierung und Gewichtung sind eure Aufgabe.

- **Kontrastive Karten:** In den Abbildungen beschreiben, *welche* Bildteile A von B trennen (Kopfform, Schnauze, Ohren, Fell, Körperbau). Oxford-IIIT Pet hat keine Annotationen für Ohren oder Augen; diese Zuordnung ist eure qualitative Beobachtung und sollte an mehreren Beispielen belegt werden. Stimmen Grad-CAM und Occlusion überein, ist die Aussage belastbarer.
- **Treue (ViT LoRA r=4):** Occlusion ist am treuesten (Fläche 0.638 gegenüber 0.916 beim zufälligen Löschen, 30 % kleiner). Alle Methoden liegen unter der Zufallsreferenz: Die markierten Regionen tragen mehr zur Entscheidung bei als zufällige. Je größer der Abstand, desto eher darf man die Karte deuten.
- **Treue (ConvNeXt-T):** Occlusion ist am treuesten (Fläche 0.590 gegenüber 0.885 beim zufälligen Löschen, 33 % kleiner). Alle Methoden liegen unter der Zufallsreferenz: Die markierten Regionen tragen mehr zur Entscheidung bei als zufällige. Je größer der Abstand, desto eher darf man die Karte deuten.
- **Treue (ResNet-50):** Grad-CAM ist am treuesten (Fläche 0.118 gegenüber 0.626 beim zufälligen Löschen, 81 % kleiner). Alle Methoden liegen unter der Zufallsreferenz: Die markierten Regionen tragen mehr zur Entscheidung bei als zufällige. Je größer der Abstand, desto eher darf man die Karte deuten.
- **Achtung (ResNet-50):** Schon zufälliges Übermalen senkt die Sicherheit stark (87 % → 10 %). Das Modell reagiert also allgemein empfindlich auf graue Flächen (vgl. Variante „nur Hund auf Grau“ in Station 3). Die Flächen dieses Modells deshalb nur untereinander vergleichen, nicht mit anderen Modellen.
- **Kopf:** Mit der jeweils treuesten Methode (Abschnitt 2) liegt die Saliency am Kopf bei ConvNeXt-T 3.7× (Occlusion), ViT LoRA r=4 3.4× (Occlusion), ResNet-50 3.1× (Grad-CAM) über dem Flächenanteil. Alle Modelle stützen sich also ähnlich stark auf den Kopf. Rassen unterscheiden sich vor allem am Kopf (Schnauze, Ohren, Augenpartie); das ist dieselbe Region, die auch ein Mensch nutzt. Der Hintergrund liegt unter 1, passend zu den Hintergrund-Tests in Station 3.
- **Methoden widersprechen sich (ViT LoRA r=4):** Kopf-Fokus mit Grad-CAM 1.3×, mit Occlusion 3.4×. Grad-CAM ist bei ViTs bekannt unzuverlässig (Chefer et al., CVPR 2021); hier zählt die Methode, die im Deletion-Test besser abschneidet.
- **Übereinstimmung:** ViT LoRA r=4 und ConvNeXt-T überlappen nur zu IoU 0.15 (kaum mehr als zwei zufällige Karten, Zufall ≈ 0.11). Gleiche Accuracy heißt also nicht, dass die Modelle dieselben Bildteile nutzen; passt zur Leitfrage „Was sehen Vision Transformer?“. Einschränkung: Die Übereinstimmung beruht auf Grad-CAM, das beim ViT im Deletion-Test am schwächsten abschneidet; außerdem sind die Karten unterschiedlich fein (ViT 14×14 Patches, CNNs 7×7). Dass die beiden CNNs untereinander stärker übereinstimmen als mit dem ViT, ist das robustere Ergebnis.

## Quellen (im Paper zitieren)

- Selvaraju et al., Grad-CAM, ICCV 2017; Implementierung: Gildenblat et al., pytorch-grad-cam (github.com/jacobgil/pytorch-grad-cam)
- Zeiler & Fergus, Visualizing and Understanding Convolutional Networks, ECCV 2014 (Occlusion); Implementierung: Kokhlikyan et al., Captum, 2020 (captum.ai)
- Abnar & Zuidema, Quantifying Attention Flow in Transformers, ACL 2020 (Attention Rollout, eigener Code)
- Prabhushankar et al., Contrastive Explanations in Neural Networks, ICIP 2020 (Idee der kontrastiven Karte)
- Petsiuk et al., RISE, BMVC 2018 (Deletion-Metrik)
- Chefer et al., Transformer Interpretability Beyond Attention Visualization, CVPR 2021 (Grenzen von Grad-CAM bei ViTs)
- Parkhi et al., Cats and Dogs, CVPR 2012 (Oxford-IIIT Pet mit Kopf-Boxen)

## Dateien

- `contrastive_*.png`, `head_examples.png`, `deletion_curves.png`
- `faithfulness.csv`, `head_focus.csv`, `agreement.csv`, `runs.json`
