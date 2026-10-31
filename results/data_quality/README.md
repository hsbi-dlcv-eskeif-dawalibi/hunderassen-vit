# Datenqualität: Oxford-IIIT Pet, `dogs`

4978 Bilder, 25 Rassen. Splits: trainval 2492, test 2486. Quelle: Parkhi et al., CVPR 2012, CC BY-SA 4.0.

## Befunde

- Klassenbalance: 189–200 Bilder pro Rasse (Verhältnis 1.06), nahezu balanciert.
- 33 Bilder sind inhaltlich Graustufen (alle Kanäle gleich).
- Bildgröße im Median 500×375 px; 133 Bilder mit kürzerer Seite < 224 px (werden hochskaliert).
- Das Tier bedeckt im Mittel 40 % des Bildes; 123 Bilder mit < 10 % (Hintergrund dominiert).
- Keine exakten Duplikate gefunden.

## Bilder pro Rasse

| Rasse | test | trainval | total |
|---|---|---|---|
| American Bulldog | 100 | 100 | 200 |
| American Pit Bull Terrier | 100 | 100 | 200 |
| Basset Hound | 100 | 100 | 200 |
| Beagle | 100 | 100 | 200 |
| Chihuahua | 100 | 100 | 200 |
| German Shorthaired | 100 | 100 | 200 |
| English Setter | 100 | 100 | 200 |
| Havanese | 100 | 100 | 200 |
| Great Pyrenees | 100 | 100 | 200 |
| Yorkshire Terrier | 100 | 100 | 200 |
| Wheaten Terrier | 100 | 100 | 200 |
| Japanese Chin | 100 | 100 | 200 |
| Leonberger | 100 | 100 | 200 |
| Pomeranian | 100 | 100 | 200 |
| Miniature Pinscher | 100 | 100 | 200 |
| Saint Bernard | 100 | 100 | 200 |
| Pug | 100 | 100 | 200 |
| Samoyed | 100 | 100 | 200 |
| Shiba Inu | 100 | 100 | 200 |
| Scottish Terrier | 99 | 100 | 199 |
| Boxer | 99 | 100 | 199 |
| Keeshond | 99 | 100 | 199 |
| English Cocker Spaniel | 100 | 96 | 196 |
| Newfoundland | 100 | 96 | 196 |
| Staffordshire Bull Terrier | 89 | 100 | 189 |

Abbildungen: `class_counts.png`, `examples.png` (gelb: Umriss der Maske). Rohdaten: `images.csv`.
