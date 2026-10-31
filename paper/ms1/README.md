# Paper Station 1 (CVPR 2026 Author Kit)

`cvpr.sty`, `ieeenat_fullname.bst` und `preamble.tex` sind unverändert aus dem Author Kit
[CVPR2026-v1(latex)](https://github.com/cvpr-org/author-kit/releases/tag/CVPR2026-v1(latex)),
identisch mit der Version im Projektordner `author-kit-CVPR2026-v1-latex-`.

```bash
make paper          # -> paper/ms1/main.pdf (oder: cd paper/ms1 && latexmk -pdf main.tex)
```

## Aufbau (Folie 16, max. 6 Seiten ohne Referenzen und Anhang)

| Datei | Abschnitt | Umfang |
|---|---|---|
| `sec/0_abstract.tex` | Abstract | 100–150 Wörter |
| `sec/1_intro.tex` | Einleitung mit konkreten Beiträgen | ½–¾ Seite |
| `sec/2_related.tex` | Stand der Forschung, mind. 5 Quellen | ¾–1 Seite |
| `sec/3_method.tex` | Methode: Datensatz, Modelle, Parameter | 1–1½ Seiten |
| `sec/4_results.tex` | Ergebnisse | 1–1½ Seiten |
| `sec/5_discussion.tex` | Diskussion | 1–1½ Seiten |
| `main.bib` | Referenzen (zählen nicht) | |
| `sec/X_appendix.tex` | Anhang im Supplementary-Format des Kits (zählt nicht) | |

## Ergebnisse automatisch einbinden

Nach `make station1` werden eingebunden, ohne dass man etwas kopieren muss:
`results/ms1/tables.tex` (Zeilen der Methodentabelle), `results/ms1/efficiency.png`,
`results/data_quality/examples.png` und `class_counts.png`. Fehlt eine Datei, steht dort ein rotes TODO.
Weitere Abbildungen mit `\resultfigure{<Pfad unter results/>}{<Breite>}{<TODO-Text>}`.
Alle Zahlen im Text kommen aus `results/ms1/REPORT.md`.

## Regeln aus dem Author Kit (`sec/2_formatting.tex`, `sec/1_intro.tex`)

- Tabellenunterschrift **über** der Tabelle, Abbildungsunterschrift **unter** der Abbildung.
- Verweise mit `\cref{...}` (am Satzanfang `\Cref{...}`), Labels `sec:`, `tab:`, `fig:`, `eq:`.
- Alle Gleichungen nummerieren (`equation` mit `\label`).
- `\eg`, `\ie`, `\etal` statt „e.g.", „et al."; `\etal` nur ab drei Autoren.
- Fußnoten möglichst vermeiden; Abbildungen mit `\centering` und Breite als Vielfaches von `\linewidth`.
- Abbildungen auch in Graustufen lesbar: nicht nur Farbe, sondern Marker/Linienstil (die Plots von
  `scripts/report_ms1.py` machen das).
- Finale Fassung: `\usepackage{cvpr}` (ohne Lineal, ohne Seitenzahlen). Für Entwürfe mit Seitenzahlen
  `\usepackage[pagenumbers]{cvpr}`.
- `\TODO{...}` vor der Abgabe alle ersetzen; zur Kontrolle in `preamble.tex`
  `\renewcommand{\TODO}[1]{}` aktivieren und prüfen, dass nichts fehlt.
- Das Kit verlangt Englisch für CVPR; laut Dozent ist Deutsch oder Englisch erlaubt.
