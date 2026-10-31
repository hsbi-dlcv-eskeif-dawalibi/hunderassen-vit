# Abgabe Meilenstein 1

| Meilenstein | Abgabe | Gliederung |
|---|---|---|
| Station 1 „Adaptation“ | PDF, max. 6 Seiten, CVPR-Template + Git-Link | [MS1.md](MS1.md) |

**Ablauf**
1. Trainings laufen lassen (`make station1`) und analysieren (`make why`, `make explain`).
2. Zahlen aus `results/ms1/REPORT.md` und `results/summary.md` in die Gliederung übertragen.
3. Paper im CVPR-Template (LaTeX) schreiben: Vorlage in [`paper/ms1/`](../../paper/ms1) (CVPR2026-v1,
   Abschnitte nach Folie 16, Tabellen und Abbildungen werden automatisch aus `results/` eingebunden),
   bauen mit `make paper`. Die Gliederung gibt Abschnitte, Abbildungen und Kernaussagen vor.
4. Checkliste am Ende der Gliederung abhaken, Abgabestand taggen: `make tag MS=ms1`, dann
   `git push origin ms1`. Nachweis aller Anforderungen: [`docs/REQUIREMENTS.md`](../REQUIREMENTS.md).

**Quellenregel:** Jede übernommene Idee, Abbildung oder Formel bekommt eine Quelle (Modulregel).

**Aufbau des Papers** (Vorgabe Folie 16): Abstract (100-150 Wörter), Einleitung (½-¾ Seite, konkrete
Beiträge), Stand der Forschung (¾-1 Seite, mind. 5 wissenschaftliche Quellen), Methode (1-1½),
Ergebnisse (1-1½), Diskussion (1-1½). Referenzen und Anhang zählen nicht zu den 6 Seiten. Der Quellcode
muss dokumentiert sein, Modelle und Bilder gehören nicht ins Repo.
