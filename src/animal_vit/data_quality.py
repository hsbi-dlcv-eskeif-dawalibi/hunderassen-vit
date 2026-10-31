"""Data quality check for station 1 ("Prüfen Sie die Qualität Ihrer Daten", lecture 01).

For every image of the selected breeds this module records file format, colour mode, size,
an MD5 hash (exact duplicates) and the share of the image covered by the pet (from the trimap).
:func:`summarize` turns the records into the numbers and findings that go into the
"Dataset" section of the paper. ``scripts/data_report.py`` writes them to
``results/data_quality/``.

Nothing is deleted or changed here: the report lists problems, and the decision
("Datenbereinigung soweit notwendig") is documented in the paper.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: The short image side below which an image is upsampled noticeably for 224 px input.
SMALL_SIDE = 224
#: Pet share of the image below which the animal is very small (background dominates).
SMALL_PET_FRACTION = 0.10


def image_record(split: str, breed: str, image_path: Path, trimap_path: Path | None = None) -> dict[str, Any]:
    """Facts about one image: mode, size, hash, pet fraction, whether colour channels are equal."""
    import numpy as np
    from PIL import Image

    raw = Path(image_path).read_bytes()
    with Image.open(image_path) as img:
        mode, (width, height) = img.mode, img.size
        rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    record: dict[str, Any] = {
        "split": split,
        "breed": breed,
        "file": Path(image_path).name,
        "mode": mode,
        "width": width,
        "height": height,
        "md5": hashlib.md5(raw).hexdigest(),
        # RGB files whose three channels are identical are grayscale photos in disguise.
        "gray_content": bool(
            np.abs(rgb[..., 0] - rgb[..., 1]).max() == 0 and np.abs(rgb[..., 1] - rgb[..., 2]).max() == 0
        ),
        "pet_fraction": None,
    }
    if trimap_path is not None and Path(trimap_path).exists():
        with Image.open(trimap_path) as trimap:
            t = np.asarray(trimap)
        record["pet_fraction"] = float(np.isin(t, (1, 3)).mean())  # 1 = pet, 3 = border (counted as pet)
    return record


def pets_records(sources: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Records for Oxford-IIIT Pet sources (``animal_vit.data._PetsSource`` per split)."""
    for split, source in sources.items():
        for i, pos in enumerate(source.positions):
            yield image_record(
                split,
                source.class_names[source.labels[i]],
                source.base._images[pos],
                source.base._segs[pos],
            )


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate records into per-breed counts and a list of findings."""
    import pandas as pd

    df = pd.DataFrame(records)
    per_breed = df.pivot_table(index="breed", columns="split", values="file", aggfunc="count", fill_value=0)
    per_breed["total"] = per_breed.sum(axis=1)
    per_breed = per_breed.sort_values("total", ascending=False)

    duplicate_groups = df[df.duplicated("md5", keep=False)].groupby("md5")
    duplicates = [
        {"files": list(group["file"]), "breeds": sorted(set(group["breed"])), "splits": sorted(set(group["split"]))}
        for _, group in duplicate_groups
    ]
    short_side = df[["width", "height"]].min(axis=1)
    pet = df["pet_fraction"].dropna()
    summary = {
        "images": len(df),
        "breeds": int(df["breed"].nunique()),
        "per_split": df["split"].value_counts().to_dict(),
        "per_breed": per_breed,
        "balance": {
            "min": int(per_breed["total"].min()),
            "max": int(per_breed["total"].max()),
            "ratio_max_min": float(per_breed["total"].max() / per_breed["total"].min()),
        },
        "modes": df["mode"].value_counts().to_dict(),
        "gray_content": int(df["gray_content"].sum()),
        "size": {
            "width_median": float(df["width"].median()),
            "height_median": float(df["height"].median()),
            "short_side_min": int(short_side.min()),
            f"short_side_below_{SMALL_SIDE}": int((short_side < SMALL_SIDE).sum()),
        },
        "pet_fraction": {
            "mean": float(pet.mean()) if len(pet) else None,
            "min": float(pet.min()) if len(pet) else None,
            f"below_{SMALL_PET_FRACTION}": int((pet < SMALL_PET_FRACTION).sum()),
        },
        "duplicates": duplicates,
        "duplicates_across_splits": [d for d in duplicates if len(d["splits"]) > 1],
        "duplicates_with_different_labels": [d for d in duplicates if len(d["breeds"]) > 1],
    }
    return summary


def findings(summary: dict[str, Any]) -> list[str]:
    """Plain-language findings (German) for the report; each one is something to mention in the paper."""
    out = []
    b = summary["balance"]
    out.append(
        f"Klassenbalance: {b['min']}–{b['max']} Bilder pro Rasse (Verhältnis {b['ratio_max_min']:.2f}), "
        + ("nahezu balanciert." if b["ratio_max_min"] < 1.2 else "unbalanciert, Macro-F1 mitberichten.")
    )
    non_rgb = {m: n for m, n in summary["modes"].items() if m != "RGB"}
    if non_rgb:
        out.append(f"Farbmodi ungleich RGB: {non_rgb}. Werden beim Laden nach RGB konvertiert.")
    if summary["gray_content"]:
        out.append(f"{summary['gray_content']} Bilder sind inhaltlich Graustufen (alle Kanäle gleich).")
    s = summary["size"]
    out.append(
        f"Bildgröße im Median {s['width_median']:.0f}×{s['height_median']:.0f} px; "
        f"{s[f'short_side_below_{SMALL_SIDE}']} Bilder mit kürzerer Seite < {SMALL_SIDE} px (werden hochskaliert)."
    )
    p = summary["pet_fraction"]
    if p["mean"] is not None:
        out.append(
            f"Das Tier bedeckt im Mittel {100 * p['mean']:.0f} % des Bildes; "
            f"{p[f'below_{SMALL_PET_FRACTION}']} Bilder mit < {100 * SMALL_PET_FRACTION:.0f} % (Hintergrund dominiert)."
        )
    if summary["duplicates"]:
        out.append(
            f"{len(summary['duplicates'])} Gruppen exakter Duplikate, davon "
            f"{len(summary['duplicates_across_splits'])} über Splits hinweg (Leakage-Risiko) und "
            f"{len(summary['duplicates_with_different_labels'])} mit unterschiedlichen Labels."
        )
    else:
        out.append("Keine exakten Duplikate gefunden.")
    return out
