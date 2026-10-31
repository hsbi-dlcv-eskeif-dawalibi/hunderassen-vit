"""Data quality check (station 1) on a few synthetic images and trimaps."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("pandas")

from animal_vit.data_quality import findings, image_record, summarize  # noqa: E402


def _save(path, array, mode=None):
    from PIL import Image

    Image.fromarray(array, mode=mode).save(path)
    return path


def test_records_and_summary(tmp_path):
    rng = np.random.default_rng(0)
    colour = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, (100, 120), dtype=np.uint8)
    trimap = np.full((300, 400), 2, dtype=np.uint8)
    trimap[100:200, 100:300] = 1  # pet covers 1/6 of the image

    a = _save(tmp_path / "a.png", colour)
    b = _save(tmp_path / "b.png", colour)  # exact duplicate of a, different split
    c = _save(tmp_path / "c.png", gray)
    t = _save(tmp_path / "t.png", trimap)

    records = [
        image_record("trainval", "Beagle", a, t),
        image_record("test", "Beagle", b),
        image_record("trainval", "Pug", c),
    ]
    assert records[0]["pet_fraction"] == pytest.approx(1 / 6)
    assert records[2]["mode"] == "L" and records[2]["gray_content"]

    summary = summarize(records)
    assert summary["images"] == 3 and summary["breeds"] == 2
    assert summary["modes"] == {"RGB": 2, "L": 1}
    assert len(summary["duplicates_across_splits"]) == 1
    assert summary["size"]["short_side_below_224"] == 1
    text = " ".join(findings(summary))
    assert "Duplikate" in text and "RGB" in text
