"""Tests for the explanation helpers of make explain (no downloads)."""

import numpy as np
import pytest

from animal_vit.explain import (
    ContrastiveTarget,
    box_focus,
    box_mask,
    curve_area,
    map_box,
    normalise,
    top_region_iou,
    vit_reshape,
)


def test_map_box_through_resize_and_center_crop():
    # 600x400 image -> shorter side 224 (scale 0.56), width 336 -> crop 56 px on each side
    box = map_box((300, 0, 400, 400), 600, 400, 224)
    assert box[0] == pytest.approx(300 * 0.56 - 56)
    assert box[1] == 0 and box[3] == pytest.approx(224)
    # boxes outside the crop are clipped
    assert map_box((0, 0, 50, 50), 600, 400, 224)[2] == 0


def test_box_focus_and_mask():
    sal = np.zeros((10, 10))
    sal[:5, :5] = 1  # all saliency in the top-left quarter
    mask = box_mask((0, 0, 5, 5), 10)
    assert mask.sum() == 25
    assert box_focus(sal, mask) == pytest.approx(4.0)  # 100 % of the mass on 25 % of the area
    assert box_focus(np.ones((10, 10)), mask) == pytest.approx(1.0)


def test_normalise_and_iou():
    maps = normalise(np.random.default_rng(0).normal(size=(2, 8, 8)))
    assert np.allclose(maps.reshape(2, -1).sum(1), 1) and (maps >= 0).all()
    a = np.zeros((10, 10))
    a[:2] = 1
    assert top_region_iou(a, a) == pytest.approx(1.0)
    assert top_region_iou(a, a[::-1]) == pytest.approx(0.0)


def test_contrastive_target_and_curve_area():
    import torch

    out = torch.tensor([1.0, 4.0, 2.0])
    assert ContrastiveTarget(1, 2)(out).item() == pytest.approx(2.0)
    assert curve_area(np.ones(6)) == pytest.approx(1.0)


def test_vit_reshape_drops_class_token():
    import torch

    tokens = torch.arange(2 * 197 * 4, dtype=torch.float32).reshape(2, 197, 4)
    grid = vit_reshape(tokens)
    assert grid.shape == (2, 4, 14, 14)
    assert torch.equal(grid[0, :, 0, 0], tokens[0, 1])
