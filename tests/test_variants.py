"""Tests for the test-image variants and the data split (numpy only)."""

import numpy as np
import pytest

from animal_vit.data import stratified_split, trimap_to_pet_mask
from animal_vit.style_transfer import shape_bias
from animal_vit.variants import GREY, apply_variant


@pytest.fixture()
def sample():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return image, mask


def test_foreground_and_background(sample):
    image, mask = sample
    rng = np.random.default_rng(0)
    fg = apply_variant("foreground", image, mask, rng)
    bg = apply_variant("background", image, mask, rng)
    assert (fg[mask] == image[mask]).all() and (fg[~mask] == GREY).all()
    assert (bg[~mask] == image[~mask]).all() and (bg[mask] == GREY).all()


def test_mixed_background_keeps_pet_and_hides_other_pet(sample):
    image, mask = sample
    other = np.full_like(image, 200)
    other_mask = np.zeros_like(mask)
    other_mask[0:3, 0:3] = True
    other[other_mask] = 10  # the other pet
    mixed = apply_variant("mixed_random", image, mask, np.random.default_rng(0), (other, other_mask))
    assert (mixed[mask] == image[mask]).all()
    assert (mixed[~mask] == 200).all()  # other pet covered by its background colour


def test_patch_shuffle_is_a_permutation(sample):
    image, mask = sample
    shuffled = apply_variant("patch_shuffle_4", image, mask, np.random.default_rng(1))
    assert shuffled.shape == image.shape
    assert sorted(shuffled.reshape(-1, 3).tolist()) == sorted(image.reshape(-1, 3).tolist())
    assert not np.array_equal(shuffled, image)


def test_grayscale(sample):
    image, mask = sample
    grey = apply_variant("grayscale", image, mask, np.random.default_rng(0))
    assert (grey[..., 0] == grey[..., 1]).all()


def test_trimap_mask():
    trimap = np.array([[1, 2], [3, 2]])
    assert trimap_to_pet_mask(trimap).tolist() == [[True, False], [True, False]]


def test_stratified_split_is_balanced_and_disjoint():
    labels = [0] * 50 + [1] * 30 + [2] * 20
    train, val = stratified_split(labels, 0.1, seed=0)
    assert not set(train) & set(val)
    assert len(train) + len(val) == 100
    assert [sum(labels[i] == c for i in val) for c in range(3)] == [5, 3, 2]
    assert stratified_split(labels, 0.1, seed=0) == (train, val)  # deterministic


def test_shape_bias():
    result = shape_bias([0, 0, 1, 2], shape_labels=[0, 0, 0, 0], texture_labels=[1, 1, 1, 1])
    assert result["shape_bias"] == pytest.approx(2 / 3)
    assert result["texture_accuracy"] == pytest.approx(0.25)


def test_confusion_matrix_with_missing_classes(tmp_path):
    from animal_vit.plotting import plot_confusion_matrix

    # class 2 never occurs - the plot must still have three rows/columns
    out = plot_confusion_matrix([0, 1, 1], [0, 1, 0], ["a", "b", "c"], tmp_path / "cm.png")
    assert out.exists()


def test_train_fraction_is_stratified_subset():
    from animal_vit.data import ClassificationDataset, stratified_split  # noqa: F401

    labels = [0] * 40 + [1] * 20
    keep, drop = stratified_split(labels, 1 - 0.25, seed=0)
    assert [sum(labels[i] == c for i in keep) for c in (0, 1)] == [10, 5]
    assert not set(keep) & set(drop)


def test_datasets_with_masks_are_picklable():
    """DataLoader workers (spawn on macOS/Windows) pickle the dataset, including its geometry."""
    import pickle

    from animal_vit.data import eval_geometry

    geometry = pickle.loads(pickle.dumps(eval_geometry(32)))
    assert geometry.size == 32
