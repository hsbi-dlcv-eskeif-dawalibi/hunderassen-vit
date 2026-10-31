"""Test-image variants that separate *what* a model relies on.

All variants are computed on the fly from the original test image and its pet mask
(Oxford-IIIT Pet trimap), after the evaluation geometry (resize + center crop):

=================  ==========================================================  =====================
Variant            What stays                                                  Question
=================  ==========================================================  =====================
``original``       everything                                                  reference accuracy
``foreground``     the pet; background replaced by neutral grey                needs context?
``background``     the background; pet replaced by neutral grey                background shortcut?
``mixed_random``   the pet pasted onto the background of a *different* breed   pet vs. context
``grayscale``      shape and texture, no colour                                colour cue?
``patch_shuffle``  local texture; global arrangement destroyed (k x k grid)    shape vs. texture
=================  ==========================================================  =====================

Inspired by the "Backgrounds Challenge" (Xiao et al., ICLR 2021) and the patch-shuffle
experiments of Naseer et al. (*Intriguing Properties of Vision Transformers*, NeurIPS
2021). Lecture link: 04 (bias, "what does the model really learn?") and 05 (texture).
"""

from __future__ import annotations

import numpy as np

VARIANTS = ("original", "foreground", "background", "mixed_random", "grayscale", "patch_shuffle_4", "patch_shuffle_8")
GREY = np.array([124, 116, 104], dtype=np.uint8)  # ImageNet mean colour


def _fill(image: np.ndarray, region: np.ndarray, colour=GREY) -> np.ndarray:
    out = image.copy()
    out[region] = colour
    return out


def foreground_only(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep the pet, replace the background with grey."""
    return _fill(image, ~mask)


def background_only(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep the background, replace the pet with grey (can the model guess the breed from the scene?)."""
    return _fill(image, mask)


def mixed_background(image: np.ndarray, mask: np.ndarray, other: np.ndarray, other_mask: np.ndarray) -> np.ndarray:
    """Paste the pet of ``image`` onto the background of ``other``.

    The other image's own pet is covered with that image's mean background colour, so no
    second animal is visible.
    """
    bg_pixels = other[~other_mask]
    fill = bg_pixels.mean(axis=0).astype(np.uint8) if len(bg_pixels) else GREY
    canvas = _fill(other, other_mask, fill)
    canvas[mask] = image[mask]
    return canvas


def grayscale(image: np.ndarray) -> np.ndarray:
    """Remove colour (ITU-R BT.601 luma), keep shape and fur pattern."""
    grey = (image @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=-1)


def patch_shuffle(image: np.ndarray, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Cut the image into ``grid x grid`` tiles and permute them."""
    h, w = image.shape[:2]
    th, tw = h // grid, w // grid
    image = image[: th * grid, : tw * grid]
    tiles = [image[r * th : (r + 1) * th, c * tw : (c + 1) * tw] for r in range(grid) for c in range(grid)]
    order = rng.permutation(len(tiles))
    rows = [np.concatenate([tiles[order[r * grid + c]] for c in range(grid)], axis=1) for r in range(grid)]
    return np.concatenate(rows, axis=0)


def apply_variant(
    name: str,
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    other: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return the variant ``name`` of an ``HxWx3`` uint8 image with boolean pet ``mask``."""
    if name == "original":
        return image
    if name == "foreground":
        return foreground_only(image, mask)
    if name == "background":
        return background_only(image, mask)
    if name == "mixed_random":
        if other is None:
            raise ValueError("mixed_random needs another image")
        return mixed_background(image, mask, *other)
    if name == "grayscale":
        return grayscale(image)
    if name.startswith("patch_shuffle_"):
        return patch_shuffle(image, int(name.rsplit("_", 1)[1]), rng)
    raise ValueError(f"Unknown variant '{name}', choose from {VARIANTS}")


class VariantDataset:
    """Test split with a fixed variant applied; yields ``pixel_values``, ``label``, ``mask``.

    ``mixed_random`` pairs every image with a random image of a *different* class
    (seeded, so all models see identical inputs).
    """

    def __init__(self, base, variant: str, size: int, mean, std, seed: int = 0):
        from animal_vit.data import eval_geometry, normalize_tensor, trimap_to_pet_mask

        self.base, self.variant = base, variant
        self.geometry = eval_geometry(size)
        self.to_tensor = normalize_tensor(mean, std)
        self.trimap_to_mask = trimap_to_pet_mask
        self.seed = seed
        labels = base.labels if variant == "mixed_random" else None
        self.partner = None
        if labels is not None:
            rng = np.random.default_rng(seed)
            labels = np.asarray(labels)
            self.partner = []
            for i, lab in enumerate(labels):
                others = np.flatnonzero(labels != lab)
                if len(others) == 0:  # only one class present (tiny subsets): any other image
                    others = np.array([j for j in range(len(labels)) if j != i] or [i])
                self.partner.append(int(rng.choice(others)))

    def __len__(self) -> int:
        return len(self.base.indices)

    def _load(self, i: int):
        image, label, trimap = self.base.load(self.base.indices[i])
        image, trimap = self.geometry(image.convert("RGB"), trimap)
        return np.asarray(image), label, self.trimap_to_mask(trimap)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image

        image, label, mask = self._load(i)
        other = self._load(self.partner[i])[::2] if self.partner is not None else None
        rng = np.random.default_rng(self.seed + i)
        out = apply_variant(self.variant, image, mask, rng, other)
        return {
            "pixel_values": self.to_tensor(Image.fromarray(out)),
            "label": int(label),
            "mask": torch.from_numpy(mask),
        }
