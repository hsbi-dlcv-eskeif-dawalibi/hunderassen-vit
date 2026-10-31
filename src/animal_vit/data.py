"""Datasets, transforms and batching.

Supported datasets (``data.name`` in the config):

``oxford_pets`` (default)
    Oxford-IIIT Pet: 37 cat and dog breeds, ~7.4k images, CC BY-SA 4.0
    (Parkhi et al., CVPR 2012). The project uses only the **25 dog breeds**
    (``data.species: dogs``, ~2.5k trainval / ~2.5k test images): breeds of one species
    look alike, which makes the task fine-grained (recommendation of the lecturer).
    The official ``trainval`` split is divided into train/validation with a seeded,
    stratified split; the official ``test`` split is used only for the final
    evaluation. Every image comes with a
    *trimap* (1 = pet, 2 = background, 3 = border), which the analysis code uses to
    measure whether a model looks at the animal or at the background.

``image_folder``
    Any ImageFolder directory ``<dir>/{train,validation,test}/<class>/`` (``IMAGE_FOLDER_DIR``,
    default ``data/image_folder``). Used by the offline tests; it has no masks.

All datasets return dicts ``{"pixel_values": Tensor, "label": int}``; with
``with_mask=True`` an additional ``"mask"`` (bool tensor, True = pet) is returned,
aligned with the image after the evaluation geometry (resize + center crop).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from animal_vit.paths import DATA_DIR, PETS_DIR, require_data_dir

DATASETS = ("oxford_pets", "image_folder")
AUGMENTATIONS = ("none", "light", "strong")

#: ImageNet statistics (CNN checkpoints, style transfer).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Trimap values of Oxford-IIIT Pet.
TRIMAP_PET, TRIMAP_BACKGROUND, TRIMAP_BORDER = 1, 2, 3


# --------------------------------------------------------------------- transforms
def build_transforms(size: int, mean: Sequence[float], std: Sequence[float], augmentation: str = "light"):
    """Return ``{"train", "eval"}`` torchvision transforms working on PIL images.

    - ``none``:   resize + center crop (identical to evaluation)
    - ``light``:  random resized crop (scale 0.6-1) + horizontal flip. Project default:
      plausible transformations only (lecture 01) and not "too much" augmentation for
      fine-tuning a pretrained model (lecture 02, "Strategien").
    - ``strong``: ``light`` + RandAugment (Cubuk et al., CVPRW 2020) + Random Erasing, i.e.
      CutOut (DeVries & Taylor 2017; Zhong et al., AAAI 2020). AugReg-style augmentation for
      small datasets (Steiner et al., TMLR 2022, lecture 03).

    Dogs have a meaningful up/down orientation, so there is no vertical flip and no hue
    jitter: fur colour is a breed cue.
    Station 1 compares the three presets (``make ms1-augment``).
    """
    from torchvision import transforms as T

    if augmentation not in AUGMENTATIONS:
        raise ValueError(f"augmentation must be one of {AUGMENTATIONS}, got '{augmentation}'")
    to_tensor = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    evaluation = T.Compose([T.Resize(size), T.CenterCrop(size), *to_tensor])
    if augmentation == "none":
        train = evaluation
    else:
        steps = [T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
        after_tensor = []
        if augmentation == "strong":
            # RandAugment already contains brightness/contrast/colour ops, so no extra colour jitter.
            steps += [T.RandAugment(num_ops=2, magnitude=9)]
            after_tensor = [T.RandomErasing(p=0.25, scale=(0.02, 0.2), value=0)]  # CutOut
        train = T.Compose([*steps, *to_tensor, *after_tensor])
    return {"train": train, "eval": evaluation}


class EvalGeometry:
    """Resize + center crop applied identically to images and masks (mask: nearest).

    A class instead of a closure, so that datasets using it can be sent to DataLoader worker
    processes (spawn start method on macOS and Windows pickles the dataset).
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, mask=None):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

        image = F.center_crop(F.resize(image, self.size), self.size)
        if mask is not None:
            mask = F.center_crop(F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST), self.size)
        return image, mask


def eval_geometry(size: int) -> EvalGeometry:
    """Resize + center crop for images and masks (see :class:`EvalGeometry`)."""
    return EvalGeometry(size)


def normalize_tensor(mean: Sequence[float], std: Sequence[float]):
    """PIL image -> normalised tensor (no resizing), for images that already have the model size."""
    from torchvision import transforms as T

    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def trimap_to_pet_mask(trimap) -> np.ndarray:
    """Boolean mask, True where the pet is (pet + border pixels)."""
    arr = np.asarray(trimap)
    return arr != TRIMAP_BACKGROUND


# ---------------------------------------------------------------------- splitting
def stratified_split(labels: Sequence[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split indices into (train, validation) with the same class proportions."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


# ----------------------------------------------------------------------- datasets
class ClassificationDataset:
    """Wrap a sample source into ``{"pixel_values", "label"[, "mask"]}`` dicts.

    ``load(i)`` returns ``(PIL image, label, PIL trimap or None)``.
    """

    def __init__(
        self,
        load: Callable[[int], tuple],
        indices: Sequence[int],
        transform: Callable,
        class_names: Sequence[str],
        with_mask: bool = False,
        size: int | None = None,
    ):
        self.load, self.indices, self.transform = load, list(indices), transform
        self.class_names = list(class_names)
        self.with_mask = with_mask
        self._geometry = eval_geometry(size) if with_mask else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        image, label, trimap = self.load(self.indices[i])
        image = image.convert("RGB")
        item: dict[str, Any] = {"label": int(label)}
        if self.with_mask:
            image, trimap = self._geometry(image, trimap)
            item["mask"] = torch.from_numpy(trimap_to_pet_mask(trimap))
        item["pixel_values"] = self.transform(image)
        return item

    @property
    def labels(self) -> list[int]:
        """Labels of this split without decoding images."""
        return [self.load.labels[i] for i in self.indices]


#: The 12 cat breeds of Oxford-IIIT Pet; the remaining 25 breeds are dogs
#: (Parkhi et al., CVPR 2012, Table 1). Stored normalised (lower case, no separators).
CAT_BREEDS = frozenset(
    {
        "abyssinian",
        "bengal",
        "birman",
        "bombay",
        "britishshorthair",
        "egyptianmau",
        "mainecoon",
        "persian",
        "ragdoll",
        "russianblue",
        "siamese",
        "sphynx",
    }
)
#: Allowed values of ``data.species``.
SPECIES = ("dogs", "cats", "all")


def _breed_key(name: str) -> str:
    """Normalise a breed name for matching (``"British Shorthair"`` -> ``"britishshorthair"``)."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def species_classes(class_names: Sequence[str], species: str) -> list[int]:
    """Indices of the Oxford-IIIT Pet classes that belong to ``species``.

    The professor recommended a single species to make the task fine-grained (breeds of one
    animal look alike), so the project uses ``"dogs"`` (25 breeds) by default.
    """
    if species not in SPECIES:
        raise ValueError(f"Unknown species '{species}', choose from {SPECIES}")
    if species == "all":
        return list(range(len(class_names)))
    is_cat = [_breed_key(name) in CAT_BREEDS for name in class_names]
    if sum(is_cat) != len(CAT_BREEDS):
        raise RuntimeError("Could not identify the 12 cat breeds in the class list")
    want_cat = species == "cats"
    return [i for i, cat in enumerate(is_cat) if cat == want_cat]


class _PetsSource:
    """Picklable sample loader for Oxford-IIIT Pet (DataLoader workers use spawn on macOS).

    With ``species`` other than ``"all"`` only the breeds of that species are kept and their
    labels are renumbered to ``0 .. K-1`` (in the original alphabetical order). Index ``i``
    then refers to the ``i``-th *kept* image.
    """

    def __init__(self, root: Path, split: str, download: bool, species: str = "dogs"):
        from torchvision.datasets import OxfordIIITPet

        self.base = OxfordIIITPet(str(root), split=split, target_types=("category", "segmentation"), download=download)
        all_names = [name.replace("_", " ") for name in self.base.classes]
        kept = species_classes(all_names, species)
        self.label_map = {old: new for new, old in enumerate(kept)}
        self.class_names = [all_names[i] for i in kept]
        base_labels = list(self.base._labels)  # labels without decoding images
        self.positions = [pos for pos, label in enumerate(base_labels) if label in self.label_map]
        self.labels = [self.label_map[base_labels[pos]] for pos in self.positions]

    def __call__(self, i: int):
        image, (label, trimap) = self.base[self.positions[i]]
        return image, self.label_map[label], trimap


class _FolderSource:
    """Picklable sample loader for an ImageFolder split (no masks)."""

    def __init__(self, root: Path):
        from torchvision.datasets import ImageFolder

        self.base = ImageFolder(str(root))
        self.class_names = list(self.base.classes)
        self.labels = list(self.base.targets)

    def __call__(self, i: int):
        image, label = self.base[i]
        return image, label, None


def build_datasets(
    name: str,
    transforms: dict,
    val_fraction: float = 0.1,
    split_seed: int = 0,
    max_samples: int | None = None,
    download: bool = False,
    with_mask: bool = False,
    size: int | None = None,
    species: str = "dogs",
    train_fraction: float = 1.0,
) -> tuple[dict[str, ClassificationDataset], list[str]]:
    """Return ``({"train", "validation", "test"}: dataset, class_names)``.

    ``species`` (Oxford-IIIT Pet only) selects ``"dogs"`` (25 breeds), ``"cats"`` (12) or ``"all"`` (37).
    """
    if name == "oxford_pets":
        trainval = _PetsSource(PETS_DIR, "trainval", download, species)
        test = _PetsSource(PETS_DIR, "test", download, species)
        class_names = trainval.class_names
        train_idx, val_idx = stratified_split(trainval.labels, val_fraction, split_seed)
        sources = {
            "train": (trainval, train_idx),
            "validation": (trainval, val_idx),
            "test": (test, list(range(len(test.labels)))),
        }
    elif name == "image_folder":
        require_data_dir(DATA_DIR)
        sources = {}
        for split in ("train", "validation", "test"):
            source = _FolderSource(DATA_DIR / split)
            class_names = source.class_names
            sources[split] = (source, list(range(len(source.labels))))
        if with_mask:
            raise ValueError("image_folder has no segmentation masks")
    else:
        raise ValueError(f"Unknown dataset '{name}', choose from {DATASETS}")

    if not 0 < train_fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
    if train_fraction < 1:
        # Stratified subset of the training split; the complement is simply not used.
        load, indices = sources["train"]
        labels = [load.labels[i] for i in indices]
        keep, _ = stratified_split(labels, 1 - train_fraction, split_seed)
        sources["train"] = (load, [indices[i] for i in keep])

    datasets = {}
    for split, (load, indices) in sources.items():
        if max_samples:
            indices = np.random.default_rng(0).permutation(indices)[:max_samples].tolist()
        transform = transforms["train"] if split == "train" else transforms["eval"]
        datasets[split] = ClassificationDataset(
            load, indices, transform, class_names, with_mask=with_mask and split != "train", size=size
        )
    return datasets, class_names


def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of dataset items into a batch for the Hugging Face ``Trainer``."""
    import torch

    batch = {
        "pixel_values": torch.stack([ex["pixel_values"] for ex in examples]),
        "labels": torch.tensor([ex["label"] for ex in examples]),
    }
    if "mask" in examples[0]:
        batch["mask"] = torch.stack([ex["mask"] for ex in examples])
    return batch
