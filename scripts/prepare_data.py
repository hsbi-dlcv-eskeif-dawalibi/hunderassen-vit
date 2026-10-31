"""Download Oxford-IIIT Pet (images + trimaps) into ``data/oxford_pets``.

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/ (Parkhi et al., CVPR 2012),
license CC BY-SA 4.0. Downloaded through ``torchvision.datasets.OxfordIIITPet`` (~800 MB).

Example::

    python scripts/prepare_data.py
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from animal_vit.paths import PETS_DIR


def main() -> None:
    from torchvision.datasets import OxfordIIITPet

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    from animal_vit.data import _PetsSource

    for split in ("trainval", "test"):
        ds = OxfordIIITPet(str(PETS_DIR), split=split, target_types=("category", "segmentation"), download=True)
        dogs = _PetsSource(PETS_DIR, split, download=False, species="dogs")
        print(
            f"{split:9s} {len(ds):5d} images, {len(ds.classes)} classes"
            f" | used: {len(dogs.labels)} dog images, 25 breeds"
        )
    print(f"ready: {PETS_DIR}")


if __name__ == "__main__":
    main()
