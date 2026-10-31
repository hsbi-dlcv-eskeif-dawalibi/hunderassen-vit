"""Species filter of Oxford-IIIT Pet (``data.species``): no download, uses the class list only."""

from __future__ import annotations

import pytest

from animal_vit.data import species_classes

# Class names as produced by torchvision's OxfordIIITPet (12 cats, 25 dogs, alphabetical).
PET_CLASSES = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]  # fmt: skip


def test_dogs_and_cats_partition_all_classes():
    dogs = species_classes(PET_CLASSES, "dogs")
    cats = species_classes(PET_CLASSES, "cats")
    assert len(dogs) == 25
    assert len(cats) == 12
    assert sorted(dogs + cats) == list(range(37))
    assert "Beagle" in [PET_CLASSES[i] for i in dogs]
    assert "Persian" in [PET_CLASSES[i] for i in cats]


def test_all_keeps_every_class_in_order():
    assert species_classes(PET_CLASSES, "all") == list(range(37))


def test_underscore_names_are_matched():
    raw = [name.replace(" ", "_") for name in PET_CLASSES]
    assert len(species_classes(raw, "cats")) == 12


def test_unknown_species_is_rejected():
    with pytest.raises(ValueError):
        species_classes(PET_CLASSES, "birds")
