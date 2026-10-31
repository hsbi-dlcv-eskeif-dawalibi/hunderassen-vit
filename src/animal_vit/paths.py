"""Central path definitions.

All scripts resolve files relative to the repository root instead of the current
working directory, so they can be started from anywhere. Locations that differ between
machines (data, runs) can be overridden with environment variables, e.g. on a GPU
cluster where the data lives on a shared volume.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository root (``src/animal_vit/paths.py`` -> three levels up).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Root folder for all datasets (``DATA_ROOT``). Each dataset lives in its own subfolder.
DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

#: Oxford-IIIT Pet (downloaded by ``scripts/prepare_data.py``).
PETS_DIR: Path = DATA_ROOT / "oxford_pets"

#: Generic ImageFolder dataset ``<dir>/{train,validation,test}/<class>/`` (offline tests).
DATA_DIR: Path = Path(os.environ.get("IMAGE_FOLDER_DIR", DATA_ROOT / "image_folder"))

#: Committed results of *this* project (reports, figures, summary tables).
RESULTS_DIR: Path = REPO_ROOT / "results"

#: Model checkpoints. Not committed (see ``.gitignore``).
MODELS_DIR: Path = Path(os.environ.get("ANIMAL_MODELS_DIR", REPO_ROOT / "models"))

#: YAML experiment configurations.
CONFIGS_DIR: Path = REPO_ROOT / "configs"

#: Fresh training runs write here by default (ignored by git).
RUNS_DIR: Path = Path(os.environ.get("ANIMAL_RUNS_DIR", REPO_ROOT / "runs"))

SPLITS = ("train", "validation", "test")


def resolve(path: str | os.PathLike, base: Path = REPO_ROOT) -> Path:
    """Return ``path`` as an absolute path, interpreting relative paths against ``base``."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (base / candidate)


def require_data_dir(data_dir: Path = DATA_DIR) -> Path:
    """Fail early with a helpful message if an ImageFolder dataset is missing."""
    missing = [split for split in SPLITS if not (data_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Dataset not found or incomplete at '{data_dir}' (missing: {', '.join(missing)}).\n"
            "See docs/TECHNICAL.md#3-datensatz."
        )
    return data_dir
