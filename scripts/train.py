"""Train and evaluate any model (CNN or transformer) from a YAML config.

Examples::

    # one run
    python scripts/train.py configs/ms1/vit_b16.yaml

    # three seeds (every reported number is mean +- std over seeds)
    for s in 1 2 3; do
        python scripts/train.py configs/ms1/vit_b16.yaml --set training.seed=$s name=ms1_vit_b16_s$s
    done

    # pipeline smoke test (tiny subset, CPU friendly)
    python scripts/train.py configs/debug.yaml

Outputs in ``runs/<name>/``: ``config.json``, ``class_names.json``, ``metrics.json``,
``classification_report.csv``, ``confusion_matrix.png``, ``model/`` (weights for the
analysis scripts), ``tensorboard/`` and ``checkpoints/``.
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from animal_vit.config import ExperimentConfig
from animal_vit.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="Path to an experiment YAML file")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. training.epochs=10"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_experiment(ExperimentConfig.from_yaml(args.config, args.set))


if __name__ == "__main__":
    main()
