"""Evaluate trained runs on background / colour / patch-shuffle variants of the test set.

For every run and variant: accuracy and macro-F1 on the test split with that variant
applied (see ``animal_vit.variants``). Writes ``<run>/variants.csv`` and prints a table.

Example::

    python scripts/evaluate_variants.py runs/ms1_vit_b16_s1 runs/ms1_resnet50_s1
    python scripts/evaluate_variants.py runs/ms1_* --variants original background mixed_random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from animal_vit.variants import VARIANTS


def main() -> None:
    import numpy as np
    import pandas as pd

    from animal_vit.evaluation import predict, select_device, test_dataset
    from animal_vit.metrics import classification_metrics
    from animal_vit.training import load_run
    from animal_vit.variants import VariantDataset

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None, help="evaluate on a subset (debugging)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    rows = []
    for run in args.runs:
        model, cfg, _, (size, mean, std) = load_run(run, device)
        base, _ = test_dataset(cfg, size, mean, std, with_mask=True)
        if args.max_samples:
            # Random subset (fixed seed): the test split is sorted by class.
            base.indices = sorted(np.random.default_rng(0).permutation(base.indices)[: args.max_samples].tolist())
        run_rows = []
        for variant in args.variants:
            logits, labels = predict(model, VariantDataset(base, variant, size, mean, std), device, args.batch_size)
            m = classification_metrics(logits, labels)
            run_rows.append({"run": run.name, "variant": variant, "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
            print(f"{run.name:30s} {variant:16s} acc {m['accuracy']:.4f}  F1m {m['f1_macro']:.4f}")
        pd.DataFrame(run_rows).to_csv(run / "variants.csv", index=False)
        rows += run_rows

    table = pd.DataFrame(rows).pivot(index="run", columns="variant", values="accuracy")
    print("\naccuracy per variant\n", table.round(4).to_string())


if __name__ == "__main__":
    main()
