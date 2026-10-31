"""Parameters, GFLOPs and latency of trained runs (or plain checkpoints).

Writes ``<run>/efficiency.json``. Measure all models on the same machine.

Example::

    python scripts/benchmark.py runs/ms1_vit_b16_s1 runs/ms1_resnet50_s1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    from animal_vit.efficiency import profile
    from animal_vit.evaluation import select_device
    from animal_vit.training import load_run

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    for run in args.runs:
        model, _, _, (size, _, _) = load_run(run, device)
        result = {"device": device, **profile(model, size)}
        (run / "efficiency.json").write_text(json.dumps(result, indent=2))
        print(
            f"{run.name:30s} {result['parameters_m']:6.1f} M params  {result['gflops']:6.2f} GFLOPs  "
            f"{result['latency_ms_bs1']:7.2f} ms (bs 1)"
        )


if __name__ == "__main__":
    main()
