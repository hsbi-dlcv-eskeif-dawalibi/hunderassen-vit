"""Detailed per-run records, so that the paper can be written from the files alone.

Every training run (``runs/<name>/``) gets, in addition to ``metrics.json``:

``data_stats.json``    number of images per split and per class
``history.json``       the complete training log (loss, validation metrics per epoch)
``learning_curves.png``training loss and validation loss/accuracy over the epochs
``predictions.csv``    test prediction per image (true class, predicted class, confidence)
``environment.json``   hardware and library versions (reproducibility)
``README.md``          one-page human-readable summary of the run

``scripts/report_ms1.py`` combines these files of all station-1 runs into one report.
"""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any


def data_stats(datasets: dict, class_names: list[str]) -> dict[str, Any]:
    """Images per split and per class (for the dataset table and the class-balance check)."""
    stats: dict[str, Any] = {"classes": len(class_names), "splits": {}}
    for split, dataset in datasets.items():
        counts = Counter(dataset.labels)
        stats["splits"][split] = {
            "images": len(dataset),
            "per_class": {class_names[k]: int(counts.get(k, 0)) for k in range(len(class_names))},
        }
    return stats


def environment() -> dict[str, Any]:
    """Library versions and the compute device of this run."""
    import torch
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "device": "cpu",
    }
    if torch.cuda.is_available():
        env["device"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        env["device"] = "mps"
    return env


def epoch_table(log_history: list[dict]) -> list[dict[str, float]]:
    """One row per evaluated epoch: epoch, mean training loss of that epoch, validation metrics."""
    train_losses: dict[int, list[float]] = {}
    rows = []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_losses.setdefault(int(-(-entry["epoch"] // 1)), []).append(entry["loss"])
        if "eval_loss" in entry:
            epoch = round(entry["epoch"])
            losses = train_losses.get(epoch, [])
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses) if losses else None,
                    "val_loss": entry.get("eval_loss"),
                    "val_accuracy": entry.get("eval_accuracy"),
                    "val_f1_macro": entry.get("eval_f1_macro"),
                }
            )
    return rows


def plot_learning_curves(rows: list[dict], out_path: Path, title: str) -> Path:
    """Loss (train/val) and validation accuracy per epoch, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["epoch"] for r in rows]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.2))
    train = [(r["epoch"], r["train_loss"]) for r in rows if r["train_loss"] is not None]
    if train:
        ax_loss.plot(*zip(*train), marker="o", label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in rows], marker="o", label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross entropy", title="Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [100 * (r["val_accuracy"] or 0) for r in rows], marker="o", color="tab:green")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Validation accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def save_predictions(logits, labels, class_names: list[str], out_path: Path) -> Path:
    """Per test image: true class, predicted class, softmax confidence, correct or not."""
    import numpy as np
    import pandas as pd

    logits = np.asarray(logits, dtype=np.float64)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "index": np.arange(len(labels)),
            "true": [class_names[i] for i in labels],
            "predicted": [class_names[i] for i in preds],
            "confidence": probs.max(axis=1).round(4),
            "correct": preds == np.asarray(labels),
        }
    )
    frame.to_csv(out_path, index=False)
    return out_path


def write_run_report(output_dir: Path, cfg_dict: dict, metrics: dict, info: dict, rows: list[dict]) -> Path:
    """``README.md`` of a run: setup, cost, best epoch and test metrics on one page."""
    mods = cfg_dict["modifications"]
    method = mods["trainable"]
    if method == "lora":
        lora = mods["lora"]
        method += f" (rank {lora['rank']}, alpha {lora['alpha']}, targets {lora['targets']})"
    trainable, total = info.get("trainable_parameters"), info.get("total_parameters")
    best = max(rows, key=lambda r: r["val_accuracy"] or 0) if rows else {}
    t = cfg_dict["training"]
    lines = [
        f"# Run `{cfg_dict['name']}`",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{cfg_dict['model']['checkpoint']}` |",
        f"| Adaptation | {method} |",
        f"| Data | {cfg_dict['data']['name']}, species `{cfg_dict['data'].get('species')}`, "
        f"augmentation `{cfg_dict['data']['augmentation']}` |",
        f"| Images train / val / test | {info.get('n_train')} / {info.get('n_validation')} / {info.get('n_test')} |",
        f"| Optimiser | AdamW, lr {t['learning_rate']}, {t['strategy']} schedule, batch {t['batch_size']}, "
        f"{t['epochs']} epochs, seed {t['seed']} |",
    ]
    if trainable and total:
        lines.append(f"| Trainable parameters | {trainable:,} of {total:,} ({100 * trainable / total:.2f} %) |")
    if info.get("train_runtime_s") is not None:
        lines.append(f"| Training time | {info['train_runtime_s'] / 60:.1f} min |")
    if best:
        lines.append(f"| Best epoch (validation) | {best['epoch']} of {len(rows)} trained |")
    lines += [
        f"| **Test accuracy** | **{100 * metrics['accuracy']:.2f} %** |",
        f"| Test macro-F1 | {100 * metrics['f1_macro']:.2f} % |",
        f"| Test ECE | {metrics.get('ece', float('nan')):.4f} |",
        "",
        "## Epochs",
        "",
        "| Epoch | Train loss | Val loss | Val acc (%) | Val macro-F1 (%) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:

        def pct(value):
            return "-" if value is None else f"{100 * value:.2f}"

        train_loss = "-" if r["train_loss"] is None else f"{r['train_loss']:.4f}"
        lines.append(
            f"| {r['epoch']} | {train_loss} | {r['val_loss']:.4f} | "
            f"{pct(r['val_accuracy'])} | {pct(r['val_f1_macro'])} |"
        )
    lines += [
        "",
        "Files: `learning_curves.png`, `confusion_matrix.png`, `classification_report.csv` (per breed),",
        "`predictions.csv` (per image), `data_stats.json`, `history.json`, `environment.json`, `config.json`.",
    ]
    out = output_dir / "README.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def export_run_records(trainer, datasets: dict, class_names: list[str], output_dir: Path, info: dict) -> list[dict]:
    """Write data stats, history, learning curves and environment; add counts to ``info``."""
    stats = data_stats(datasets, class_names)
    (output_dir / "data_stats.json").write_text(json.dumps(stats, indent=2))
    for split, values in stats["splits"].items():
        info[f"n_{split}"] = values["images"]

    history = trainer.state.log_history
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    rows = epoch_table(history)
    if rows:
        plot_learning_curves(rows, output_dir / "learning_curves.png", output_dir.name)
        best = max(rows, key=lambda r: r["val_accuracy"] or 0)
        info["best_epoch"] = best["epoch"]
        info["epochs_trained"] = len(rows)
    (output_dir / "environment.json").write_text(json.dumps(environment(), indent=2))
    return rows
