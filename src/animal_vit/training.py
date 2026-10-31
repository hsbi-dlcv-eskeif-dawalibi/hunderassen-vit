"""Training pipeline (Hugging Face ``Trainer``) shared by all models and milestones.

Pipeline of :func:`run_experiment`::

    seed -> datasets + transforms -> model -> component swaps / attention changes
         -> [teacher for distillation] -> Trainer.train() (selection on validation)
         -> single evaluation on test -> metrics.json, classification_report.csv,
            confusion_matrix.png, model/ (weights for the analysis scripts)

A trained run is restored with :func:`load_run`, which rebuilds the architecture from
the saved ``config.json`` (including all modifications) and loads the weights.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from animal_vit.attention import (
    attention_entropy,
    prune_alternating_heads,
    prune_random_heads,
    set_attention_temperature,
)
from animal_vit.components import apply_components
from animal_vit.config import ExperimentConfig
from animal_vit.data import build_datasets, build_transforms, collate_fn
from animal_vit.lora import apply_lora, merge_lora
from animal_vit.metrics import classification_metrics, make_compute_metrics
from animal_vit.models import count_parameters, load_image_classifier, load_image_processor, set_trainable
from animal_vit.paths import RUNS_DIR, resolve
from animal_vit.reporting import export_run_records, save_predictions, write_run_report
from animal_vit.seed import set_seed

log = logging.getLogger(__name__)

#: Scheduler presets ("fixed" = HF default: linear decay without warmup).
SCHEDULERS = {"fixed": None, "linear": "linear", "cosine": "cosine", "constant": "constant"}


# ---------------------------------------------------------------- Trainer
def _trainer_class():
    from transformers import Trainer

    class ExperimentTrainer(Trainer):
        """``Trainer`` with optional extra loss terms.

        loss = CE                                    (or alpha*CE + (1-alpha)*T^2*KL with a teacher)
               - lambda_out * H(prediction)          (output-entropy regularisation)
               - lambda_att * H(attention)           (attention-entropy regularisation)

        Knowledge distillation follows lecture 03 / Hinton et al. (2015): the student
        matches the temperature-softened teacher distribution via KL divergence.
        """

        def __init__(
            self,
            *args,
            output_entropy_lambda: float = 0.0,
            attention_entropy_lambda: float = 0.0,
            teacher=None,
            teacher_norm=None,
            distill_alpha: float = 0.5,
            distill_temperature: float = 4.0,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.output_entropy_lambda = output_entropy_lambda
            self.attention_entropy_lambda = attention_entropy_lambda
            self.teacher = teacher
            self.teacher_norm = teacher_norm  # converts student-normalised pixels to teacher normalisation
            self.distill_alpha = distill_alpha
            self.distill_temperature = distill_temperature

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Cross entropy, optionally plus distillation (KL to the teacher) and entropy terms."""
            import torch
            import torch.nn.functional as F

            inputs = {k: v for k, v in inputs.items() if k != "mask"}
            # Attention weights only while training; collecting them for every evaluation
            # batch would exhaust memory.
            want_attentions = self.attention_entropy_lambda != 0.0 and model.training
            outputs = model(**inputs, output_attentions=want_attentions) if want_attentions else model(**inputs)
            loss = outputs.loss

            if self.teacher is not None and model.training:
                with torch.no_grad():
                    if next(self.teacher.parameters()).device != inputs["pixel_values"].device:
                        self.teacher.to(inputs["pixel_values"].device)
                    teacher_logits = self.teacher(pixel_values=self.teacher_norm(inputs["pixel_values"])).logits
                t = self.distill_temperature
                kl = F.kl_div(
                    F.log_softmax(outputs.logits / t, dim=-1),
                    F.softmax(teacher_logits / t, dim=-1),
                    reduction="batchmean",
                )
                loss = self.distill_alpha * loss + (1.0 - self.distill_alpha) * (t * t) * kl

            if self.output_entropy_lambda:
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                loss = loss - self.output_entropy_lambda * entropy
            if want_attentions:
                loss = loss - self.attention_entropy_lambda * attention_entropy(outputs.attentions)

            return (loss, outputs) if return_outputs else loss

    return ExperimentTrainer


def build_training_args(cfg: ExperimentConfig, output_dir: Path):
    """Translate the ``training`` section into ``transformers.TrainingArguments``.

    Evaluates and saves once per epoch and keeps the checkpoint with the best validation metric.
    """
    from transformers import TrainingArguments

    t = cfg.training
    if t.strategy not in SCHEDULERS:
        raise ValueError(f"Unknown training.strategy '{t.strategy}', choose from {list(SCHEDULERS)}")
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir / "checkpoints"),
        logging_dir=str(output_dir / "tensorboard"),
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        num_train_epochs=t.epochs,
        learning_rate=t.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.metric_for_best_model not in ("loss", "cross_entropy", "ece"),
        fp16=t.fp16,
        remove_unused_columns=False,
        report_to="tensorboard",
        seed=t.seed,
        # Override with ANIMAL_NUM_WORKERS (e.g. 0 on macOS, 8+ on many-core cloud machines).
        dataloader_num_workers=int(os.environ.get("ANIMAL_NUM_WORKERS", 4)),
    )
    if SCHEDULERS[t.strategy]:
        kwargs["lr_scheduler_type"] = SCHEDULERS[t.strategy]
    kwargs.update(t.extra)
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------- model setup
def prepare_model(cfg: ExperimentConfig, id2label, label2id, pretrained: bool = True):
    """Build the model and apply every configured modification. Returns ``(model, info)``."""
    mods = cfg.modifications
    model_cfg = cfg.model
    if mods.attention_entropy_lambda and model_cfg.attn_implementation != "eager":
        log.info("attention_entropy_lambda set -> switching to eager attention")
        model_cfg.attn_implementation = "eager"

    model = load_image_classifier(model_cfg, id2label, label2id, pretrained=pretrained)
    info: dict[str, Any] = {}

    apply_components(model, mods.positional_embedding, mods.patch_embedding, seed=cfg.training.seed)
    if mods.prune_heads_random:
        info["pruned_heads"] = prune_random_heads(model, mods.prune_heads_random, seed=cfg.training.seed)
    if mods.prune_heads_parity:
        info["pruned_heads"] = prune_alternating_heads(model, mods.prune_heads_parity)
    # After pruning: pruning replaces the query projection, which would drop the wrapper.
    set_attention_temperature(model, mods.attention_temperature)

    if mods.trainable == "lora":
        # Training: insert adapters. Restoring a finished run (pretrained=False): the saved
        # weights already contain the merged update, so the plain model is rebuilt.
        if pretrained:
            info["lora_layers"] = apply_lora(
                model, mods.lora.rank, mods.lora.alpha, mods.lora.dropout, mods.lora.targets
            )
        trainable, total = count_parameters(model)
    else:
        trainable, total = set_trainable(model, mods.trainable)
    info.update(trainable_parameters=trainable, total_parameters=total)
    log.info("Parameters: %s trainable / %s total", f"{trainable:,}", f"{total:,}")
    return model, info


def load_run(run_dir: str | Path, device: str | None = None):
    """Restore a trained run of :func:`run_experiment`.

    Returns ``(model, cfg, class_names, (size, mean, std))`` with the model in eval mode.
    """
    import torch

    run_dir = resolve(run_dir)
    if run_dir.name == "model":
        run_dir = run_dir.parent
    cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    class_names = json.loads((run_dir / "class_names.json").read_text())
    id2label = dict(enumerate(class_names))
    model, _ = prepare_model(cfg, id2label, {v: k for k, v in id2label.items()}, pretrained=False)

    weights = run_dir / "model" / "model.safetensors"
    if weights.exists():
        from safetensors.torch import load_file

        state = load_file(str(weights))
    else:
        state = torch.load(run_dir / "model" / "pytorch_model.bin", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Weights do not match the rebuilt model: missing={missing}, unexpected={unexpected}")

    _, size, mean, std = load_image_processor(cfg.model.checkpoint)
    model.eval()
    if device:
        model.to(device)
    return model, cfg, class_names, (size, mean, std)


def _renormalizer(src_mean, src_std, dst_mean, dst_std):
    """Map pixels normalised with (src_mean, src_std) to (dst_mean, dst_std)."""
    import torch

    def convert(x):
        shape = (1, 3, 1, 1)
        sm, ss = torch.tensor(src_mean, device=x.device).view(shape), torch.tensor(src_std, device=x.device).view(shape)
        dm, ds = torch.tensor(dst_mean, device=x.device).view(shape), torch.tensor(dst_std, device=x.device).view(shape)
        return ((x * ss + sm) - dm) / ds

    return convert


# ---------------------------------------------------------------- evaluation export
def export_evaluation(logits, labels, class_names, output_dir: Path, extra: dict | None = None) -> dict:
    """Write ``confusion_matrix.png``, ``classification_report.csv`` and ``metrics.json``."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report

    from animal_vit.plotting import plot_confusion_matrix

    output_dir.mkdir(parents=True, exist_ok=True)
    preds = np.asarray(logits).argmax(axis=-1)

    plot_confusion_matrix(labels, preds, class_names, output_dir / "confusion_matrix.png")
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")

    metrics = classification_metrics(logits, labels)
    payload = {"test_metrics": metrics, **(extra or {})}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    return metrics


# ---------------------------------------------------------------- full pipeline
def run_experiment(cfg: ExperimentConfig) -> dict:
    """Train, evaluate once on the test split and export all artefacts."""
    from transformers import EarlyStoppingCallback

    set_seed(cfg.training.seed)
    output_dir = resolve(cfg.output_dir) if cfg.output_dir else RUNS_DIR / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    processor, size, mean, std = load_image_processor(cfg.model.checkpoint)
    d = cfg.data
    datasets, class_names = build_datasets(
        d.name,
        build_transforms(size, mean, std, d.augmentation),
        val_fraction=d.val_fraction,
        split_seed=d.split_seed,
        max_samples=d.max_samples,
        download=d.download,
        species=d.species,
        train_fraction=d.train_fraction,
    )
    (output_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    id2label = dict(enumerate(class_names))
    log.info("Data: %s | train %d | val %d | test %d", d.name, *(len(datasets[s]) for s in datasets))

    model, info = prepare_model(cfg, id2label, {v: k for k, v in id2label.items()}, pretrained=cfg.model.pretrained)
    if not cfg.model.pretrained:
        log.info("model.pretrained=false: training %s from random initialisation", cfg.model.checkpoint)

    teacher_kwargs: dict[str, Any] = {}
    if cfg.distillation.teacher:
        teacher, _, teacher_classes, (_, t_mean, t_std) = load_run(cfg.distillation.teacher)
        if teacher_classes != class_names:
            raise ValueError("Teacher was trained on different classes")
        for p in teacher.parameters():
            p.requires_grad = False
        teacher_kwargs = dict(
            teacher=teacher,
            teacher_norm=_renormalizer(mean, std, t_mean, t_std),
            distill_alpha=cfg.distillation.alpha,
            distill_temperature=cfg.distillation.temperature,
        )
        info["teacher"] = cfg.distillation.teacher

    trainer = _trainer_class()(
        model=model,
        args=build_training_args(cfg, output_dir),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],  # model selection on validation only
        data_collator=collate_fn,
        processing_class=processor,
        compute_metrics=make_compute_metrics(),
        callbacks=[EarlyStoppingCallback(cfg.training.early_stopping_patience, cfg.training.early_stopping_threshold)],
        output_entropy_lambda=cfg.modifications.output_entropy_lambda,
        attention_entropy_lambda=cfg.modifications.attention_entropy_lambda,
        **teacher_kwargs,
    )
    train_output = trainer.train()
    # Wall-clock training time: one of the criteria for comparing fine-tuning strategies (MS1).
    info["train_runtime_s"] = train_output.metrics.get("train_runtime")
    if cfg.modifications.trainable == "lora":
        # Fold B A into W: the saved model is a plain Hugging Face model (no extra inference cost,
        # loadable as a teacher in station 2).
        info["lora_merged_layers"] = merge_lora(trainer.model)
    trainer.save_model(str(output_dir / "model"))
    if not cfg.training.keep_checkpoints:
        shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)

    # Detailed records for writing the paper (see animal_vit.reporting).
    epochs = export_run_records(trainer, datasets, class_names, output_dir, info)

    # The test split is touched exactly once, after training and model selection.
    prediction = trainer.predict(datasets["test"])
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    best_val = trainer.state.best_metric
    metrics = export_evaluation(
        logits,
        prediction.label_ids,
        class_names,
        output_dir,
        extra={"experiment": cfg.name, "best_validation_metric": best_val, **info},
    )
    save_predictions(logits, prediction.label_ids, class_names, output_dir / "predictions.csv")
    write_run_report(output_dir, cfg.to_dict(), metrics, info, epochs)
    log.info("Test metrics: %s", json.dumps(metrics, indent=2))
    return metrics
