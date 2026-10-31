"""Experiment configuration.

Every experiment is described by one YAML file in ``configs/``; nothing is hard-coded
in notebooks.

Example (``configs/extra/pos_embedding_none.yaml``)::

    name: extra_vit_pos_none
    model:
      checkpoint: google/vit-base-patch16-224
    modifications:
      positional_embedding: none
    training:
      epochs: 10

Values can be overridden on the command line with dotted keys::

    python scripts/train.py configs/ms1/vit_b16.yaml --set training.seed=2 name=vit_b16_s2
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Which pretrained model to fine-tune (``model:`` section of a config)."""

    #: Hugging Face checkpoint, e.g. ``google/vit-base-patch16-224``.
    checkpoint: str = "google/vit-base-patch16-224"
    #: Override the patch size. Anything other than the pretrained size re-initialises
    #: the patch projection (see docs/CONTEXT.md, "Bekannte Probleme").
    patch_size: int | None = None
    #: Interpolate position embeddings when the token grid changes.
    interpolate_pos_encoding: bool = False
    #: ``"eager"`` is required whenever attention weights are needed.
    attn_implementation: str | None = None
    hidden_dropout_prob: float | None = None
    #: ``False`` trains the architecture from random initialisation (``make why``: how much of
    #: the accuracy comes from pretraining?).
    pretrained: bool = True


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation (station 1), used when ``modifications.trainable`` is ``"lora"``."""

    #: Rank r of the update B A. Station 1 compares several ranks (4, 8, 16).
    rank: int = 8
    #: Scaling alpha; the update is multiplied by alpha / r.
    alpha: float = 16.0
    #: Dropout on the LoRA input path.
    dropout: float = 0.1
    #: ``"qv"`` (query + value, as in the LoRA paper), ``"all"`` (every linear layer of a block)
    #: or a list of module-name suffixes.
    targets: Any = "qv"


@dataclass
class ModificationConfig:
    """Changes to the model: fine-tuning strategy (MS1), component swaps (MS2), attention tweaks."""

    #: Attention temperature T in softmax(QK^T / (sqrt(d_k) * T)); 1.0 = unchanged.
    attention_temperature: float = 1.0
    #: Number of heads to prune per layer, chosen at random (seeded).
    prune_heads_random: int = 0
    #: Prune every other head per layer: ``"even"`` or ``"odd"``.
    prune_heads_parity: str | None = None
    #: lambda for the *output* (prediction) entropy term: loss = CE - lambda * H(p).
    #: This is what the original "entropy regularization" notebook implemented.
    output_entropy_lambda: float = 0.0
    #: lambda for a real *attention* entropy term: loss = CE - lambda * H(attention).
    attention_entropy_lambda: float = 0.0
    #: Which parameters to train (station 1): ``"head"`` (linear probing), ``"lora"`` (LoRA adapters
    #: + head, see ``lora``), ``"last_n:<k>"`` (last k blocks + head) or ``"all"`` (full fine-tuning).
    trainable: str = "all"
    #: LoRA settings, only used with ``trainable: lora``.
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    #: MS2 component swap - position embeddings: ``"learned"`` (pretrained, default),
    #: ``"none"`` (zeroed + frozen: the encoder sees a bag of patches) or ``"shuffled"``
    #: (pretrained vectors assigned to random positions).
    positional_embedding: str = "learned"
    #: MS2 component swap - patch embedding: ``"linear"`` (pretrained 16x16 projection)
    #: or ``"conv_stem"`` (small convolutional stem, Xiao et al. 2021, trained from scratch).
    patch_embedding: str = "linear"


@dataclass
class DistillationConfig:
    """Knowledge distillation from a trained teacher run (MS3, lecture 03)."""

    #: Path to a trained teacher (run directory of scripts/train.py, i.e. ``runs/<name>``
    #: or its ``model/`` subfolder). ``None`` disables distillation.
    teacher: str | None = None
    #: Weight of the hard-label loss: L = alpha * CE + (1 - alpha) * T^2 * KL (lecture 03).
    alpha: float = 0.5
    #: Softmax temperature for the soft targets.
    temperature: float = 4.0


@dataclass
class DataConfig:
    """Dataset, class subset (species), augmentation and train/validation split."""

    #: ``"oxford_pets"`` (default) or ``"image_folder"`` (generic ImageFolder, offline tests).
    name: str = "oxford_pets"
    #: Oxford-IIIT Pet only: ``"dogs"`` (25 breeds, project default), ``"cats"`` (12) or ``"all"`` (37).
    #: One species makes the task fine-grained (recommendation of the lecturer).
    species: str = "dogs"
    #: Augmentation preset for the training split: ``"none"``, ``"light"`` or ``"strong"``.
    augmentation: str = "light"
    #: Fraction of the official trainval split held out for validation (stratified).
    val_fraction: float = 0.1
    #: Seed of the train/validation split. Kept fixed across training seeds so that all
    #: runs are selected on the same validation images.
    split_seed: int = 0
    #: Share of the training split that is used (stratified, fixed by ``split_seed``). Station 2
    #: uses 0.25 / 0.5 to test whether distillation helps more with little data. Validation and
    #: test are never reduced.
    train_fraction: float = 1.0
    #: Optional cap on samples per split for quick debugging runs.
    max_samples: int | None = None
    #: Download the dataset if it is missing (Oxford-IIIT Pet only).
    download: bool = True


@dataclass
class TrainingConfig:
    """Optimisation settings passed to the Hugging Face ``Trainer``."""

    #: Scheduler preset: ``"fixed"`` (HF default: linear decay), ``"linear"``,
    #: ``"cosine"``, ``"constant"``.
    strategy: str = "fixed"
    epochs: float = 10
    learning_rate: float = 5e-5
    batch_size: int = 32
    seed: int = 123
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    #: Model selection metric on the validation set.
    metric_for_best_model: str = "accuracy"
    fp16: bool = True
    #: Keep the Trainer checkpoints (weights + optimizer state, ~1 GB for ViT-B) after training.
    #: Off by default: the selected model is saved to ``model/`` anyway, and on cloud GPUs
    #: (RunPod) disk space is limited.
    keep_checkpoints: bool = False
    #: Additional ``transformers.TrainingArguments`` (weight_decay, warmup_steps, ...).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """One complete experiment, loaded from ``configs/*.yaml`` (see docs/TECHNICAL.md)."""

    name: str = "experiment"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    modifications: ModificationConfig = field(default_factory=ModificationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    #: Output directory; defaults to ``runs/<name>``.
    output_dir: str | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        """Build a config from a nested dict; unknown keys raise an error (typo safety)."""
        return _build(cls, raw)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
        """Load a YAML config and apply ``key=value`` overrides (``--set`` on the command line)."""
        import yaml

        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Nested plain dict, written to ``runs/<name>/config.json`` for reproducibility."""
        return asdict(self)


# ---------------------------------------------------------------------- helpers
def _build(cls: type, raw: dict[str, Any]):
    """Recursively build nested dataclasses and reject unknown keys (typo safety)."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        else:
            kwargs[name] = _coerce(value, known[name].type)
    return cls(**kwargs)


def _coerce(value: Any, annotation: Any) -> Any:
    """Fix values whose type YAML / the override parser cannot know.

    - ``2e-4`` is read as a string by PyYAML: numeric strings become floats for float fields.
    - ``none`` (YAML and ``--set``) is read as ``None``: for plain ``str`` fields such as
      ``data.augmentation`` it means the string ``"none"``.
    """
    if value is None and str(annotation) == "str":
        return "none"
    if isinstance(value, str) and "float" in str(annotation):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _apply_override(raw: dict[str, Any], item: str) -> None:
    """Apply a ``a.b.c=value`` override to a nested dict."""
    if "=" not in item:
        raise ValueError(f"Override must look like key=value, got '{item}'")
    key, value = item.split("=", 1)
    try:
        parsed: Any = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = {"none": None, "null": None, "true": True, "false": False}.get(value.lower(), value)
    node = raw
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = parsed
