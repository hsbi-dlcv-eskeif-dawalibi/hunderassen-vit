"""Model loading.

Every model is a Hugging Face image classifier loaded through ``AutoConfig`` /
``AutoModelForImageClassification``, so CNNs and transformers share one pipeline:

=================  ============================================  ==========
Model              Checkpoint                                    Role
=================  ============================================  ==========
ResNet-50          ``microsoft/resnet-50``                       CNN (MS1)
ConvNeXt-T         ``facebook/convnext-tiny-224``                CNN (MS1), teacher (MS3)
ViT-B/16           ``google/vit-base-patch16-224``               ViT (MS1, MS2)
DeiT-S/16          ``facebook/deit-small-patch16-224``           ViT variant (MS2, MS3)
Swin-T             ``microsoft/swin-tiny-patch4-window7-224``    hierarchical ViT (MS3)
=================  ============================================  ==========

"""

from __future__ import annotations

from typing import Any

from animal_vit.config import ModelConfig


def load_image_processor(checkpoint: str):
    """Return ``(processor, size, mean, std)`` for a Hugging Face checkpoint."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(checkpoint, do_rescale=False)
    size = processor.size.get("height") or processor.size.get("shortest_edge") or 224
    return processor, int(size), processor.image_mean, processor.image_std


def load_image_classifier(
    cfg: ModelConfig, id2label: dict[int, str], label2id: dict[str, int], pretrained: bool = True
):
    """Load a pretrained model and replace its head with a ``len(id2label)``-class classifier.

    With ``pretrained=False`` only the architecture is built (used when restoring a
    trained run, whose weights are loaded afterwards).
    """
    from transformers import AutoConfig, AutoModelForImageClassification

    config = AutoConfig.from_pretrained(cfg.checkpoint)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id
    if cfg.patch_size is not None:
        config.patch_size = cfg.patch_size
    if cfg.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = cfg.hidden_dropout_prob
    if cfg.attn_implementation is not None:
        config._attn_implementation = cfg.attn_implementation

    if not pretrained:
        model = AutoModelForImageClassification.from_config(config)
    else:
        model = AutoModelForImageClassification.from_pretrained(
            cfg.checkpoint,
            config=config,
            # The 1000/21k-class head is replaced by a new one for our classes.
            ignore_mismatched_sizes=True,
        )
    if cfg.interpolate_pos_encoding:
        enable_pos_encoding_interpolation(model)
    return model


def enable_pos_encoding_interpolation(model) -> None:
    """Make every forward pass interpolate the position embeddings.

    Swaps the instance class for a subclass whose ``forward`` passes
    ``interpolate_pos_encoding=True``. Works for any model that accepts that argument.
    """
    base = type(model)

    class _WithPosInterpolation(base):  # type: ignore[misc, valid-type]
        def forward(self, pixel_values=None, labels=None, **kwargs):
            kwargs.pop("interpolate_pos_encoding", None)
            return super().forward(pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=True, **kwargs)

    _WithPosInterpolation.__name__ = f"{base.__name__}WithPosInterpolation"
    model.__class__ = _WithPosInterpolation


def model_family(model) -> str:
    """``"vit"`` for ViT/DeiT/BEiT (global attention over a token sequence), ``"swin"``, or ``"cnn"``."""
    prefix = model.base_model_prefix
    if prefix in ("vit", "deit", "beit"):
        return "vit"
    if prefix == "swin":
        return "swin"
    return "cnn"


def backbone(model):
    """Return the backbone (``model.vit``, ``model.resnet``, ``model.swin``, ...)."""
    return getattr(model, model.base_model_prefix)


def encoder_layers(model) -> list[Any]:
    """The transformer blocks of a ViT-like model, in order (used by ``last_n:k`` fine-tuning)."""
    return list(backbone(model).encoder.layer)


def set_trainable(model, mode: str = "all") -> tuple[int, int]:
    """Freeze parameters according to ``mode`` and return ``(trainable, total)``.

    - ``"all"``       full fine-tuning (default in every original experiment)
    - ``"head"``      classifier only (lecture 03: strategy for < 10k images)
    - ``"last_n:k"``  classifier + last ``k`` encoder blocks
    """
    if mode != "all":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        if mode.startswith("last_n:"):
            k = int(mode.split(":", 1)[1])
            for block in encoder_layers(model)[-k:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif mode != "head":
            raise ValueError(f"Unknown trainable mode '{mode}'")
    return count_parameters(model)


def count_parameters(model) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts of ``model``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
