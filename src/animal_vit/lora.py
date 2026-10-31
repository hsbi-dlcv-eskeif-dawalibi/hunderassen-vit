"""Low-Rank Adaptation (LoRA) for the linear layers of a Vision Transformer (station 1).

Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

Idea: the pretrained weight ``W`` (``d_out x d_in``) stays frozen. Only a low-rank update
``delta W = B A`` is learned, with ``A`` (``r x d_in``) and ``B`` (``d_out x r``) and a rank
``r << min(d_in, d_out)``::

    h = W x + (alpha / r) * B A x

``B`` starts at zero, so training starts exactly from the pretrained model. With ``r = 8`` on the
query and value projections of ViT-B/16 only about 0.3 M of 86 M parameters are trained
(plus the new classifier head).

After training the update is *merged* into ``W`` (:func:`merge_lora`): the saved model is a
plain Hugging Face model without extra layers and without extra inference cost. This is also
what lets station 2 load a LoRA model as a teacher like any other run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Linear layers inside a ViT encoder block (Hugging Face naming).
#: ``attention.attention.{query,key,value}``, ``attention.output.dense`` (attention output projection),
#: ``intermediate.dense`` and ``output.dense`` (the two MLP layers).
ATTENTION_QV = ("query", "value")
ALL_LINEAR = ("query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense")
TARGET_PRESETS = {"qv": ATTENTION_QV, "all": ALL_LINEAR}


def _lora_linear_class():
    """Define the module lazily so that importing this file does not require torch."""
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """``base(x) + scaling * B(A(dropout(x)))`` with a frozen ``base`` linear layer."""

        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
            super().__init__()
            if rank < 1:
                raise ValueError(f"LoRA rank must be >= 1, got {rank}")
            self.base = base
            for param in self.base.parameters():
                param.requires_grad = False
            self.rank, self.alpha = rank, alpha
            self.scaling = alpha / rank
            self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype))
            # Same initialisation as the paper / reference implementation: A random, B zero.
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        @property
        def weight(self):
            """The frozen pretrained weight (some Hugging Face code reads ``layer.weight.dtype``)."""
            return self.base.weight

        def forward(self, x):
            update = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            return self.base(x) + self.scaling * update

        def merged(self) -> nn.Linear:
            """A plain ``nn.Linear`` with weight ``W + scaling * B A`` (same output in eval mode)."""
            layer = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
            layer = layer.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
            with torch.no_grad():
                layer.weight.copy_(self.base.weight + self.scaling * (self.lora_B @ self.lora_A))
                if self.base.bias is not None:
                    layer.bias.copy_(self.base.bias)
            return layer

    return LoRALinear


def _encoder(model):
    """The encoder of a Hugging Face ViT-like classifier (``model.vit.encoder`` etc.)."""
    backbone = getattr(model, model.base_model_prefix)
    if not hasattr(backbone, "encoder"):
        raise ValueError(f"LoRA needs a transformer encoder; {type(model).__name__} has none")
    return backbone.encoder


def _matches(name: str, targets: Sequence[str]) -> bool:
    """``name`` is a full module path; a target matches its end (``...attention.query``)."""
    return any(name == t or name.endswith("." + t) for t in targets)


def apply_lora(model, rank: int, alpha: float, dropout: float = 0.0, targets: str | Sequence[str] = "qv") -> int:
    """Freeze ``model`` and insert LoRA adapters into the encoder's linear layers in place.

    ``targets`` is a preset (``"qv"``: query + value, as in the LoRA paper; ``"all"``: every
    linear layer of each block) or a list of module-name suffixes. The classifier head stays
    trainable. Returns the number of adapted layers.
    """
    import torch.nn as nn

    LoRALinear = _lora_linear_class()
    suffixes = TARGET_PRESETS[targets] if isinstance(targets, str) else tuple(targets)
    for param in model.parameters():
        param.requires_grad = False

    encoder = _encoder(model)
    replaced = 0
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _matches(full, suffixes):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    if replaced == 0:
        raise ValueError(f"No linear layer matched the LoRA targets {suffixes}")

    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
    return replaced


def merge_lora(model) -> int:
    """Replace every LoRA layer by its merged ``nn.Linear`` in place. Returns the number merged."""
    LoRALinear = _lora_linear_class()
    merged = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == LoRALinear.__name__ and hasattr(child, "merged"):
                setattr(module, child_name, child.merged())
                merged += 1
    return merged
