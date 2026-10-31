"""Attention manipulations used in milestone 2 (and reused on DeiT/BEiT in milestone 3).

All functions operate in place on a Hugging Face image classifier.

Lecture background (03 - Transformer): scaled dot-product attention

    A = softmax(Q K^T / sqrt(d_k)),   out = A V

- *Temperature* T rescales the logits: softmax(Q K^T / (sqrt(d_k) * T)).
  T > 1 flattens the distribution (softer attention), T < 1 sharpens it.
- *Head pruning* removes whole attention heads (``model.prune_heads``).
- *Attention entropy* H(A) = -sum_j A_ij log A_ij measures how spread out a query's
  attention is.
"""

from __future__ import annotations

import random

from animal_vit.models import encoder_layers


def self_attention_modules(model) -> list:
    """Return the self-attention module of every encoder layer (ViT/DeiT/BEiT/Swin)."""
    import torch.nn as nn

    modules = [m for m in model.modules() if isinstance(m, nn.Module) and hasattr(m, "query") and hasattr(m, "key")]
    if not modules:  # fall back to the ViT layout
        modules = [layer.attention.attention for layer in encoder_layers(model)]
    return modules


def _scaled(inner, factor: float):
    """Wrap ``inner`` so that its output is multiplied by ``factor``."""
    import torch.nn as nn

    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.factor = factor

        def forward(self, x):
            return self.inner(x) * self.factor

    return Scaled()


def set_attention_temperature(model, temperature: float) -> None:
    """Scale attention logits by ``1 / T``: softmax(Q K^T / (sqrt(d_k) * T)).

    Implemented by scaling the query projection, softmax((Q/T) K^T / sqrt(d_k)), which
    is mathematically identical and independent of how a given ``transformers`` version
    computes the scaling internally. Works for ViT, DeiT, BEiT and Swin (the original
    project's version only worked for ViT and silently did nothing for BEiT).
    """
    if temperature == 1.0:
        return
    for module in self_attention_modules(model):
        module.query = _scaled(module.query, 1.0 / temperature)
        module.temperature = temperature


def prune_random_heads(model, n_heads: int, seed: int) -> dict[int, list[int]]:
    """Prune ``n_heads`` randomly chosen heads in every layer (reproducible via ``seed``)."""
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    if not 0 <= n_heads < num_heads:
        raise ValueError(f"n_heads must be in [0, {num_heads}), got {n_heads}")
    rng = random.Random(seed)
    plan = {layer: sorted(rng.sample(range(num_heads), n_heads)) for layer in range(num_layers)}
    if n_heads:
        model.prune_heads(plan)
    return plan


def prune_alternating_heads(model, parity: str) -> dict[int, list[int]]:
    """Prune every other head per layer: ``"even"`` (0, 2, ...) or ``"odd"`` (1, 3, ...)."""
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'")
    start = 0 if parity == "even" else 1
    heads = list(range(start, model.config.num_attention_heads, 2))
    plan = {layer: heads for layer in range(model.config.num_hidden_layers)}
    model.prune_heads(plan)
    return plan


def attention_entropy(attentions) -> torch.Tensor:  # noqa: F821
    """Mean Shannon entropy over all layers, heads and query tokens.

    ``attentions`` is the tuple returned with ``output_attentions=True``; each element
    has shape ``(batch, heads, tokens, tokens)`` and rows sum to 1.
    """
    import torch

    eps = 1e-12
    per_layer = [-(a * (a + eps).log()).sum(dim=-1).mean() for a in attentions]
    return torch.stack(per_layer).mean()
