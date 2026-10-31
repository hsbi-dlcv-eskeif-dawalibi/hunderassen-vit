"""Milestone 2: swap components of a pretrained ViT/DeiT.

The task of milestone 2 is to *replace components of the ViT architecture* and study the
effect. This module replaces the two components that decide how a ViT turns an image
into tokens - the parts that most clearly separate a ViT from a CNN:

Position embeddings (``modifications.positional_embedding``)
    A transformer treats its input as a *set*; spatial layout only enters through the
    position embeddings (lecture 03, "Positional Encoding").

    - ``learned``  - pretrained embeddings (baseline)
    - ``none``     - embeddings zeroed and frozen: the encoder sees a bag of patches.
      If accuracy barely drops, the model classifies from local texture, not from the
      global arrangement of parts (= shape).
    - ``shuffled`` - pretrained vectors permuted across patch positions (fixed seed):
      the layout prior learned during pretraining is wrong and must be re-learned.

Patch embedding (``modifications.patch_embedding``)
    - ``linear``    - pretrained 16x16 linear projection (baseline)
    - ``conv_stem`` - a small stack of stride-2 3x3 convolutions followed by a 1x1
      projection, producing the same 14x14 token grid (Xiao et al., *Early Convolutions
      Help Transformers See Better*, NeurIPS 2021). Gives the ViT a CNN-like inductive
      bias (locality, translation equivariance) at the input.
"""

from __future__ import annotations

from animal_vit.models import backbone


def _embeddings(model):
    module = getattr(backbone(model), "embeddings", None)
    if module is None or not hasattr(module, "position_embeddings"):
        raise NotImplementedError(f"{type(model).__name__}: component swaps support ViT/DeiT-style models only")
    return module


def set_positional_embedding(model, mode: str, seed: int = 0) -> None:
    """Apply the position-embedding variant ``mode`` in place."""
    import torch

    if mode == "learned":
        return
    pos = _embeddings(model).position_embeddings  # (1, 1 + num_patches, dim); index 0 = [CLS]
    with torch.no_grad():
        if mode == "none":
            pos.zero_()
            pos.requires_grad = False
        elif mode == "shuffled":
            generator = torch.Generator().manual_seed(seed)
            perm = torch.randperm(pos.shape[1] - 1, generator=generator) + 1
            pos[:, 1:] = pos[:, perm].clone()
        else:
            raise ValueError(f"positional_embedding must be learned|none|shuffled, got '{mode}'")


def build_conv_stem(out_dim: int, channels=(48, 96, 192, 384), in_channels: int = 3):
    """Four stride-2 3x3 conv blocks (overall stride 16) + 1x1 projection to ``out_dim``."""
    import torch.nn as nn

    class ConvStem(nn.Sequential):
        # ViTEmbeddings reads ``projection.weight.dtype`` to cast its input.
        @property
        def weight(self):
            return self[-1].weight

    layers, prev = [], in_channels
    for width in channels:
        layers += [nn.Conv2d(prev, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU()]
        prev = width
    layers.append(nn.Conv2d(prev, out_dim, kernel_size=1))
    return ConvStem(*layers)


def set_patch_embedding(model, mode: str) -> None:
    """Apply the patch-embedding variant ``mode`` in place."""
    if mode == "linear":
        return
    if mode != "conv_stem":
        raise ValueError(f"patch_embedding must be linear|conv_stem, got '{mode}'")
    patch = _embeddings(model).patch_embeddings
    projection = patch.projection
    if projection.kernel_size != (16, 16):
        raise ValueError("conv_stem has an overall stride of 16 and needs a /16 model")
    patch.projection = build_conv_stem(projection.out_channels)


def apply_components(
    model, positional_embedding: str = "learned", patch_embedding: str = "linear", seed: int = 0
) -> None:
    """Apply both MS2 component swaps to a ViT in place.

    The patch embedding is replaced first, then the position embeddings are changed
    (``seed`` controls the permutation of ``"shuffled"``).
    """
    set_patch_embedding(model, patch_embedding)
    set_positional_embedding(model, positional_embedding, seed=seed)
