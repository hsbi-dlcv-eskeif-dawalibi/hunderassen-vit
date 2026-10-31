"""Where does a model look? Saliency maps and the *foreground focus* metric.

Two methods (lecture 03: attention; lecture 04: "what does the model really learn?"):

Attention rollout (ViT/DeiT only)
    Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020. The
    attention matrices of all layers (averaged over heads, plus identity for the residual
    connection, row-normalised) are multiplied; the [CLS] row gives how much each patch
    contributes to the classification token.

Grad-CAM (all models)
    Selvaraju et al., ICCV 2017. Gradient of the predicted logit w.r.t. the last feature
    map, channel-weighted and ReLU'd. Applied to the last hidden state, so it works for
    CNNs (B, C, H, W), Swin (B, N, C) and ViTs (B, 1+N, C) alike.

Foreground focus
    ``focus = (saliency mass inside the pet mask) / (area fraction of the pet)``.
    1.0 = the model looks at pet and background in proportion to their size; > 1 = it
    concentrates on the animal; < 1 = it prefers the background.
"""

from __future__ import annotations

import numpy as np

from animal_vit.models import model_family


def _to_grid(tokens, drop_cls: bool):
    """(B, N[+1], C) tokens -> (B, C, h, w) feature map."""
    import math

    if drop_cls:
        tokens = tokens[:, 1:]
    b, n, c = tokens.shape
    side = int(math.isqrt(n))
    if side * side != n:  # e.g. DeiT distilled (2 special tokens)
        tokens = tokens[:, n - side * side :]
        n = side * side
    return tokens.transpose(1, 2).reshape(b, c, side, side)


def attention_rollout(model, pixel_values, discard_ratio: float = 0.0):
    """Return (B, h, w) rollout maps. Requires a ViT-family model."""
    import torch

    if model_family(model) != "vit":
        raise ValueError("attention rollout needs a ViT/DeiT/BEiT model")
    outputs = model(pixel_values=pixel_values, output_attentions=True)
    result = None
    for attn in outputs.attentions:  # (B, heads, N, N)
        a = attn.mean(dim=1)
        if discard_ratio:
            flat = a.flatten(1)
            k = int(flat.shape[1] * discard_ratio)
            idx = flat.topk(k, dim=1, largest=False).indices
            flat.scatter_(1, idx, 0)
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a if result is None else a @ result
    cls_row = result[:, 0, 1:]
    n = cls_row.shape[1]
    side = int(n**0.5)
    cls_row = cls_row[:, n - side * side :]
    return cls_row.reshape(-1, side, side)


def grad_cam(model, pixel_values, target: torch.Tensor | None = None):  # noqa: F821
    """Return (B, h, w) Grad-CAM maps for the predicted (or ``target``) class."""
    import torch

    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    hidden.retain_grad()
    logits = outputs.logits
    target = logits.argmax(-1) if target is None else target
    model.zero_grad(set_to_none=True)
    logits.gather(1, target[:, None]).sum().backward()
    feats, grads = hidden, hidden.grad
    if feats.dim() == 3:  # token sequence
        drop_cls = model_family(model) == "vit"
        feats, grads = _to_grid(feats, drop_cls), _to_grid(grads, drop_cls)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1))
    return cam.detach()


def upsample(maps, size: int) -> np.ndarray:
    """(B, h, w) tensor -> (B, size, size) numpy, each map scaled to sum 1."""
    import torch.nn.functional as F

    up = F.interpolate(maps[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    up = up.clamp_min(0)
    up = up / up.flatten(1).sum(1).clamp_min(1e-12)[:, None, None]
    return up.cpu().numpy()


def foreground_focus(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Saliency mass inside ``mask`` divided by the mask's area fraction (see module doc)."""
    saliency = saliency / max(saliency.sum(), 1e-12)
    area = mask.mean()
    if area == 0 or area == 1:
        return float("nan")
    return float(saliency[mask].sum() / area)
