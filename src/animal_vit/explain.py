"""Why this breed and not the other? Explanation methods and checks for ``make explain``.

Standard methods come from two libraries (cite them in the paper):

- **Grad-CAM** (Selvaraju et al., ICCV 2017) via ``pytorch-grad-cam`` (Gildenblat et al.,
  https://github.com/jacobgil/pytorch-grad-cam).
- **Occlusion** (Zeiler & Fergus, ECCV 2014) via **Captum** (Kokhlikyan et al., 2020,
  https://captum.ai): grey patches slide over the image; where the class score drops most, the
  model takes its evidence from.
- **Attention rollout** (Abnar & Zuidema, ACL 2020) for ViTs, our own implementation in
  :mod:`animal_vit.saliency`.

Our own additions on top:

- **Contrastive Grad-CAM** (:class:`ContrastiveTarget`): the target is ``logit_A - logit_B``, so
  the map shows the regions that speak for breed A *and against* breed B (cf. Prabhushankar et al.,
  ICIP 2020, "Contrastive explanations in neural networks").
- **Head focus** (:func:`box_focus`): saliency mass inside the annotated head box of Oxford-IIIT
  Pet relative to the box area (the dataset has head boxes for trainval only).
- **Deletion test** (:func:`deletion_curve`, after Petsiuk et al., BMVC 2018): the patches a
  method marks as important are greyed out first; a faithful explanation makes the class
  probability fall faster than random deletion.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from animal_vit.models import model_family

#: Fractions of the image (in 16x16 patches) removed by the deletion test.
DELETION_STEPS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def logits_only(model):
    """Wrap a Hugging Face classifier so that ``wrapper(x)`` returns the logits tensor (library API)."""
    import torch

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(pixel_values=x).logits

    return _LogitsOnly(model)


class ContrastiveTarget:
    """pytorch-grad-cam target ``logit[a] - logit[b]``: evidence for ``a`` against ``b``."""

    def __init__(self, a: int, b: int):
        self.a, self.b = a, b

    def __call__(self, output):
        if output.dim() == 1:
            return output[self.a] - output[self.b]
        return output[:, self.a] - output[:, self.b]


def target_layer(model):
    """Layer whose activations Grad-CAM uses (pytorch-grad-cam recommendations per architecture)."""
    family = model_family(model)
    base = getattr(model, model.base_model_prefix)
    if family == "vit":
        return base.encoder.layer[-1].layernorm_before
    if hasattr(base, "encoder") and hasattr(base.encoder, "stages"):  # ResNet, ConvNeXt
        return base.encoder.stages[-1]
    raise ValueError(f"no Grad-CAM layer known for {type(model).__name__}")


def vit_reshape(tensor):
    """(B, 1+N[+1], C) tokens -> (B, C, h, w); drops the class (and distillation) token."""
    n = tensor.shape[1]
    side = math.isqrt(n - 1)
    tokens = tensor[:, n - side * side :]
    return tokens.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(0, 3, 1, 2)


def grad_cam_maps(model, x, targets) -> np.ndarray:
    """(B, H, W) Grad-CAM maps in [0, 1] for one pytorch-grad-cam target per image."""
    from pytorch_grad_cam import GradCAM

    reshape = vit_reshape if model_family(model) == "vit" else None
    with GradCAM(model=logits_only(model), target_layers=[target_layer(model)], reshape_transform=reshape) as cam:
        return cam(input_tensor=x, targets=targets)


def occlusion_maps(model, x, labels, window: int = 32, stride: int = 16, per_eval: int = 64) -> np.ndarray:
    """(B, H, W) occlusion maps (Captum): drop of the class logit when a grey window covers a region.

    Baseline 0 is the dataset mean after normalisation (a grey patch).
    """
    from captum.attr import Occlusion

    occ = Occlusion(logits_only(model))
    attr = occ.attribute(
        x,
        target=labels,
        sliding_window_shapes=(3, window, window),
        strides=(3, stride, stride),
        baselines=0,
        perturbations_per_eval=per_eval,
    )
    return attr.sum(1).detach().cpu().numpy()


def normalise(maps: np.ndarray) -> np.ndarray:
    """Clip negatives and scale each map to sum 1 (comparable saliency mass)."""
    maps = np.clip(maps, 0, None)
    sums = maps.reshape(len(maps), -1).sum(1).clip(min=1e-12)
    return maps / sums[:, None, None]


def head_box(xml_path: Path) -> tuple[float, float, float, float] | None:
    """Head bounding box ``(xmin, ymin, xmax, ymax)`` in original pixels from an Oxford-IIIT Pet xml."""
    if not Path(xml_path).exists():
        return None
    box = ET.parse(xml_path).getroot().find("object/bndbox")
    return tuple(float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))


def map_box(box, width: int, height: int, size: int):
    """Map a box through ``Resize(size)`` (shorter side) + ``CenterCrop(size)``; clipped, in pixels."""
    scale = size / min(width, height)
    off_x, off_y = (width * scale - size) / 2, (height * scale - size) / 2
    x0, y0, x1, y1 = (box[0] * scale - off_x, box[1] * scale - off_y, box[2] * scale - off_x, box[3] * scale - off_y)
    clip = lambda v: float(min(max(v, 0), size))  # noqa: E731
    return clip(x0), clip(y0), clip(x1), clip(y1)


def box_mask(box, size: int) -> np.ndarray:
    """Boolean (size, size) mask of a pixel box."""
    mask = np.zeros((size, size), dtype=bool)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask[y0:y1, x0:x1] = True
    return mask


def box_focus(saliency: np.ndarray, mask: np.ndarray) -> float:
    """Saliency mass inside ``mask`` divided by its area fraction (1 = proportional, > 1 = focused)."""
    area = mask.mean()
    if area in (0, 1):
        return float("nan")
    return float(saliency[mask].sum() / max(saliency.sum(), 1e-12) / area)


def top_region_iou(a: np.ndarray, b: np.ndarray, share: float = 0.2) -> float:
    """IoU of the top-``share`` pixels of two saliency maps: do two models look at the same place?"""
    ka, kb = np.quantile(a, 1 - share), np.quantile(b, 1 - share)
    ma, mb = a >= ka, b >= kb
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else float("nan")


def deletion_curve(model, x, labels, saliency: np.ndarray, patch: int = 16, steps=DELETION_STEPS) -> np.ndarray:
    """Mean probability of the true class after greying out the most salient patches first.

    Returns an array with one value per entry of ``steps`` (fraction of patches removed). The
    saliency maps are pooled to the patch grid, patches are removed in descending order.
    """
    import torch

    b, _, h, w = x.shape
    gh, gw = h // patch, w // patch
    pooled = saliency[:, : gh * patch, : gw * patch].reshape(b, gh, patch, gw, patch).sum((2, 4)).reshape(b, -1)
    order = np.argsort(-pooled, axis=1)
    out = []
    with torch.no_grad():
        for frac in steps:
            k = int(round(frac * gh * gw))
            keep = torch.ones(b, gh * gw, device=x.device)
            if k:
                keep[torch.arange(b)[:, None], torch.as_tensor(order[:, :k], device=x.device)] = 0
            mask = keep.reshape(b, 1, gh, gw).repeat_interleave(patch, 2).repeat_interleave(patch, 3)
            probs = model(pixel_values=x * mask).logits.softmax(-1)
            out.append(probs[torch.arange(b), labels].mean().item())
    return np.array(out)


def curve_area(values: np.ndarray, steps=DELETION_STEPS) -> float:
    """Area under a deletion curve, normalised by the x-range (lower = more faithful explanation)."""
    xs = np.asarray(steps, dtype=float)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy >= 2.0 renamed trapz
    return float(trapezoid(values, xs) / (xs[-1] - xs[0]))
