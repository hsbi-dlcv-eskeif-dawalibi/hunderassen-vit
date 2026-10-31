"""Shape vs. texture: cue-conflict images via neural style transfer (lecture 05).

Geirhos et al. (*ImageNet-trained CNNs are biased towards texture*, ICLR 2019) showed
that CNNs classify a cat with elephant skin as an elephant, while humans see a cat. We
build the same kind of *cue-conflict* images inside our dataset: the **content** (shape)
of a test image of breed A is rendered with the **style** (texture) of an image of breed
B, using the original optimisation-based style transfer of Gatys et al. (CVPR 2016)
exactly as presented in lecture 05:

    x* = argmin_x  alpha * L_content(x) + beta * L_style(x)
    L_content = || F_l(x) - F_l(content) ||^2                 (one deep VGG layer)
    L_style   = sum_l w_l || G_l(x) - G_l(style) ||^2,  G = F F^T / (C H W)   (Gram matrices)

A model "decides by shape" if it predicts breed A and "by texture" if it predicts breed
B. ``shape_bias = shape decisions / (shape + texture decisions)``.

VGG-19 ImageNet weights are loaded from torchvision.
"""

from __future__ import annotations

from animal_vit.data import IMAGENET_MEAN, IMAGENET_STD

CONTENT_LAYERS = (22,)  # relu4_2
STYLE_LAYERS = (1, 6, 11, 20, 29)  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1


def gram_matrix(features):
    """Normalised Gram matrix G = F F^T of feature maps ``(B, C, H, W)``: which features co-occur
    (texture). Its difference between images is the style loss (Gatys et al. 2016, lecture 05)."""
    b, c, h, w = features.shape
    f = features.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


class StyleTransfer:
    """Gatys et al. style transfer with a frozen VGG-19 feature extractor."""

    def __init__(self, device: str = "cpu", size: int = 224):
        import torch
        from torchvision.models import VGG19_Weights, vgg19

        self.device, self.size = device, size
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: max(STYLE_LAYERS + CONTENT_LAYERS) + 1]
        self.vgg.eval().requires_grad_(False).to(device)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _features(self, x):
        feats = {}
        x = (x - self.mean) / self.std
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in STYLE_LAYERS or i in CONTENT_LAYERS:
                feats[i] = x
        return feats

    def _load(self, image):
        from torchvision.transforms import functional as F

        image = F.center_crop(F.resize(image.convert("RGB"), self.size), self.size)
        return F.to_tensor(image)[None].to(self.device)

    def __call__(
        self, content_image, style_image, steps: int = 300, style_weight: float = 1e6, content_weight: float = 1.0
    ):
        """Return the stylised PIL image (content of ``content_image``, texture of ``style_image``)."""
        import torch
        from torchvision.transforms import functional as F

        content, style = self._load(content_image), self._load(style_image)
        with torch.no_grad():
            content_feats = self._features(content)
            style_grams = {i: gram_matrix(f) for i, f in self._features(style).items() if i in STYLE_LAYERS}
        x = content.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS([x], max_iter=steps)

        def closure():
            optimizer.zero_grad()
            x.data.clamp_(0, 1)
            feats = self._features(x)
            c_loss = sum(torch.mean((feats[i] - content_feats[i]) ** 2) for i in CONTENT_LAYERS)
            s_loss = sum(torch.mean((gram_matrix(feats[i]) - style_grams[i]) ** 2) for i in STYLE_LAYERS)
            loss = content_weight * c_loss + style_weight * s_loss
            loss.backward()
            return loss

        optimizer.step(closure)
        return F.to_pil_image(x.detach().clamp(0, 1)[0].cpu())


def shape_bias(predictions, shape_labels, texture_labels) -> dict[str, float]:
    """Fraction of shape vs. texture decisions among cue-conflict images."""
    import numpy as np

    p, s, t = (np.asarray(a) for a in (predictions, shape_labels, texture_labels))
    shape_hits, texture_hits = (p == s).sum(), (p == t).sum()
    decided = shape_hits + texture_hits
    return {
        "shape_accuracy": float(shape_hits / len(p)),
        "texture_accuracy": float(texture_hits / len(p)),
        "shape_bias": float(shape_hits / decided) if decided else float("nan"),
        "n": int(len(p)),
    }
