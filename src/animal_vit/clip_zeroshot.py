"""Station 3 "Generalization": CLIP as a classifier without task-specific training.

CLIP (Radford et al., *Learning Transferable Visual Models From Natural Language
Supervision*, ICML 2021) embeds images and texts into one space. A zero-shot classifier is
built from text alone: every breed name is inserted into prompt templates ("a photo of a
{}, a type of dog."), the text embeddings of a class are averaged, and an image is assigned
to the class with the highest cosine similarity. No image of the dataset is used.

Compared with station 1 (fine-tuned ViT / CNN) and station 2 (distilled students), this
answers the station-3 question: *how well can CLIP replace a ViT without training?*

Two optional references:

- **Prompt sets** (``PROMPT_SETS``): how much the wording matters (prompt engineering).
- **Linear probe** on frozen CLIP image features (logistic regression, C selected on the
  validation split): CLIP's representation with a minimal amount of task-specific training,
  the bridge to linear probing in station 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Prompt templates; ``{}`` is replaced by the breed name.
PROMPT_SETS: dict[str, list[str]] = {
    "name": ["{}"],
    "photo": ["a photo of a {}."],
    "dog": ["a photo of a {}, a type of dog."],
    # Ensemble in the spirit of the CLIP paper (averaging several templates).
    "ensemble": [
        "a photo of a {}, a type of dog.",
        "a photo of the {}, a dog breed.",
        "a close-up photo of a {}.",
        "a cropped photo of a {}.",
        "a bright photo of a {}.",
        "a low resolution photo of a {}.",
        "an image of a {} dog.",
    ],
}


def format_prompts(class_names: Sequence[str], templates: Sequence[str]) -> list[list[str]]:
    """``prompts[c][t]``: template ``t`` filled with class ``c`` (lower case, underscores removed)."""
    return [[t.format(name.replace("_", " ").lower()) for t in templates] for name in class_names]


def load_clip(checkpoint: str, device: str):
    """Return ``(model, tokenizer, size, mean, std)`` for a Hugging Face CLIP checkpoint."""
    from transformers import AutoImageProcessor, AutoTokenizer, CLIPModel

    model = CLIPModel.from_pretrained(checkpoint).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    size = processor.crop_size.get("height") if isinstance(processor.crop_size, dict) else 224
    return model, tokenizer, int(size or 224), processor.image_mean, processor.image_std


def text_classifier(model, tokenizer, class_names: Sequence[str], templates: Sequence[str], device: str):
    """(num_classes, dim) tensor: L2-normalised mean text embedding per class."""
    import torch

    weights = []
    with torch.no_grad():
        for prompts in format_prompts(class_names, templates):
            tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            emb = model.get_text_features(**tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            weights.append(mean / mean.norm())
    return torch.stack(weights)


def image_features(model, dataset, device: str, batch_size: int = 128, num_workers: int = 4):
    """L2-normalised image embeddings and labels of a dataset (numpy)."""
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = model.get_image_features(pixel_values=batch["pixel_values"].to(device))
            feats.append((emb / emb.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def zero_shot_logits(image_feats: np.ndarray, text_weights, logit_scale: float = 100.0) -> np.ndarray:
    """Cosine similarity times CLIP's logit scale: (num_images, num_classes)."""
    text = text_weights.float().cpu().numpy() if hasattr(text_weights, "cpu") else np.asarray(text_weights)
    return logit_scale * image_feats @ text.T


def linear_probe(train, val, test, c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Logistic regression on frozen features; C chosen on validation accuracy.

    ``train``, ``val``, ``test`` are ``(features, labels)`` tuples. Returns
    ``(test_logits, best_C, val_accuracy)``.
    """
    from sklearn.linear_model import LogisticRegression

    best = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=5000)
        clf.fit(*train)
        acc = clf.score(*val)
        if best is None or acc > best[2]:
            best = (clf, c, acc)
    clf, c, acc = best
    scores = clf.decision_function(test[0])
    # A subset of the training data may miss classes: pad to one column per class (missing = very unlikely).
    n_classes = int(max(train[1].max(), val[1].max(), test[1].max())) + 1
    logits = np.full((len(scores), n_classes), -1e9)
    logits[:, clf.classes_] = scores
    return logits, c, acc


def few_shot_indices(labels: Sequence[int], shots: int, seed: int) -> list[int]:
    """``shots`` random indices per class (all of a class if it has fewer), reproducible via ``seed``."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    picked = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        picked.extend(rng.choice(idx, size=min(shots, len(idx)), replace=False).tolist())
    return sorted(picked)
