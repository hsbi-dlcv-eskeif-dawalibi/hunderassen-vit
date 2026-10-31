"""Why do the models reach more than 90 %? Building blocks for ``make why``.

Four checks, each answering one candidate explanation:

1. **Pretraining** - train the same architectures from random initialisation
   (``configs/why/*_scratch.yaml``, option ``model.pretrained: false``).
2. **Prior knowledge** - 22 of the 25 breeds are ImageNet-1k classes. The original 1000-class
   heads of the station-1 checkpoints are evaluated *without any training*
   (:data:`IMAGENET_IDS`, :func:`imagenet_head_logits`).
3. **Leakage** - near-duplicate images between test and train/validation, found with CLIP
   embeddings (:func:`nearest_neighbours`).
4. **Separability** - how few images per breed a linear classifier on frozen ImageNet-ViT
   features needs (:func:`backbone_features`; few-shot logic shared with station 3).

The ImageNet-1k class indices follow the standard ILSVRC-2012 order used by every Hugging Face
ImageNet checkpoint (Russakovsky et al., IJCV 2015). :func:`check_imagenet_ids` verifies them
against the label names of a loaded model, so a different label order fails loudly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Breed of Oxford-IIIT Pet -> (ImageNet-1k class index, first word of the ImageNet label).
#: American Bulldog, Havanese and Shiba Inu have no ImageNet-1k class.
IMAGENET_IDS: dict[str, tuple[int, str]] = {
    "American Pit Bull Terrier": (180, "american staffordshire terrier"),
    "Basset Hound": (161, "basset"),
    "Beagle": (162, "beagle"),
    "Boxer": (242, "boxer"),
    "Chihuahua": (151, "chihuahua"),
    "English Cocker Spaniel": (219, "cocker spaniel"),
    "English Setter": (212, "english setter"),
    "German Shorthaired": (210, "german short-haired pointer"),
    "Great Pyrenees": (257, "great pyrenees"),
    "Japanese Chin": (152, "japanese spaniel"),
    "Keeshond": (261, "keeshond"),
    "Leonberger": (255, "leonberg"),
    "Miniature Pinscher": (237, "miniature pinscher"),
    "Newfoundland": (256, "newfoundland"),
    "Pomeranian": (259, "pomeranian"),
    "Pug": (254, "pug"),
    "Saint Bernard": (247, "saint bernard"),
    "Samoyed": (258, "samoyed"),
    "Scottish Terrier": (199, "scotch terrier"),
    "Staffordshire Bull Terrier": (179, "staffordshire bullterrier"),
    "Wheaten Terrier": (202, "soft-coated wheaten terrier"),
    "Yorkshire Terrier": (187, "yorkshire terrier"),
}
#: Breeds without an ImageNet-1k class (the models never saw them as a label before our training).
NOT_IN_IMAGENET = ("American Bulldog", "Havanese", "Shiba Inu")

#: Look-alike breeds (bull-type dogs); the most frequent confusions of station 1 are among these.
HARD_SUBSET = ("American Bulldog", "American Pit Bull Terrier", "Staffordshire Bull Terrier", "Boxer")


def check_imagenet_ids(id2label: dict) -> None:
    """Raise if a model's ImageNet label order does not match :data:`IMAGENET_IDS`."""
    for breed, (idx, expected) in IMAGENET_IDS.items():
        label = str(id2label.get(idx, id2label.get(str(idx), ""))).lower()
        if not label.startswith(expected):
            raise ValueError(f"ImageNet id {idx} is '{label}', expected '{expected}' for {breed}")


def imagenet_head_logits(logits_1000: np.ndarray, class_names: Sequence[str]) -> tuple[np.ndarray, list[int]]:
    """Restrict 1000-class ImageNet logits to our breeds that exist in ImageNet.

    Returns ``(logits, kept)``: ``logits[:, j]`` is the ImageNet logit of breed ``kept[j]``
    (an index into ``class_names``). Breeds without an ImageNet class are left out.
    """
    kept = [i for i, name in enumerate(class_names) if name in IMAGENET_IDS]
    cols = [IMAGENET_IDS[class_names[i]][0] for i in kept]
    return logits_1000[:, cols], kept


def nearest_neighbours(query: np.ndarray, reference: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of each (L2-normalised) query row to its nearest reference row.

    Returns ``(max_similarity, argmax_index)`` per query row; works in chunks to bound memory.
    """
    best_sim = np.empty(len(query), dtype=np.float32)
    best_idx = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk):
        sims = query[start : start + chunk] @ reference.T
        best_idx[start : start + chunk] = sims.argmax(axis=1)
        best_sim[start : start + chunk] = sims.max(axis=1)
    return best_sim, best_idx


def backbone_features(model, dataset, device: str, batch_size: int = 128, num_workers: int = 4):
    """L2-normalised features of a frozen Hugging Face backbone and the labels (numpy).

    Transformers: the final (layer-normed) CLS token, i.e. exactly what linear probing in
    station 1 trains on. CNNs: the globally pooled feature map.
    """
    import torch
    from torch.utils.data import DataLoader

    from animal_vit.data import collate_fn

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    feats, labels = [], []
    model.eval().to(device)
    with torch.no_grad():
        for batch in loader:
            out = model(pixel_values=batch["pixel_values"].to(device))
            hidden = out.last_hidden_state
            # (B, tokens, dim) for transformers -> CLS token; (B, C, H, W) for CNNs -> pooled output.
            emb = hidden[:, 0] if hidden.dim() == 3 else out.pooler_output.flatten(1)
            feats.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
            labels.append(batch["labels"].numpy())
    return np.concatenate(feats), np.concatenate(labels)
