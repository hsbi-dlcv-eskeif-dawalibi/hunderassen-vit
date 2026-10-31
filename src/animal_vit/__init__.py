"""animal_vit: shared code for the "Deep Learning for Computer Vision" project on 25 dog breeds.

The package bundles everything the experiments of the three stations need:

- :mod:`animal_vit.paths`      central repository and data paths
- :mod:`animal_vit.config`     YAML experiment configuration
- :mod:`animal_vit.data`       dataset loading, transforms, collate function
- :mod:`animal_vit.models`     loading ViT / DeiT / CNN classifiers (Hugging Face)
- :mod:`animal_vit.attention`  attention manipulations (temperature, head pruning, entropy)
- :mod:`animal_vit.metrics`    classification metrics and calibration
- :mod:`animal_vit.training`   Hugging Face ``Trainer`` setup and evaluation export
- :mod:`animal_vit.lora`       LoRA adapters (station 1)
- :mod:`animal_vit.clip_zeroshot` CLIP zero-/few-shot (station 3)
- :mod:`animal_vit.why`, :mod:`animal_vit.explain` analyses "why > 90 %" and explanation maps
- :mod:`animal_vit.plotting`   confusion matrix

Heavy dependencies (torch, transformers, ...) are imported inside the modules that
need them, so lightweight helpers such as :mod:`animal_vit.paths` stay importable
without a deep-learning stack.
"""

__version__ = "1.0.0"
