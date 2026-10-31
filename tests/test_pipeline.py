"""End-to-end tests with tiny randomly initialised models (CPU, no downloads).

They exercise the real code paths - training, saving, restoring, distillation, component
swaps and the analysis methods - on a fake 3-class ImageFolder dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from animal_vit.config import ExperimentConfig  # noqa: E402

SIZE = 32


def _cfg(tmp_path: Path, checkpoint: Path, name: str, **sections) -> ExperimentConfig:
    raw = {
        "name": name,
        "output_dir": str(tmp_path / name),
        "model": {"checkpoint": str(checkpoint)},
        "data": {"name": "image_folder", "download": False},
        "training": {
            "epochs": 1,
            "batch_size": 6,
            "fp16": False,
            "extra": {"dataloader_num_workers": 0, "use_cpu": True, "report_to": "none"},
        },
    }
    for key, value in sections.items():
        raw.setdefault(key, {}).update(value)
    return ExperimentConfig.from_dict(raw)


def test_train_save_restore_and_distill(tmp_path, tiny_checkpoint, use_fake_data):
    from animal_vit.training import load_run, run_experiment

    teacher_cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        "teacher",
        modifications={"positional_embedding": "shuffled", "attention_temperature": 1.5, "prune_heads_random": 1},
    )
    metrics = run_experiment(teacher_cfg)
    run_dir = tmp_path / "teacher"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    for artefact in ("metrics.json", "classification_report.csv", "confusion_matrix.png", "class_names.json"):
        assert (run_dir / artefact).exists()

    model, cfg, class_names, (size, _, _) = load_run(run_dir)
    assert class_names == ["cat", "dog", "fox"] and size == SIZE
    assert cfg.modifications.prune_heads_random == 1

    student_cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        "student",
        distillation={"teacher": str(run_dir)},
        modifications={"patch_embedding": "conv_stem"},
    )
    run_experiment(student_cfg)
    assert json.loads((tmp_path / "student" / "metrics.json").read_text())["teacher"] == str(run_dir)
    load_run(tmp_path / "student")  # conv stem must be rebuilt on restore


def _tiny_model(**overrides):
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        image_size=SIZE,
        patch_size=16,
        num_labels=3,
        **overrides,
    )
    return ViTForImageClassification(config).eval()


def test_temperature_equals_scaled_logits():
    from animal_vit.attention import set_attention_temperature

    torch.manual_seed(0)
    model = _tiny_model(attn_implementation="eager")
    x = torch.randn(2, 3, SIZE, SIZE)
    base = model(pixel_values=x, output_attentions=True).attentions[0]
    set_attention_temperature(model, 1e6)  # T -> inf: attention becomes uniform
    flat = model(pixel_values=x, output_attentions=True).attentions[0]
    assert not torch.allclose(base, flat)
    assert torch.allclose(flat, torch.full_like(flat, 1 / flat.shape[-1]), atol=1e-4)


def test_positional_embedding_variants():
    from animal_vit.components import set_positional_embedding

    model = _tiny_model()
    pos = model.vit.embeddings.position_embeddings
    original = pos.detach().clone()
    set_positional_embedding(model, "shuffled", seed=0)
    assert torch.equal(pos[:, 0], original[:, 0])  # [CLS] untouched
    assert torch.allclose(pos[:, 1:].sum(1), original[:, 1:].sum(1), atol=1e-5)  # permutation
    set_positional_embedding(model, "none")
    assert pos.abs().sum() == 0 and not pos.requires_grad


def test_conv_stem_keeps_token_grid():
    from animal_vit.components import set_patch_embedding

    model = _tiny_model()
    set_patch_embedding(model, "conv_stem")
    out = model(pixel_values=torch.randn(2, 3, SIZE, SIZE), output_hidden_states=True)
    assert out.hidden_states[-1].shape == (2, 1 + (SIZE // 16) ** 2, 32)


def test_saliency_methods_and_focus():
    from animal_vit.saliency import attention_rollout, foreground_focus, grad_cam, upsample

    model = _tiny_model(attn_implementation="eager")
    x = torch.randn(2, 3, SIZE, SIZE)
    rollout = attention_rollout(model, x)
    cam = grad_cam(model, x)
    assert rollout.shape == cam.shape == (2, SIZE // 16, SIZE // 16)
    maps = upsample(cam, SIZE)
    assert np.allclose(maps.reshape(2, -1).sum(1), 1.0, atol=1e-4) or np.all(maps == 0)

    sal = np.zeros((4, 4))
    sal[:2] = 1.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2] = True
    assert foreground_focus(sal, mask) == pytest.approx(2.0)  # all mass on half the image


def test_efficiency_profile():
    from animal_vit.efficiency import count_flops

    assert count_flops(_tiny_model(), SIZE) > 0


class _FakeMaskedSource:
    """Mimics the Oxford-IIIT Pet source: (PIL image, label, PIL trimap)."""

    def __init__(self, n: int = 6):
        from PIL import Image

        rng = np.random.default_rng(0)
        self.items = []
        for i in range(n):
            image = Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
            trimap = np.full((48, 48), 2, dtype=np.uint8)
            trimap[12:36, 12:36] = 1
            self.items.append((image, i % 3, Image.fromarray(trimap)))
        self.labels = [item[1] for item in self.items]

    def __call__(self, i):
        return self.items[i]


def test_variant_dataset_and_predict():
    from animal_vit.data import ClassificationDataset, normalize_tensor
    from animal_vit.evaluation import predict
    from animal_vit.variants import VARIANTS, VariantDataset

    source = _FakeMaskedSource()
    base = ClassificationDataset(
        source, range(6), normalize_tensor((0.5,) * 3, (0.5,) * 3), ["a", "b", "c"], with_mask=True, size=SIZE
    )
    model = _tiny_model()
    for variant in VARIANTS:
        ds = VariantDataset(base, variant, SIZE, (0.5,) * 3, (0.5,) * 3)
        item = ds[0]
        assert item["pixel_values"].shape == (3, SIZE, SIZE) and item["mask"].dtype == torch.bool
        logits, labels = predict(model, ds, "cpu", batch_size=4, num_workers=0)
        assert logits.shape == (6, 3) and labels.tolist() == source.labels
    partner = VariantDataset(base, "mixed_random", SIZE, (0.5,) * 3, (0.5,) * 3).partner
    assert all(source.labels[p] != source.labels[i] for i, p in enumerate(partner))
