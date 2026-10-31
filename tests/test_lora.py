"""LoRA (station 1): adapters, trainable parameters, merging, and the full train/restore path."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_pipeline import SIZE, _cfg, _tiny_model  # noqa: E402

from animal_vit.lora import apply_lora, merge_lora  # noqa: E402
from animal_vit.models import count_parameters  # noqa: E402


def _batch(n: int = 2):
    return torch.randn(n, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(0))


def test_lora_starts_from_the_pretrained_function():
    model = _tiny_model().eval()
    x = _batch()
    before = model(pixel_values=x).logits
    replaced = apply_lora(model, rank=2, alpha=4, targets="qv")
    assert replaced == 2 * 2  # 2 blocks x (query, value)
    # B is initialised with zeros -> identical output before any training.
    assert torch.allclose(model.eval()(pixel_values=x).logits, before, atol=1e-6)


def test_only_adapters_and_head_are_trainable():
    model = _tiny_model()
    apply_lora(model, rank=2, alpha=4, targets="qv")
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all("lora_" in n or "classifier" in n for n in names)
    trainable, total = count_parameters(model)
    assert 0 < trainable < total


def test_all_targets_adapt_every_linear_layer_of_a_block():
    model = _tiny_model()
    assert apply_lora(model, rank=2, alpha=4, targets="all") == 2 * 6


def test_merge_keeps_the_output_and_restores_plain_layers():
    model = _tiny_model()
    apply_lora(model, rank=2, alpha=4, dropout=0.0, targets="qv")
    with torch.no_grad():  # pretend training changed B
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    model.eval()
    x = _batch()
    adapted = model(pixel_values=x).logits
    assert merge_lora(model) == 4
    assert not any("lora_" in n for n, _ in model.named_parameters())
    assert torch.allclose(model(pixel_values=x).logits, adapted, atol=1e-5)


def test_lora_run_trains_merges_and_restores(tmp_path, tiny_checkpoint, use_fake_data):
    from animal_vit.training import load_run, run_experiment

    cfg = _cfg(
        tmp_path,
        tiny_checkpoint,
        "lora",
        modifications={"trainable": "lora", "lora": {"rank": 2, "alpha": 4, "targets": "qv"}},
    )
    run_experiment(cfg)
    payload = json.loads((tmp_path / "lora" / "metrics.json").read_text())
    assert payload["lora_layers"] == payload["lora_merged_layers"] == 4
    assert payload["trainable_parameters"] < payload["total_parameters"]
    for record in ("README.md", "history.json", "data_stats.json", "predictions.csv", "environment.json"):
        assert (tmp_path / "lora" / record).exists(), record
    assert payload["n_train"] == 18 and payload["n_test"] == 9
    model, cfg, _, _ = load_run(tmp_path / "lora")  # merged weights load into the plain model
    assert cfg.modifications.lora.rank == 2
    assert not any("lora_" in n for n, _ in model.named_parameters())
