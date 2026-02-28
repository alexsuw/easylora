"""Tests for sharded device_map training flags."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from easylora.config import DataConfig, ModelConfig, TrainConfig

trainer_module = importlib.import_module("easylora.train.trainer")
_enable_model_parallel_if_sharded = trainer_module._enable_model_parallel_if_sharded


def test_enable_model_parallel_for_sharded_auto_device_map(monkeypatch):
    cfg = TrainConfig(
        model=ModelConfig(base_model="gpt2", device_map="auto"),
        data=DataConfig(dataset_path="dummy.jsonl"),
    )
    model = SimpleNamespace(base_model=SimpleNamespace())

    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, "device_count", lambda: 2)

    _enable_model_parallel_if_sharded(model, cfg)

    assert model.is_parallelizable is True
    assert model.model_parallel is True
    assert model.base_model.is_parallelizable is True
    assert model.base_model.model_parallel is True


def test_skip_model_parallel_for_non_auto_device_map(monkeypatch):
    cfg = TrainConfig(
        model=ModelConfig(base_model="gpt2", device_map=None),
        data=DataConfig(dataset_path="dummy.jsonl"),
    )
    model = SimpleNamespace(base_model=SimpleNamespace())

    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.torch.cuda, "device_count", lambda: 4)

    _enable_model_parallel_if_sharded(model, cfg)

    assert not hasattr(model, "model_parallel")
