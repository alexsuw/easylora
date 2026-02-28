"""Tests for target module registry and resolution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from easylora.lora.targets import (
    ARCH_TARGET_MAP,
    MODEL_TYPE_TARGET_MAP,
    _scan_linear_modules,
    resolve_target_modules_from_config,
)


class TestRegistry:
    def test_arch_map_loaded(self):
        assert len(ARCH_TARGET_MAP) > 0
        assert "LlamaForCausalLM" in ARCH_TARGET_MAP

    def test_model_type_map_loaded(self):
        assert len(MODEL_TYPE_TARGET_MAP) > 0
        assert "llama" in MODEL_TYPE_TARGET_MAP

    def test_arch_and_type_consistent(self):
        assert ARCH_TARGET_MAP["LlamaForCausalLM"] == MODEL_TYPE_TARGET_MAP["llama"]
        assert ARCH_TARGET_MAP["GPTNeoXForCausalLM"] == MODEL_TYPE_TARGET_MAP["gpt_neox"]


class TestResolveFromConfig:
    def test_known_architecture(self):
        config = SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            model_type="llama",
        )
        targets, source = resolve_target_modules_from_config(config)
        assert targets == ["q_proj", "v_proj"]
        assert "architecture" in source

    def test_known_model_type_fallback(self):
        config = SimpleNamespace(
            architectures=["SomeNewLlamaVariant"],
            model_type="llama",
        )
        targets, source = resolve_target_modules_from_config(config)
        assert targets == ["q_proj", "v_proj"]
        assert "model_type" in source

    def test_unknown_architecture(self):
        config = SimpleNamespace(
            architectures=["CompletelyNewModel"],
            model_type="completely_new",
        )
        targets, source = resolve_target_modules_from_config(config)
        assert targets == ["q_proj", "v_proj"]
        assert "fallback" in source

    def test_no_architectures(self):
        config = SimpleNamespace(model_type="llama")
        targets, _source = resolve_target_modules_from_config(config)
        assert targets == ["q_proj", "v_proj"]


class TestScanLinear:
    def _make_mock_model(self, module_names: list[str]):
        """Create a mock model with named Linear modules."""
        import torch.nn as nn

        model = MagicMock()
        modules = []
        for name in module_names:
            mod = nn.Linear(10, 10)
            modules.append((name, mod))
        # Add a non-linear module that should be skipped
        modules.append(("layernorm", nn.LayerNorm(10)))
        model.named_modules.return_value = modules
        return model

    def test_finds_attn_modules(self):
        model = self._make_mock_model(
            [
                "layer.0.self_attn.q_proj",
                "layer.0.self_attn.k_proj",
                "layer.0.self_attn.v_proj",
                "layer.0.mlp.gate_proj",
                "lm_head",
            ]
        )
        result = _scan_linear_modules(model)
        assert "q_proj" in result
        assert "k_proj" in result
        assert "v_proj" in result
        assert "lm_head" not in result
        # Should prefer attention patterns, so gate_proj should not be included
        assert "gate_proj" not in result

    def test_no_attn_returns_all(self):
        model = self._make_mock_model(
            [
                "layer.0.custom_a",
                "layer.0.custom_b",
            ]
        )
        result = _scan_linear_modules(model)
        assert "custom_a" in result
        assert "custom_b" in result
