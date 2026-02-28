"""Tests for LoRA adapter save/load with a tiny model."""

from __future__ import annotations

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from easylora.config import LoRAConfig, ModelConfig
from easylora.lora.adapter import apply_lora, load_adapter, save_adapter
from easylora.lora.targets import ARCH_TARGET_MAP, resolve_target_modules

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def base_model():
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL)


# ---------------------------------------------------------------------------
# Target module resolution
# ---------------------------------------------------------------------------


class TestTargetModules:
    def test_explicit_list_passthrough(self, base_model):
        result = resolve_target_modules(base_model, ["q_proj", "k_proj"])
        assert result == ["q_proj", "k_proj"]

    def test_auto_resolves_llama(self, base_model):
        result = resolve_target_modules(base_model, "auto")
        # tiny-random-LlamaForCausalLM should match LlamaForCausalLM
        assert "q_proj" in result

    def test_arch_map_has_common_archs(self):
        for arch in ["LlamaForCausalLM", "MistralForCausalLM", "GPTNeoXForCausalLM"]:
            assert arch in ARCH_TARGET_MAP


# ---------------------------------------------------------------------------
# Apply / Save / Load
# ---------------------------------------------------------------------------


class TestAdapterIO:
    def test_apply_lora(self, base_model):
        lora_cfg = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
        model_cfg = ModelConfig(base_model=TINY_MODEL)
        peft_model = apply_lora(base_model, lora_cfg, model_cfg)
        assert isinstance(peft_model, PeftModel)

    def test_save_and_reload(self, base_model, tmp_path):
        lora_cfg = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
        model_cfg = ModelConfig(base_model=TINY_MODEL)
        peft_model = apply_lora(base_model, lora_cfg, model_cfg)

        adapter_dir = tmp_path / "adapter"
        save_adapter(peft_model, adapter_dir, metadata={"test": True})

        assert (adapter_dir / "adapter_config.json").exists()
        assert (adapter_dir / "easylora_metadata.json").exists()

        reloaded = load_adapter(TINY_MODEL, adapter_dir, device_map=None)
        assert isinstance(reloaded, PeftModel)

    def test_saved_weights_match(self, base_model, tmp_path):
        lora_cfg = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
        model_cfg = ModelConfig(base_model=TINY_MODEL)
        peft_model = apply_lora(base_model, lora_cfg, model_cfg)

        adapter_dir = tmp_path / "adapter_match"
        save_adapter(peft_model, adapter_dir)

        reloaded = load_adapter(TINY_MODEL, adapter_dir, device_map=None)

        for (n1, p1), (_n2, p2) in zip(
            peft_model.named_parameters(), reloaded.named_parameters(), strict=False
        ):
            if "lora" in n1:
                assert torch.allclose(p1.data, p2.data), f"Mismatch in {n1}"
