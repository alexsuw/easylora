"""Merge LoRA adapter weights into a base model and save the result."""

from __future__ import annotations

import logging
from pathlib import Path

from easylora.lora.adapter import load_adapter
from easylora.utils.hf import load_tokenizer

logger = logging.getLogger(__name__)


def merge_adapter(
    base_model_name_or_path: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    *,
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
) -> Path:
    """Merge LoRA adapter into the base model and save the full merged weights.

    The merged model can be loaded with ``AutoModelForCausalLM.from_pretrained``
    without any PEFT dependency.

    Args:
        base_model_name_or_path: HF model ID or local path.
        adapter_dir: Directory containing saved LoRA adapter.
        output_dir: Where to save the merged model + tokenizer.
        device_map: Device placement strategy.
        trust_remote_code: Trust remote code.
        torch_dtype: Dtype for loading.

    Returns:
        Path to the merged model directory.
    """
    from easylora.config import ModelConfig

    peft_model = load_adapter(
        base_model_name_or_path,
        adapter_dir,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    merged = peft_model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)

    tok_cfg = ModelConfig(
        base_model=base_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = load_tokenizer(tok_cfg)
    tokenizer.save_pretrained(out)

    logger.info("Merged model saved to %s", out)
    return out
