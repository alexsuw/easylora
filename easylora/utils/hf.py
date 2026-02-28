"""Hugging Face model / tokenizer loading helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from easylora.exceptions import EasyLoRADependencyError

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from easylora.config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, torch.dtype | str] = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_dtype(dtype_str: str) -> torch.dtype | str:
    return _DTYPE_MAP[dtype_str]


def _check_bitsandbytes() -> None:
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise EasyLoRADependencyError(
            package="bitsandbytes",
            feature="4-bit / 8-bit quantisation (QLoRA)",
        ) from exc


def _build_quantisation_config(model_config: ModelConfig) -> BitsAndBytesConfig | None:
    if not model_config.load_in_4bit and not model_config.load_in_8bit:
        return None

    _check_bitsandbytes()

    if model_config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_base_model(model_config: ModelConfig) -> PreTrainedModel:
    """Load a causal-LM base model according to *model_config*.

    Handles dtype selection, quantisation, device mapping, and
    trust_remote_code.
    """
    quant_config = _build_quantisation_config(model_config)
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_config.base_model,
        "trust_remote_code": model_config.trust_remote_code,
        "device_map": model_config.device_map,
    }

    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    else:
        kwargs["torch_dtype"] = _resolve_dtype(model_config.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    logger.info("Loaded base model: %s  (dtype=%s)", model_config.base_model, model.dtype)
    return model


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    """Load the tokenizer, fixing common footguns (missing pad token, wrong padding side)."""
    tok_name = model_config.tokenizer or model_config.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token was None — set to eos_token ('%s')", tokenizer.eos_token)

    # Causal LM convention: pad on the right so attention sees prefix first
    tokenizer.padding_side = "right"
    return tokenizer
