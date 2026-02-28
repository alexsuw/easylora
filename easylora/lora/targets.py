"""Auto-detection of LoRA target modules per model architecture.

Maps well-known HuggingFace architecture class names to their attention
projection module names. Falls back to scanning the model for ``nn.Linear``
layers when the architecture is unknown.

The mapping is loaded from ``targets_registry.yaml`` (data-driven) so that
new architectures can be added without code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch.nn as nn
import yaml

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent / "targets_registry.yaml"

# Attention-like module name patterns, preferred during fallback linear scan
_ATTN_PATTERNS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "query_key_value",
        "qkv_proj",
        "c_attn",
        "c_proj",
        "Wqkv",
    }
)

_SKIP_MODULES = frozenset(
    {
        "lm_head",
        "embed_out",
        "embed_tokens",
        "wte",
        "wpe",
        "score",
    }
)


def _load_registry() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load architecture and model_type target maps from the YAML registry."""
    if not _REGISTRY_PATH.exists():
        return {}, {}

    data = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    arch_map: dict[str, list[str]] = data.get("architectures", {})
    type_map: dict[str, list[str]] = data.get("model_types", {})
    return arch_map, type_map


ARCH_TARGET_MAP, MODEL_TYPE_TARGET_MAP = _load_registry()


def resolve_target_modules(
    model: PreTrainedModel,
    target_modules: str | list[str],
) -> list[str]:
    """Resolve which modules to apply LoRA adapters to.

    Args:
        model: The loaded base model.
        target_modules: ``"auto"`` to choose by architecture heuristic,
            or an explicit list of module name patterns.

    Returns:
        List of module name patterns for PEFT ``LoraConfig.target_modules``.
    """
    if isinstance(target_modules, list):
        return target_modules

    if target_modules != "auto":
        return [target_modules]

    arch = _get_arch_name(model)
    if arch and arch in ARCH_TARGET_MAP:
        targets = ARCH_TARGET_MAP[arch]
        logger.info("Auto-detected LoRA targets for %s: %s", arch, targets)
        return targets

    model_type = getattr(model.config, "model_type", None)
    if model_type and model_type in MODEL_TYPE_TARGET_MAP:
        targets = MODEL_TYPE_TARGET_MAP[model_type]
        logger.info("Auto-detected LoRA targets via model_type '%s': %s", model_type, targets)
        return targets

    targets = _scan_linear_modules(model)
    if targets:
        logger.info(
            "Architecture %s not in target map — scanned model and found: %s",
            arch,
            targets,
        )
        return targets

    fallback = ["q_proj", "v_proj"]
    logger.warning("Could not detect LoRA targets for %s; falling back to %s", arch, fallback)
    return fallback


def resolve_target_modules_from_config(
    config: Any,
) -> tuple[list[str], str]:
    """Resolve target modules from a HuggingFace model config (no loaded model needed).

    Args:
        config: A ``transformers.AutoConfig`` instance.

    Returns:
        Tuple of (target_modules, detection_source_description).
    """
    archs = getattr(config, "architectures", None)
    arch = archs[0] if archs else None
    model_type = getattr(config, "model_type", None)

    if arch and arch in ARCH_TARGET_MAP:
        return ARCH_TARGET_MAP[arch], f"architecture registry ({arch})"

    if model_type and model_type in MODEL_TYPE_TARGET_MAP:
        return MODEL_TYPE_TARGET_MAP[model_type], f"model_type registry ({model_type})"

    fallback = ["q_proj", "v_proj"]
    return fallback, "fallback (architecture not in registry)"


def _get_arch_name(model: PreTrainedModel) -> str | None:
    archs = getattr(model.config, "architectures", None)
    if archs:
        return archs[0]
    return type(model).__name__


def _scan_linear_modules(model: PreTrainedModel) -> list[str]:
    """Find unique leaf ``nn.Linear`` module name suffixes in the model.

    Prefers attention-like patterns when found, otherwise returns all linear
    module suffixes (excluding output heads).
    """
    linear_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            if suffix not in _SKIP_MODULES:
                linear_names.add(suffix)

    attn_names = linear_names & _ATTN_PATTERNS
    if attn_names:
        return sorted(attn_names)
    return sorted(linear_names)
