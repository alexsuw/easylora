"""LoRA adapter creation, saving, and loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, TaskType, get_peft_model

from easylora.exceptions import EasyLoRAConfigError
from easylora.lora.targets import resolve_target_modules

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from easylora.config import LoRAConfig, ModelConfig

logger = logging.getLogger(__name__)


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoRAConfig,
    model_config: ModelConfig,
) -> PeftModel:
    """Wrap *model* with LoRA adapters according to *lora_config*.

    If the model was loaded in 4-bit or 8-bit mode, the model is first
    prepared for k-bit training.
    """
    if model_config.load_in_4bit or model_config.load_in_8bit:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise EasyLoRAConfigError(
                "prepare_model_for_kbit_training requires peft >= 0.10. "
                "Please upgrade: pip install -U peft"
            ) from exc
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    targets = resolve_target_modules(model, lora_config.target_modules)

    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "SEQ_CLS": TaskType.SEQ_CLS,
    }
    task = task_type_map.get(lora_config.task_type)
    if task is None:
        raise EasyLoRAConfigError(
            f"Unknown task_type '{lora_config.task_type}'. Supported: {list(task_type_map.keys())}"
        )

    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=targets,
        bias=lora_config.bias,
        task_type=task,
        modules_to_save=lora_config.modules_to_save,
    )

    peft_model: PeftModel = get_peft_model(model, peft_config)  # type: ignore[assignment]
    trainable, total = peft_model.get_nb_trainable_parameters()
    pct = 100 * trainable / total if total > 0 else 0
    logger.info(
        "LoRA applied — trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
    return peft_model


def save_adapter(
    model: PeftModel,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save LoRA adapter weights and optional metadata.

    Args:
        model: A PEFT model with LoRA adapters.
        output_dir: Directory to write adapter files into.
        metadata: Optional dict saved as ``easylora_metadata.json``.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    if metadata:
        meta_path = out / "easylora_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    logger.info("Adapter saved to %s", out)
    return out


def load_adapter(
    base_model_name_or_path: str,
    adapter_dir: str | Path,
    *,
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
) -> PeftModel:
    """Load a base model and apply a saved LoRA adapter on top.

    Args:
        base_model_name_or_path: HF model ID or local path for the base model.
        adapter_dir: Directory containing the saved adapter.
        device_map: Device placement strategy.
        trust_remote_code: Whether to trust remote code in the model.
        torch_dtype: Dtype for loading (``"auto"`` to infer).

    Returns:
        A PeftModel with the adapter loaded.
    """
    from easylora.config import ModelConfig
    from easylora.utils.hf import load_base_model

    cfg = ModelConfig(
        base_model=base_model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,  # type: ignore[arg-type]
        device_map=device_map,
    )
    base = load_base_model(cfg)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    logger.info("Adapter loaded from %s onto %s", adapter_dir, base_model_name_or_path)
    return peft_model
