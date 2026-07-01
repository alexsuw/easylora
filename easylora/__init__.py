"""easylora — batteries-included LoRA / QLoRA fine-tuning toolkit.

Public API::

    from easylora import train, TrainConfig, EasyLoRATrainer
    from easylora import autopilot_plan, autopilot_train, save_autopilot_report
    from easylora import train_dpo
    from easylora import save_adapter, load_adapter, merge_adapter
"""

from __future__ import annotations

from importlib.metadata import version

__version__: str = version("easylora")

from easylora.align import train_dpo
from easylora.autopilot.api import autopilot_plan, autopilot_train, save_autopilot_report
from easylora.config import (
    DataConfig,
    DPOConfig,
    DPOTrainConfig,
    ModelConfig,
    PreferenceDataConfig,
    RunArtifacts,
    TrainConfig,
    load_config,
)
from easylora.lora.adapter import load_adapter, save_adapter
from easylora.lora.merge import merge_adapter
from easylora.train.trainer import EasyLoRATrainer

# TODO: Multi-node distributed training (out of scope for MVP)
# TODO: Vision-language model support (out of scope for MVP)
# TODO: Full Diffusers adapter support (architecture stubs only for now)
# TODO: Custom CUDA kernels (out of scope for MVP)


def train(config: TrainConfig) -> RunArtifacts:
    """Fine-tune a model with LoRA / QLoRA in a single call.

    This is the simplest entry point — it creates an ``EasyLoRATrainer``
    and runs the full pipeline.

    Args:
        config: Fully validated ``TrainConfig``.

    Returns:
        ``RunArtifacts`` describing all output paths.
    """
    return EasyLoRATrainer(config).fit()


__all__ = [
    "DPOConfig",
    "DPOTrainConfig",
    "DataConfig",
    "EasyLoRATrainer",
    "ModelConfig",
    "PreferenceDataConfig",
    "RunArtifacts",
    "TrainConfig",
    "autopilot_plan",
    "autopilot_train",
    "load_adapter",
    "load_config",
    "merge_adapter",
    "save_adapter",
    "save_autopilot_report",
    "train",
    "train_dpo",
]
