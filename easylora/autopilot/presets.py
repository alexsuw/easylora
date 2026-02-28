"""Quality presets for autopilot planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AutopilotQuality = Literal["fast", "balanced", "high"]


@dataclass(frozen=True)
class QualityPreset:
    name: AutopilotQuality
    base_lora_rank: int
    base_lr: float
    default_epochs: int
    max_steps_large_dataset: int
    target_effective_batch: int
    val_split_ratio: float


_PRESETS: dict[AutopilotQuality, QualityPreset] = {
    "fast": QualityPreset(
        name="fast",
        base_lora_rank=8,
        base_lr=3e-4,
        default_epochs=2,
        max_steps_large_dataset=1500,
        target_effective_batch=8,
        val_split_ratio=0.01,
    ),
    "balanced": QualityPreset(
        name="balanced",
        base_lora_rank=16,
        base_lr=2e-4,
        default_epochs=3,
        max_steps_large_dataset=3000,
        target_effective_batch=16,
        val_split_ratio=0.02,
    ),
    "high": QualityPreset(
        name="high",
        base_lora_rank=32,
        base_lr=1.2e-4,
        default_epochs=4,
        max_steps_large_dataset=6000,
        target_effective_batch=32,
        val_split_ratio=0.03,
    ),
}


def get_quality_preset(quality: AutopilotQuality) -> QualityPreset:
    return _PRESETS[quality]
