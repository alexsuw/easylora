"""Autopilot planner that resolves a full TrainConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from easylora.autopilot.dataset_analysis import DatasetProfile, analyze_dataset
from easylora.autopilot.hardware import HardwareProfile, detect_hardware
from easylora.autopilot.heuristics import PlanDecision, resolve_plan
from easylora.autopilot.model_analysis import ModelProfile, analyze_model
from easylora.autopilot.presets import AutopilotQuality, get_quality_preset
from easylora.config import (
    DataConfig,
    LoRAConfig,
    ModelConfig,
    OptimConfig,
    OutputConfig,
    ReproConfig,
    TrainConfig,
    TrainLoopConfig,
)
from easylora.utils.hf import load_tokenizer


@dataclass
class AutopilotPlan:
    """Resolved config plus analysis data and reasoning."""

    config: TrainConfig
    quality: AutopilotQuality
    hardware: HardwareProfile
    dataset: DatasetProfile
    model: ModelProfile
    decision: PlanDecision

    def report(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "hardware": self.hardware.to_dict(),
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_dict(),
            "decision": self.decision.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(),
            "report": self.report(),
        }

    def to_pretty_lines(self) -> list[str]:
        gpu_display = self.hardware.gpu_name or "CPU/MPS"
        return [
            "AUTOPILOT PLAN",
            "",
            f"Model: {self.model.model_name}",
            f"Dataset samples: {self.dataset.examples:,}",
            f"GPU: {gpu_display}",
            "",
            f"Strategy: {'QLoRA' if self.decision.use_qlora else 'LoRA'}",
            f"Seq length: {self.decision.max_seq_len}",
            f"Batch size: {self.decision.batch_size}",
            f"Grad accum: {self.decision.grad_accum}",
            f"LR: {self.decision.learning_rate:.2e}",
            f"Estimated VRAM: ~{self.decision.estimated_vram_gb} GB",
            f"Estimated speed: ~{self.decision.estimated_steps_per_sec} steps/sec",
        ]


def plan_autopilot(
    *,
    model: str,
    dataset: str,
    quality: AutopilotQuality = "balanced",
    output_dir: str = "./output",
    subset: str | None = None,
    split: str = "train",
    trust_remote_code: bool = False,
    allow_overwrite: bool = False,
    seed: int = 42,
    max_analysis_samples: int = 512,
) -> AutopilotPlan:
    """Build an autopilot plan and return a fully validated TrainConfig."""
    hardware = detect_hardware()
    model_profile = analyze_model(model, trust_remote_code=trust_remote_code)
    tokenizer = load_tokenizer(
        ModelConfig(
            base_model=model,
            trust_remote_code=trust_remote_code,
        )
    )
    dataset_profile = analyze_dataset(
        dataset,
        tokenizer=tokenizer,
        split=split,
        subset=subset,
        max_samples=max_analysis_samples,
    )
    preset = get_quality_preset(quality)
    decision = resolve_plan(hardware, dataset_profile, model_profile, preset)

    source_path = dataset if _is_local_path(dataset) else None
    source_name = None if source_path else dataset

    cfg = TrainConfig(
        model=ModelConfig(
            base_model=model,
            trust_remote_code=trust_remote_code,
            load_in_4bit=decision.use_qlora,
            torch_dtype="auto",
        ),
        data=DataConfig(
            dataset_path=source_path,
            dataset_name=source_name,
            subset=subset,
            split=split,
            format=dataset_profile.inferred_format,  # type: ignore[arg-type]
            text_field=dataset_profile.text_field,
            max_seq_len=decision.max_seq_len,
            val_split_ratio=preset.val_split_ratio,
        ),
        lora=LoRAConfig(
            r=decision.lora_r,
            alpha=decision.lora_alpha,
            target_modules="auto",
        ),
        optim=OptimConfig(
            lr=decision.learning_rate,
        ),
        training=TrainLoopConfig(
            epochs=decision.epochs,
            max_steps=decision.max_steps,
            batch_size=decision.batch_size,
            grad_accum=decision.grad_accum,
            save_steps=decision.save_steps,
            logging_steps=decision.logging_steps,
            gradient_checkpointing=True,
        ),
        output=OutputConfig(
            output_dir=output_dir,
            allow_overwrite=allow_overwrite,
        ),
        repro=ReproConfig(
            seed=seed,
            deterministic=False,
        ),
    )
    return AutopilotPlan(
        config=cfg,
        quality=quality,
        hardware=hardware,
        dataset=dataset_profile,
        model=model_profile,
        decision=decision,
    )


def _is_local_path(dataset: str) -> bool:
    from pathlib import Path

    return Path(dataset).exists()
