"""Public autopilot APIs for planning and training."""

from __future__ import annotations

from pathlib import Path

from easylora.autopilot.planner import AutopilotPlan, plan_autopilot
from easylora.autopilot.presets import AutopilotQuality
from easylora.config import RunArtifacts
from easylora.train.trainer import EasyLoRATrainer
from easylora.utils.io import save_json, save_yaml


def autopilot_plan(
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
    """Resolve autopilot decisions into a full `TrainConfig`."""
    return plan_autopilot(
        model=model,
        dataset=dataset,
        quality=quality,
        output_dir=output_dir,
        subset=subset,
        split=split,
        trust_remote_code=trust_remote_code,
        allow_overwrite=allow_overwrite,
        seed=seed,
        max_analysis_samples=max_analysis_samples,
    )


def autopilot_train(
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
) -> RunArtifacts:
    """Run autopilot planning and execute training with resolved config."""
    plan = autopilot_plan(
        model=model,
        dataset=dataset,
        quality=quality,
        output_dir=output_dir,
        subset=subset,
        split=split,
        trust_remote_code=trust_remote_code,
        allow_overwrite=allow_overwrite,
        seed=seed,
        max_analysis_samples=max_analysis_samples,
    )
    trainer = EasyLoRATrainer(plan.config)
    artifacts = trainer.fit()
    _save_autopilot_artifacts(plan, Path(plan.config.output.output_dir))
    return artifacts


def _save_autopilot_artifacts(plan: AutopilotPlan, output_dir: Path) -> None:
    save_yaml(plan.config.model_dump(), output_dir / "resolved_config.yaml")
    save_json(plan.report(), output_dir / "autopilot_report.json")
