"""Configuration schema for easylora.

Uses Pydantic v2 for validation because:
- Field-level validators with clear error messages catch invalid configs early
- `model_config = ConfigDict(extra="forbid")` catches typos in YAML/JSON
- Built-in serialisation via `.model_dump()` / `.model_validate()`
- Nested model composition maps naturally to TrainConfig's sub-configs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from easylora.exceptions import EasyLoRAConfigError


class ModelConfig(BaseModel):
    """Base model and tokenizer settings."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(..., description="HF model ID or local path (required)")
    tokenizer: str | None = Field(None, description="Tokenizer ID or path; defaults to base_model")
    trust_remote_code: bool = False
    torch_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device_map: str | dict[str, Any] | None = "auto"

    @model_validator(mode="after")
    def _check_quantisation_exclusivity(self) -> ModelConfig:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive. Pick one.")
        return self


class DataConfig(BaseModel):
    """Dataset loading and formatting settings."""

    model_config = ConfigDict(extra="forbid")

    dataset_path: str | None = Field(None, description="Local path to JSON/JSONL/CSV file")
    dataset_name: str | None = Field(None, description="HuggingFace Datasets hub identifier")
    subset: str | None = None
    split: str = "train"
    text_field: str = "text"
    format: Literal["alpaca", "chatml", "raw"] = "raw"
    max_seq_len: int = Field(2048, ge=32, le=131072)
    val_split_ratio: float = Field(0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check_dataset_source(self) -> DataConfig:
        if not self.dataset_path and not self.dataset_name:
            raise ValueError(
                "Provide either 'dataset_path' (local file) or 'dataset_name' (HF hub)."
            )
        return self


class LoRAConfig(BaseModel):
    """LoRA adapter hyper-parameters."""

    model_config = ConfigDict(extra="forbid")

    r: int = Field(16, ge=1, description="LoRA rank")
    alpha: int = Field(32, ge=1, description="LoRA alpha (scaling = alpha / r)")
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: str | list[str] = Field(
        "auto",
        description=(
            '"auto" to pick targets by architecture, or an explicit list like ["q_proj", "v_proj"]'
        ),
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] | None = None


class OptimConfig(BaseModel):
    """Optimiser settings."""

    model_config = ConfigDict(extra="forbid")

    lr: float = Field(2e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = Field(0.03, ge=0.0, le=1.0)
    scheduler: str = "cosine"


class TrainLoopConfig(BaseModel):
    """Training loop settings."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(3, ge=1)
    batch_size: int = Field(4, ge=1)
    grad_accum: int = Field(4, ge=1)
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int = Field(10, ge=1)
    gradient_checkpointing: bool = True
    max_steps: int = Field(-1, description="-1 means use epochs")


class OutputConfig(BaseModel):
    """Output and Hub publishing settings."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./output"
    run_name: str | None = None
    save_total_limit: int | None = Field(3, ge=1)
    push_to_hub: bool = False
    hub_repo_id: str | None = None
    hub_private: bool = True
    allow_overwrite: bool = False

    @model_validator(mode="after")
    def _check_hub(self) -> OutputConfig:
        if self.push_to_hub and not self.hub_repo_id:
            raise ValueError("hub_repo_id is required when push_to_hub=True.")
        return self


class ReproConfig(BaseModel):
    """Reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    deterministic: bool = False


class TrainConfig(BaseModel):
    """Top-level training configuration that nests all sub-configs."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    data: DataConfig
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig()
    training: TrainLoopConfig = TrainLoopConfig()
    output: OutputConfig = OutputConfig()
    repro: ReproConfig = ReproConfig()


class RunArtifacts(BaseModel):
    """Describes paths produced by a training run."""

    adapter_dir: str
    config_path: str
    log_path: str
    summary_path: str
    merged_dir: str | None = None


def load_config(path: str | Path) -> TrainConfig:
    """Load a TrainConfig from a YAML or JSON file.

    Args:
        path: Path to a `.yaml`, `.yml`, or `.json` config file.

    Returns:
        Validated TrainConfig instance.

    Raises:
        EasyLoRAConfigError: If the file cannot be read or parsed.
    """
    path = Path(path)
    if not path.exists():
        raise EasyLoRAConfigError(f"Config file not found: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EasyLoRAConfigError(f"Cannot read config file: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(raw)
        elif suffix == ".json":
            data = json.loads(raw)
        else:
            raise EasyLoRAConfigError(
                f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json"
            )
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise EasyLoRAConfigError(f"Failed to parse config file: {exc}") from exc

    if not isinstance(data, dict):
        raise EasyLoRAConfigError("Config file must contain a top-level mapping/object.")

    try:
        return TrainConfig.model_validate(data)
    except Exception as exc:
        raise EasyLoRAConfigError(f"Config validation failed: {exc}") from exc
