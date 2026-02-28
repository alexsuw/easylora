# Configuration Reference

easylora uses a config-driven design. All settings are defined in a `TrainConfig`
object (Pydantic v2) which can be loaded from YAML or JSON files.

## Config File Format

=== "YAML"

    ```yaml
    model:
      base_model: "meta-llama/Llama-3.2-1B"
      torch_dtype: "auto"

    data:
      dataset_name: "tatsu-lab/alpaca"
      format: "alpaca"
      max_seq_len: 2048

    lora:
      r: 16
      alpha: 32
      target_modules: "auto"

    training:
      epochs: 3
      batch_size: 4

    output:
      output_dir: "./output"
    ```

=== "JSON"

    ```json
    {
      "model": {
        "base_model": "meta-llama/Llama-3.2-1B"
      },
      "data": {
        "dataset_name": "tatsu-lab/alpaca",
        "format": "alpaca"
      }
    }
    ```

## Sections

### `model` -- ModelConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `base_model` | str | **required** | HF model ID or local path |
| `tokenizer` | str | base_model | Tokenizer ID (if different from model) |
| `trust_remote_code` | bool | `false` | Trust remote code in model/tokenizer |
| `torch_dtype` | str | `"auto"` | `"auto"`, `"bf16"`, `"fp16"`, `"fp32"` |
| `load_in_4bit` | bool | `false` | QLoRA 4-bit quantisation |
| `load_in_8bit` | bool | `false` | 8-bit quantisation |
| `device_map` | str | `"auto"` | Device placement strategy |

!!! warning "Quantisation exclusivity"
    `load_in_4bit` and `load_in_8bit` are mutually exclusive. Setting both
    to `true` will raise a validation error.

### `data` -- DataConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `dataset_path` | str | null | Local file path (JSON/JSONL/CSV/Parquet) |
| `dataset_name` | str | null | HuggingFace Datasets identifier |
| `subset` | str | null | Dataset subset/config name |
| `split` | str | `"train"` | Dataset split |
| `text_field` | str | `"text"` | Column name for raw format |
| `format` | str | `"raw"` | `"raw"`, `"alpaca"`, or `"chatml"` |
| `max_seq_len` | int | `2048` | Maximum sequence length (32--131072) |
| `val_split_ratio` | float | `0.0` | Fraction for validation split |

!!! note "Dataset source"
    You must provide either `dataset_path` (local file) or `dataset_name`
    (HF hub). Providing neither raises a validation error.

### `lora` -- LoRAConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `r` | int | `16` | LoRA rank |
| `alpha` | int | `32` | LoRA alpha (scaling = alpha/r) |
| `dropout` | float | `0.05` | LoRA dropout |
| `target_modules` | str or list | `"auto"` | Modules to adapt |
| `bias` | str | `"none"` | `"none"`, `"all"`, or `"lora_only"` |
| `task_type` | str | `"CAUSAL_LM"` | PEFT task type |
| `modules_to_save` | list | null | Extra modules to train fully |

### `optim` -- OptimConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `2e-4` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `betas` | tuple | `(0.9, 0.999)` | Adam betas |
| `warmup_ratio` | float | `0.03` | Warmup fraction |
| `scheduler` | str | `"cosine"` | LR scheduler type |

### `training` -- TrainLoopConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | int | `3` | Number of training epochs |
| `batch_size` | int | `4` | Per-device batch size |
| `grad_accum` | int | `4` | Gradient accumulation steps |
| `logging_steps` | int | `10` | Log every N steps |
| `eval_steps` | int | null | Evaluate every N steps |
| `save_steps` | int | null | Save checkpoint every N steps |
| `gradient_checkpointing` | bool | `true` | Enable gradient checkpointing |
| `max_steps` | int | `-1` | Max steps (-1 = use epochs) |

### `output` -- OutputConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | str | `"./output"` | Output directory |
| `run_name` | str | null | Run name for logging |
| `save_total_limit` | int | `3` | Max checkpoints to keep |
| `push_to_hub` | bool | `false` | Push to HuggingFace Hub |
| `hub_repo_id` | str | null | Hub repository ID |
| `hub_private` | bool | `true` | Make hub repo private |
| `allow_overwrite` | bool | `false` | Allow overwriting output dir |

### `repro` -- ReproConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | int | `42` | Random seed |
| `deterministic` | bool | `false` | Fully deterministic mode |

## Loading Configs in Python

```python
from easylora.config import load_config

config = load_config("config.yaml")
```

## CLI Overrides

Use `--set` to override any config value:

```bash
easylora train --config config.yaml --set model.base_model=gpt2 --set optim.lr=1e-5
```
