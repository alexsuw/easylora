# CLI Reference

All commands are available via the `easylora` CLI.

```bash
easylora --help
```

## `easylora train`

Fine-tune a model with LoRA / QLoRA.

```bash
easylora train --config config.yaml [OPTIONS]
```

| Option | Description |
|---|---|
| `--config`, `-c` | Path to YAML/JSON config file (required unless `--autopilot`) |
| `--autopilot` | Auto-plan training from model + dataset |
| `--model`, `-m` | Base model ID/path (required with `--autopilot`) |
| `--dataset`, `-d` | Dataset path or HF name (required with `--autopilot`) |
| `--quality` | Autopilot quality preset: `fast`, `balanced`, `high` |
| `--output-dir` | Output directory used in autopilot mode |
| `--subset` | Dataset subset/config name for autopilot mode |
| `--split` | Dataset split for autopilot mode (default: `train`) |
| `--seed` | Seed for reproducibility |
| `--force` | Allow overwriting output directory |
| `--set` | Override config value as `key=value` (repeatable) |
| `--dry-run` | Validate config and print settings without training |
| `--print-config` | Print resolved config as YAML and exit |

**Examples:**

```bash
# Basic training
easylora train --config config.yaml

# Autopilot training (no config file)
easylora train \
    --autopilot \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca

# Override learning rate
easylora train --config config.yaml --set optim.lr=1e-5

# Validate without training
easylora train --config config.yaml --dry-run

# Overwrite existing output
easylora train --config config.yaml --force
```

## `easylora autopilot plan`

Generate and print autopilot decisions without training.

```bash
easylora autopilot plan [OPTIONS]
```

| Option | Description |
|---|---|
| `--model`, `-m` | Base model ID/path (required) |
| `--dataset`, `-d` | Dataset path or HF name (required) |
| `--quality` | Preset: `fast`, `balanced`, `high` |
| `--output-dir` | Planned output directory |
| `--subset` | Dataset subset/config name |
| `--split` | Dataset split (default: `train`) |
| `--print-config` | Print resolved TrainConfig YAML |

**Example:**

```bash
easylora autopilot plan \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --quality balanced \
    --print-config
```

## `easylora eval`

Evaluate a LoRA adapter with perplexity and optional generation.

```bash
easylora eval [OPTIONS]
```

| Option | Description |
|---|---|
| `--base-model`, `-m` | Base model ID or path (required) |
| `--adapter-dir`, `-a` | Path to saved adapter (required) |
| `--dataset`, `-d` | Dataset path or HF name (required) |
| `--max-samples` | Max samples to evaluate |
| `--max-seq-len` | Max sequence length (default: 2048) |
| `--prompt`, `-p` | Prompts for generation sanity check (repeatable) |

**Example:**

```bash
easylora eval \
    --base-model meta-llama/Llama-3.2-1B \
    --adapter-dir ./output/adapter \
    --dataset eval_data.jsonl \
    --prompt "What is machine learning?"
```

## `easylora merge`

Merge a LoRA adapter into the base model and save full weights.

```bash
easylora merge [OPTIONS]
```

| Option | Description |
|---|---|
| `--base-model`, `-m` | Base model ID or path (required) |
| `--adapter-dir`, `-a` | Path to saved adapter (required) |
| `--output-dir`, `-o` | Where to save merged model (required) |
| `--trust-remote-code` | Trust remote code |

**Example:**

```bash
easylora merge \
    --base-model meta-llama/Llama-3.2-1B \
    --adapter-dir ./output/adapter \
    --output-dir ./merged_model
```

## `easylora doctor`

Print environment diagnostics for debugging.

```bash
easylora doctor
```

Shows: Python version, PyTorch version, CUDA availability, GPU name, bf16
support, bitsandbytes availability, and versions of key dependencies.

## `easylora inspect-targets`

Show recommended LoRA target modules for a model.

```bash
easylora inspect-targets --model <model_id> [--trust-remote-code]
```

| Option | Description |
|---|---|
| `--model`, `-m` | HF model ID or path (required) |
| `--trust-remote-code` | Trust remote code |

**Example:**

```bash
easylora inspect-targets --model meta-llama/Llama-3.2-1B
```

## `easylora init-config`

Generate a starter config file.

```bash
easylora init-config --template <template_name> [--output config.yaml]
```

| Option | Description |
|---|---|
| `--template`, `-t` | Template name: `sft-lora` or `sft-qlora` (required) |
| `--output`, `-o` | Output file path (default: `easylora_config.yaml`) |

**Example:**

```bash
easylora init-config --template sft-qlora
```
