# easylora

[![CI](https://github.com/alexsuw/easylora/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsuw/easylora/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/easylora.svg)](https://pypi.org/project/easylora/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

**Batteries-included toolkit for LoRA / QLoRA fine-tuning** with Hugging Face Transformers.

Fine-tune any causal language model with LoRA in under 20 lines of Python, or with a single CLI command.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexsuw/easylora/blob/main/notebooks/quickstart_colab.ipynb)

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) freezes pre-trained model weights and injects small trainable rank-decomposition matrices, reducing trainable parameters by orders of magnitude.

**QLoRA** adds 4-bit quantisation on top of LoRA, enabling fine-tuning of large models on consumer GPUs.

easylora wraps `transformers` + `peft` with safe defaults, reproducibility, and a clean config-driven API.

## Installation

```bash
pip install easylora
```

Optional extras:

```bash
pip install "easylora[qlora]"   # adds bitsandbytes for 4-bit/8-bit quantisation
pip install "easylora[wandb]"   # adds Weights & Biases logging
pip install "easylora[dev]"     # adds ruff, pyright, pytest, mkdocs, pre-commit
pip install "easylora[all]"     # everything
```

For development from source:

```bash
git clone https://github.com/alexsuw/easylora.git
cd easylora
pip install -e ".[dev]"
```

## Quickstart (Python)

```python
from easylora import train, TrainConfig
from easylora.config import ModelConfig, DataConfig

config = TrainConfig(
    model=ModelConfig(base_model="meta-llama/Llama-3.2-1B"),
    data=DataConfig(
        dataset_name="tatsu-lab/alpaca",
        format="alpaca",
        max_seq_len=2048,
    ),
)
artifacts = train(config)
print(f"Adapter saved to: {artifacts.adapter_dir}")
```

### QLoRA (4-bit)

```python
config = TrainConfig(
    model=ModelConfig(base_model="meta-llama/Llama-3.2-1B", load_in_4bit=True),
    data=DataConfig(dataset_name="tatsu-lab/alpaca", format="alpaca"),
)
artifacts = train(config)
```

## Quickstart (CLI)

```bash
# Generate a starter config
easylora init-config --template sft-lora

# Train
easylora train --config easylora_config.yaml

# Validate config without training
easylora train --config config.yaml --dry-run

# Evaluate
easylora eval --base-model meta-llama/Llama-3.2-1B --adapter-dir ./output/adapter --dataset eval.jsonl

# Merge adapter into base model
easylora merge --base-model meta-llama/Llama-3.2-1B --adapter-dir ./output/adapter --output-dir ./merged

# Check environment
easylora doctor
```

## Config Reference

Config files use YAML or JSON. See the [full reference](https://alexsuw.github.io/easylora/configuration/).

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
  grad_accum: 4
  gradient_checkpointing: true
output:
  output_dir: "./output"
repro:
  seed: 42
```

## Model Compatibility

easylora auto-detects LoRA target modules for 16+ architectures:

| Family | Models |
|---|---|
| LLaMA | LLaMA 1/2/3, Code Llama, Vicuna |
| Mistral | Mistral, Mixtral |
| Qwen | Qwen, Qwen2 |
| Google | Gemma, Gemma 2 |
| Microsoft | Phi-2, Phi-3 |
| Others | Falcon, GPT-NeoX, Pythia, MPT, Bloom, OPT, GPT-2, StarCoder |

For unknown architectures, easylora scans the model for `nn.Linear` layers and selects attention-like modules automatically. Use `easylora inspect-targets --model <id>` to preview what would be selected.

## Merging Adapters

```python
from easylora import merge_adapter

merge_adapter(
    base_model_name_or_path="meta-llama/Llama-3.2-1B",
    adapter_dir="./output/adapter",
    output_dir="./merged_model",
)
```

The merged model loads with `AutoModelForCausalLM.from_pretrained` without PEFT.

## Dataset Formats

| Format | Columns | Description |
|---|---|---|
| `raw` | `text` | Single text field for language modelling |
| `alpaca` | `instruction`, `input` (optional), `output` | Instruction-following with prompt masking |
| `chatml` | `messages` | Chat messages with role/content dicts |

## Output Artifacts

```
output/
  adapter/           # LoRA adapter weights
  train_config.json  # Config used for this run
  train_log.jsonl    # Step-by-step training metrics
  summary.json       # Final loss, steps, runtime
  metadata.json      # Base model, versions, timestamp
  logs.jsonl         # Application logs
```

## Troubleshooting

| Issue | Solution |
|---|---|
| `bitsandbytes not installed` | `pip install bitsandbytes` (CUDA required) |
| CUDA OOM | Reduce `batch_size`, increase `grad_accum`, enable `gradient_checkpointing`, use QLoRA |
| `pad_token was None` | Handled automatically (set to `eos_token`) |
| Output dir exists | Use `--force` or `allow_overwrite: true` |

Run `easylora doctor` for environment diagnostics.

## Running Tests

```bash
pip install -e ".[dev]"
make test          # fast tests
make test-slow     # includes smoke training
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, and the PR process.

## License

[Apache-2.0](LICENSE)
