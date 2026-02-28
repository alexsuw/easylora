# easylora

**Batteries-included toolkit for LoRA / QLoRA fine-tuning** with Hugging Face
Transformers.

Fine-tune any causal language model with LoRA in under 20 lines of Python, or
with a single CLI command.

## Features

- **LoRA and QLoRA** fine-tuning for causal language models
- **Auto target module detection** for 16+ architectures
- **Config-driven** workflow with Pydantic v2 validation
- **CLI and Python API** for training, evaluation, and adapter management
- **Safe defaults**: bf16, gradient checkpointing, pad token handling
- **Reproducible**: configs saved, seeds set, deterministic mode available
- **Portable artifacts**: save, load, merge, and push adapters to HF Hub

## Quick Example

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
```

Or via CLI:

```bash
easylora train --config config.yaml
```

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) freezes pre-trained model weights and injects
small trainable rank-decomposition matrices, reducing trainable parameters by
orders of magnitude.

**QLoRA** combines LoRA with 4-bit quantisation of the base model, enabling
fine-tuning of large models on consumer GPUs.

## Next Steps

- [Quickstart guide](quickstart.md) for detailed setup instructions
- [Configuration reference](configuration.md) for all available options
- [CLI reference](cli.md) for command-line usage
- [Model support](model-support.md) for supported architectures
