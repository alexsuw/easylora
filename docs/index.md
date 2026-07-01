# easylora

**Zero-config Autopilot for LoRA / QLoRA fine-tuning.**

easylora profiles your hardware, model, and dataset, chooses safe training
settings, and writes a shareable report explaining every decision.

```bash
easylora train --autopilot \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --quality balanced
```

## What you get

- **Autopilot planning** for LoRA vs QLoRA, sequence length, batch size, rank,
  learning rate, and duration.
- **Explainable artifacts**: `resolved_config.yaml`, `autopilot_report.json`,
  `autopilot_report.md`, logs, summaries, and model cards.
- **Config-driven escape hatch** for users who want full control with YAML/JSON.
- **Model-native formatting** via `format: "auto"` and tokenizer chat templates.
- **Evaluation reports** with perplexity and base-vs-adapter generations.
- **TRL-backed DPO** for preference tuning without leaving easylora configs.
- **Portable adapters**: save, load, merge, and publish to Hugging Face Hub.

## Quick example

```python
from easylora import autopilot_plan, autopilot_train

plan = autopilot_plan(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
)
print(plan.to_markdown())

artifacts = autopilot_train(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
)
```

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) freezes pre-trained model weights and injects
small trainable rank-decomposition matrices, reducing trainable parameters by
orders of magnitude.

**QLoRA** combines LoRA with 4-bit quantisation of the base model, enabling
fine-tuning of large models on consumer GPUs.

## Next Steps

- [Quickstart guide](quickstart.md) for detailed setup instructions
- [Autopilot guide](autopilot.md) for transparent planning
- [Recipes](recipes.md) for copy-paste training scenarios
- [Benchmarks](benchmarks.md) for reproducible speed and memory checks
- [Configuration reference](configuration.md) for all available options
- [CLI reference](cli.md) for command-line usage
- [Model support](model-support.md) for supported architectures
