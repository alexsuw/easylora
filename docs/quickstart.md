# Quickstart

## Installation

```bash
pip install easylora
```

For QLoRA (4-bit quantisation), also install bitsandbytes:

```bash
pip install bitsandbytes
```

For development:

```bash
git clone https://github.com/alexsuw/easylora.git
cd easylora
pip install -e ".[dev]"
```

## Python API

### Minimal Training

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
    model=ModelConfig(
        base_model="meta-llama/Llama-3.2-1B",
        load_in_4bit=True,
    ),
    data=DataConfig(
        dataset_name="tatsu-lab/alpaca",
        format="alpaca",
    ),
)
artifacts = train(config)
```

### Autopilot (No Manual Config)

```python
from easylora import autopilot_plan, autopilot_train

plan = autopilot_plan(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
    quality="balanced",
)
for line in plan.to_pretty_lines():
    print(line)

artifacts = autopilot_train(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
)
```

### Using the Trainer Directly

```python
from easylora import EasyLoRATrainer, TrainConfig
from easylora.config import ModelConfig, DataConfig, OutputConfig

config = TrainConfig(
    model=ModelConfig(base_model="meta-llama/Llama-3.2-1B"),
    data=DataConfig(dataset_path="my_data.jsonl", format="raw"),
    output=OutputConfig(output_dir="./my_run"),
)

trainer = EasyLoRATrainer(config)
artifacts = trainer.fit()
results = trainer.evaluate()
trainer.merge_and_save("./merged_model")
```

## CLI

### Train with a Config File

```bash
easylora train --config config.yaml
```

### Train with Autopilot

```bash
easylora train \
    --autopilot \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --quality balanced
```

### Dry-run an Autopilot Plan

```bash
easylora autopilot plan \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --print-config
```

### Generate a Starter Config

```bash
easylora init-config --template sft-lora
```

### Override Config Values

```bash
easylora train --config config.yaml \
    --set training.max_steps=100 \
    --set optim.lr=1e-4
```

### Validate Without Training

```bash
easylora train --config config.yaml --dry-run
```

### Check Your Environment

```bash
easylora doctor
```

## Next Steps

- See the [configuration reference](configuration.md) for all options
- See the [CLI reference](cli.md) for all commands
- See [adapters](adapters.md) for save/load/merge workflows
