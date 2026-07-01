# easylora

[![CI](https://github.com/alexsuw/easylora/actions/workflows/ci.yml/badge.svg)](https://github.com/alexsuw/easylora/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/easylora.svg)](https://pypi.org/project/easylora/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)

**Zero-config Autopilot for LoRA / QLoRA fine-tuning.**

Fine-tune a Hugging Face causal LM with one command. easylora profiles your
hardware, model, and dataset, chooses safe LoRA/QLoRA settings, then writes a
shareable report explaining every decision.

```bash
easylora train --autopilot \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --quality balanced
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alexsuw/easylora/blob/main/notebooks/quickstart_colab.ipynb)

## Why easylora?

LoRA tooling is crowded. easylora focuses on the beginner-to-production gap:
fast first run, transparent decisions, reproducible artifacts, and a path to
alignment/evaluation without switching config systems.

| If you need... | Use... | easylora angle |
|---|---|---|
| Fastest custom kernels | Unsloth | Optional backend target; Autopilot stays backend-neutral |
| Large YAML pipelines | Axolotl | Simpler no-config start, still exports YAML |
| Broad UI workflow | LLaMA-Factory | CLI/Python-first, report-first workflow |
| Reference alignment algorithms | TRL | TRL-backed DPO under easylora configs |
| A safe first adapter quickly | easylora | One command + explainable Autopilot report |

## Install

```bash
pip install easylora
```

Optional extras:

```bash
pip install "easylora[qlora]"   # bitsandbytes for CUDA 4-bit/8-bit quantisation
pip install "easylora[align]"   # TRL for DPO preference tuning
pip install "easylora[wandb]"   # Weights & Biases logging
pip install "easylora[dev]"     # tests, linting, typing, docs
pip install "easylora[all]"     # everything
```

## Autopilot output

Autopilot generates a validated `TrainConfig`, runs the normal trainer, and
saves a reproducibility bundle:

```
output/
  adapter/                # PEFT adapter weights
  resolved_config.yaml    # full resolved config
  autopilot_report.json   # machine-readable hardware/model/dataset decisions
  autopilot_report.md     # shareable human-readable report
  train_config.json
  train_log.jsonl
  summary.json
  metadata.json
```

Preview the plan without training:

```bash
easylora autopilot plan \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --quality balanced \
  --save-report
```

Python API:

```python
from easylora import autopilot_plan, autopilot_train

plan = autopilot_plan(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
    quality="balanced",
)
print(plan.to_markdown())

artifacts = autopilot_train(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
)
```

## Config-driven training

You can still own every knob with YAML/JSON:

```yaml
model:
  base_model: "meta-llama/Llama-3.2-1B"
  torch_dtype: "auto"
data:
  dataset_name: "tatsu-lab/alpaca"
  format: "auto"
  max_seq_len: 2048
lora:
  r: 16
  alpha: 32
  target_modules: "auto"
training:
  epochs: 3
  batch_size: 4
  grad_accum: 4
output:
  output_dir: "./output"
repro:
  seed: 42
```

```bash
easylora train --config config.yaml
```

## Evaluation, merge, and alignment

```bash
# Evaluate and save a markdown report
easylora eval \
  --base-model meta-llama/Llama-3.2-1B \
  --adapter-dir ./output/adapter \
  --dataset eval.jsonl \
  --prompt "Explain LoRA in one sentence." \
  --compare-base \
  --output-report ./output/eval_report.md

# Merge adapter into a standalone model
easylora merge \
  --base-model meta-llama/Llama-3.2-1B \
  --adapter-dir ./output/adapter \
  --output-dir ./merged

# Preference tuning via TRL-backed DPO
easylora align dpo --config examples/recipes/dpo_preference.yaml
```

## Dataset formats

| Format | Columns | Description |
|---|---|---|
| `auto` | inferred | Chooses `chatml`, `alpaca`, or `raw` from columns |
| `raw` | `text` | Single text field for language modelling |
| `alpaca` | `instruction`, optional `input`, `output` | Instruction SFT with prompt masking |
| `chatml` | `messages` | Uses the tokenizer's chat template |

## Model support

easylora auto-detects LoRA target modules for LLaMA, Mistral/Mixtral, Qwen,
Gemma, Phi, Falcon, GPT-NeoX, Pythia, MPT, Bloom, OPT, GPT-2, StarCoder, and
other causal LM architectures. For unknown models, it scans `nn.Linear` modules
and selects attention-like targets.

```bash
easylora inspect-targets --model meta-llama/Llama-3.2-1B
```

## Learn more

- [Quickstart](https://alexsuw.github.io/easylora/quickstart/)
- [Autopilot guide](https://alexsuw.github.io/easylora/autopilot/)
- [Recipes](https://alexsuw.github.io/easylora/recipes/)
- [Benchmarks](https://alexsuw.github.io/easylora/benchmarks/)
- [Configuration reference](https://alexsuw.github.io/easylora/configuration/)

## Development

```bash
git clone https://github.com/alexsuw/easylora.git
cd easylora
pip install -e ".[dev]"
make test
make lint
make type
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup,
code style, and the PR process.

## License

[Apache-2.0](LICENSE)
