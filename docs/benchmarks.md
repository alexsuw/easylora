# Benchmarks

Benchmarks serve two purposes:

1. Validate Autopilot memory and speed estimates.
2. Give users reproducible commands instead of vague performance claims.

## Run a local benchmark

```bash
python benchmarks/benchmark_autopilot.py \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --quality fast \
    --output benchmarks/results/llama32_1b_fast.json
```

The script plans a run, records hardware/model/dataset metadata, and can execute
a short training smoke when `--run-training` is provided.

## Report fields

| Field | Meaning |
|---|---|
| `estimated_vram_gb` | Autopilot estimate before training |
| `actual_peak_vram_gb` | CUDA max memory allocated during optional training |
| `estimated_steps_per_sec` | Autopilot throughput estimate |
| `actual_steps_per_sec` | Measured throughput from `summary.json`, when available |
| `quality` | `fast`, `balanced`, or `high` |

## Current baseline table

These rows are placeholders for reproducible community runs. Keep results tied
to exact commands, GPU names, package versions, and model IDs.

| Model | GPU | Quality | Est. VRAM | Actual VRAM | Est. steps/sec | Actual steps/sec |
|---|---|---|---|---|---|---|
| `meta-llama/Llama-3.2-1B` | TBD | fast | TBD | TBD | TBD | TBD |
| `Qwen/Qwen2.5-3B-Instruct` | TBD | balanced | TBD | TBD | TBD | TBD |
| `mistralai/Mistral-7B-v0.1` | TBD | fast | TBD | TBD | TBD | TBD |

## Publishing results

When adding a benchmark result:

- Commit the JSON output under `benchmarks/results/`.
- Add the command used to reproduce it.
- Include CUDA, PyTorch, Transformers, PEFT, and easylora versions.
- Prefer small, comparable datasets for smoke benchmarks and real datasets for
  published throughput claims.
