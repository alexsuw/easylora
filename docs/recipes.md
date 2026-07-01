# Recipes

Recipes are copy-paste starting points. Prefer Autopilot when you do not know
the best batch size, sequence length, or QLoRA setting for your machine.

## Autopilot from a Hugging Face dataset

```bash
easylora train --autopilot \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --quality balanced
```

## Local ChatML JSONL

Dataset row:

```json
{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"}]}
```

Run:

```bash
easylora train --config examples/recipes/qwen_chat_autopilot.yaml
```

## QLoRA on a 7B-class model

```bash
pip install "easylora[qlora]"
easylora train --config examples/recipes/llama3_qlora_alpaca.yaml
```

## Tiny CPU smoke test

Use this to verify install and config validation before moving to a GPU:

```bash
easylora train --config examples/recipes/tiny_cpu_smoke.yaml --dry-run
```

## DPO preference tuning

Preference row:

```json
{"prompt":"Explain LoRA","chosen":"LoRA trains small adapter matrices.","rejected":"LoRA is unrelated to fine-tuning."}
```

Run:

```bash
pip install "easylora[align]"
easylora align dpo --config examples/recipes/dpo_preference.yaml
```

## Publish to Hugging Face Hub

Set `output.push_to_hub: true` and `output.hub_repo_id` in
`examples/recipes/push_to_hub.yaml`, then train normally.
