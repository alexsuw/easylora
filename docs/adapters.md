# Adapters

## What is a LoRA Adapter?

A LoRA adapter is a small set of delta weights (typically 10--50 MB) that can be
loaded on top of a frozen base model. This makes adapters lightweight, portable,
and easy to version.

## Saving Adapters

After training, the adapter is automatically saved to `{output_dir}/adapter/`.
You can also save manually:

```python
from easylora import save_adapter

save_adapter(model, "./my_adapter", metadata={"task": "summarisation"})
```

The adapter directory contains:

- `adapter_model.safetensors` -- adapter weights
- `adapter_config.json` -- PEFT configuration
- `easylora_metadata.json` -- optional metadata you provide

## Loading Adapters

```python
from easylora import load_adapter

model = load_adapter(
    base_model_name_or_path="meta-llama/Llama-3.2-1B",
    adapter_dir="./my_adapter",
)
```

## Merging Adapters

Merging bakes the adapter weights into the base model, producing a standalone
model that does not require PEFT at inference time:

=== "Python"

    ```python
    from easylora import merge_adapter

    merge_adapter(
        base_model_name_or_path="meta-llama/Llama-3.2-1B",
        adapter_dir="./output/adapter",
        output_dir="./merged_model",
    )
    ```

=== "CLI"

    ```bash
    easylora merge \
        --base-model meta-llama/Llama-3.2-1B \
        --adapter-dir ./output/adapter \
        --output-dir ./merged_model
    ```

The merged model can be loaded with standard HuggingFace APIs:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

## Output Artifacts

After a training run, the output directory contains:

```
output/
  adapter/              # LoRA adapter weights
  train_config.json     # Full config used
  train_log.jsonl       # Step-by-step training metrics
  summary.json          # Final loss, steps, runtime
  metadata.json         # Base model, versions, timestamp
  logs.jsonl            # Application logs
```

## Publishing to HuggingFace Hub

Set `push_to_hub: true` and `hub_repo_id` in your config:

```yaml
output:
  push_to_hub: true
  hub_repo_id: "your-username/my-lora-adapter"
  hub_private: true
```

Ensure you are logged in via `huggingface-cli login` before training.
