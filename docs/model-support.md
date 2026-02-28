# Model Support

## Auto Target Module Detection

When `target_modules` is set to `"auto"` (the default), easylora automatically
selects the appropriate LoRA target modules based on the model architecture.

### Supported Architectures

| Architecture | Model Type | Target Modules |
|---|---|---|
| LLaMA, LLaMA 2/3, Code Llama, Vicuna | `llama` | `q_proj`, `v_proj` |
| Mistral, Mixtral | `mistral`, `mixtral` | `q_proj`, `v_proj` |
| Qwen2 | `qwen2` | `q_proj`, `v_proj` |
| Qwen (v1) | `qwen` | `c_attn` |
| Gemma, Gemma 2 | `gemma`, `gemma2` | `q_proj`, `v_proj` |
| Phi-2 | `phi` | `q_proj`, `v_proj` |
| Phi-3 | `phi3` | `qkv_proj` |
| OPT | `opt` | `q_proj`, `v_proj` |
| GPT-NeoX, Pythia | `gpt_neox` | `query_key_value` |
| Falcon | `falcon` | `query_key_value` |
| Bloom | `bloom` | `query_key_value` |
| MPT | `mpt` | `Wqkv` |
| GPT-2 | `gpt2` | `c_attn` |
| StarCoder, GPT-BigCode | `gpt_bigcode` | `c_attn` |

### How Detection Works

1. **Registry lookup**: checks the model's architecture class name against a
   known mapping in `easylora/lora/targets_registry.yaml`.
2. **Model type lookup**: checks `model.config.model_type` against a secondary
   mapping for broader compatibility.
3. **Linear scan fallback**: if neither lookup matches, scans the model for
   `nn.Linear` layers and selects attention-like module names (preferring
   patterns like `q_proj`, `k_proj`, `v_proj`, `query`, `key`, `value`).
4. **Last resort**: falls back to `["q_proj", "v_proj"]` with a warning.

### Inspecting Targets

Use the CLI to see what targets would be selected for a model:

```bash
easylora inspect-targets --model meta-llama/Llama-3.2-1B
```

### Overriding Targets

To use specific modules instead of auto-detection:

=== "YAML"

    ```yaml
    lora:
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    ```

=== "Python"

    ```python
    from easylora.config import LoRAConfig

    lora = LoRAConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    ```

### Adding New Architectures

To add support for a new model architecture:

1. Add entries to `easylora/lora/targets_registry.yaml`.
2. Add tests in `tests/test_targets.py`.
3. Submit a pull request.

## QLoRA Support

QLoRA requires the `bitsandbytes` library:

```bash
pip install bitsandbytes
```

Enable via config:

```yaml
model:
  base_model: "meta-llama/Llama-3.2-1B"
  load_in_4bit: true
```

!!! note
    bitsandbytes currently requires a CUDA GPU. It does not work on CPU-only
    machines or Apple Silicon.
