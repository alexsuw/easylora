# Troubleshooting

## Environment Diagnostics

Run `easylora doctor` to print your environment info. Include this output in
bug reports.

```bash
easylora doctor
```

## Common Issues

### "bitsandbytes is not installed"

QLoRA (4-bit / 8-bit) requires the `bitsandbytes` package:

```bash
pip install bitsandbytes
```

bitsandbytes requires a CUDA GPU. It does not work on CPU-only machines or
Apple Silicon.

### CUDA out of memory

- Reduce `batch_size` (try 1 or 2)
- Increase `grad_accum` to maintain effective batch size
- Enable `gradient_checkpointing: true`
- Use QLoRA (`load_in_4bit: true`)
- Reduce `max_seq_len`

### "pad_token was None"

This is handled automatically. easylora sets `pad_token = eos_token` when the
tokenizer does not define one. You will see an info log confirming this.

### Output directory already exists

By default, easylora refuses to overwrite an existing output directory. Use
`allow_overwrite: true` in config or `--force` on the CLI.

### Training is slow

- Ensure you are using a GPU (check with `easylora doctor`)
- bf16 is used by default when available; check logs for confirmation
- `gradient_checkpointing` trades compute for memory -- disable if you have
  enough VRAM
- Check that `batch_size * grad_accum` gives a reasonable effective batch size

### "Unknown architecture" warning

If easylora cannot detect LoRA target modules for your model, it falls back to
scanning `nn.Linear` layers. You can:

- Use `easylora inspect-targets --model <id>` to see what would be selected
- Override with an explicit list in your config:

```yaml
lora:
  target_modules: ["q_proj", "v_proj"]
```

### Import errors after install

Ensure you have installed all dependencies:

```bash
pip install easylora
```

For development:

```bash
pip install -e ".[dev]"
```

### Tests fail with download errors

Some tests download a tiny model from HuggingFace Hub. If you are behind a
firewall or offline, these tests will fail. Mark them with `-m "not slow"` to
skip:

```bash
pytest -q -m "not slow"
```
