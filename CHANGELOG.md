# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

### Added

- LoRA and QLoRA (4-bit/8-bit) fine-tuning for causal language models via
  HuggingFace Transformers and PEFT.
- Typer-based CLI with `train`, `eval`, and `merge` subcommands.
- Pydantic v2 configuration with YAML and JSON file loading.
- Auto target module detection for 16+ model architectures (LLaMA, Mistral,
  Qwen, Falcon, GPT-NeoX, Phi, Gemma, and more).
- Dataset formatting for alpaca, chatml, and raw text formats.
- Adapter save, load, and merge utilities.
- Perplexity evaluation and generation-based sanity checks.
- JSONL training logs and run summary output.
- Causal LM data collator with proper label masking and padding.
- Reproducibility via seed control and deterministic mode.
- Rich console output and optional Weights & Biases integration.
