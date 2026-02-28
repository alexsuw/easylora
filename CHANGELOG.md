# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-02-28

### Added

- PyPI publishing via GitHub Actions with OIDC trusted publishing (no API tokens).
- TestPyPI workflow option for pre-release verification.
- `[qlora]` optional extra for bitsandbytes.
- `[docs]` optional extra for mkdocs-material.
- `License :: OSI Approved :: Apache Software License` classifier.
- Comprehensive release documentation in `docs/release.md`.

### Changed

- Split `[dev]` extras: docs dependencies moved to `[docs]` (included via `[dev]`).
- Pinned twine to `<6.0` for metadata compatibility.

## [0.1.0] - 2026-02-28

### Added

- LoRA and QLoRA (4-bit/8-bit) fine-tuning for causal language models via
  HuggingFace Transformers and PEFT.
- Typer-based CLI with `train`, `eval`, `merge`, `doctor`, `inspect-targets`,
  and `init-config` subcommands.
- Pydantic v2 configuration with YAML and JSON file loading.
- Auto target module detection for 16+ model architectures (LLaMA, Mistral,
  Qwen, Falcon, GPT-NeoX, Phi, Gemma, and more).
- Data-driven target registry (`targets_registry.yaml`).
- Dataset formatting for alpaca, chatml, and raw text formats.
- Adapter save, load, and merge utilities.
- Perplexity evaluation and generation-based sanity checks.
- JSONL training logs, run summaries, and metadata output.
- Run resumption from HF Trainer checkpoints.
- Model card generation in adapter output directories.
- Dataset validation with column checks and short-sample warnings.
- `--dry-run` and `--print-config` flags for config validation without training.
- Rich console output with progress and run summary panels.
- Causal LM data collator with proper label masking and padding.
- Reproducibility via seed control and deterministic mode.
- Optional Weights & Biases integration.
- MkDocs documentation site with quickstart, configuration, CLI, and
  troubleshooting guides.
- Colab quickstart notebook.
- CI/CD with GitHub Actions (lint, typecheck, test, build, docs deploy).
- OSS hygiene: LICENSE, CONTRIBUTING, SECURITY, GOVERNANCE, SUPPORT, CITATION.
