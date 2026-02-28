# Contributing to easylora

Thank you for your interest in contributing to easylora. This guide covers
the development workflow, code style expectations, and the pull request process.

## Getting Started

```bash
git clone https://github.com/alexsuw/easylora.git
cd easylora
pip install -e ".[dev]"
```

This installs all development dependencies including pytest, ruff, pyright, and
build tools.

## Running Tests

```bash
# Fast tests only (config, formatting, adapter I/O)
make test

# Include slow tests (smoke training with tiny model)
make test-slow

# Run a specific test file
pytest tests/test_config.py -v
```

## Code Style

- **Linting and formatting**: [ruff](https://docs.astral.sh/ruff/) handles both.
  Run `make lint` to check and `make format` to auto-fix.
- **Type checking**: [pyright](https://github.com/microsoft/pyright) in basic mode.
  Run `make type` to check.
- **Type hints**: required on all function signatures (public and private).
- **Docstrings**: required on all public functions, classes, and modules.
  Use Google-style docstrings.
- **Line length**: 100 characters.

```bash
make lint      # ruff check + ruff format --check
make format    # ruff format + ruff check --fix
make type      # pyright
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `chore:` maintenance, deps, CI config
- `ci:` CI workflow changes
- `test:` test additions or fixes
- `refactor:` code change that neither fixes a bug nor adds a feature

Examples:

```
feat: add easylora doctor command for environment diagnostics
fix: handle missing pad_token in tokenizer loading
docs: add model compatibility section to README
```

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes with appropriate tests.
3. Ensure all checks pass locally: `make lint && make type && make test`.
4. Open a pull request against `main`.
5. CI must be green. At least one maintainer approval is required.
6. Squash-merge is preferred for clean history.

## Adding a New Model Architecture

To add target module mappings for a new model architecture:

1. Add entries to `easylora/lora/targets_registry.yaml`.
2. Add a test case in `tests/test_targets.py`.
3. Update `docs/model-support.md`.

## Release Checklist

For maintainers preparing a release:

1. Update `CHANGELOG.md` with the new version and date.
2. Bump `version` in `pyproject.toml`.
3. Commit: `chore: release v0.x.0`.
4. Create and push a git tag: `git tag v0.x.0 && git push origin v0.x.0`.
5. The release workflow will build and publish the package.
