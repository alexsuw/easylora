.PHONY: install-dev test test-slow lint format type build docs-serve clean

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest -q -m "not slow"

test-slow:
	pytest -q

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

type:
	pyright

build:
	python -m build
	twine check dist/*

docs-serve:
	mkdocs serve

clean:
	rm -rf dist/ build/ site/ *.egg-info .ruff_cache .pytest_cache .pyright
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
