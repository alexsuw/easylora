"""File I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from easylora.exceptions import EasyLoRAIOError


def ensure_output_dir(path: str | Path, *, allow_overwrite: bool = False) -> Path:
    """Create output directory, raising if it already contains artifacts.

    Args:
        path: Directory path to create.
        allow_overwrite: If False and the directory is non-empty, raise an error.

    Returns:
        Resolved Path object.
    """
    out = Path(path).resolve()
    if out.exists() and any(out.iterdir()) and not allow_overwrite:
        raise EasyLoRAIOError(
            f"Output directory '{out}' already contains files. "
            "Set allow_overwrite=True or use --force to overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Any, path: str | Path) -> Path:
    """Serialise *data* as indented JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    """Read a JSON file and return parsed data."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(data: Any, path: str | Path) -> Path:
    """Serialise *data* as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return p


def load_yaml(path: str | Path) -> Any:
    """Read a YAML file and return parsed data."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
