"""Logging infrastructure: console + JSONL file logger, optional W&B hook."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logger(
    output_dir: str | Path | None = None,
    run_name: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the ``easylora`` root logger.

    Outputs to:
    - Console via Rich (always)
    - ``{output_dir}/logs.jsonl`` (if *output_dir* is provided)

    Returns the root ``easylora`` logger.
    """
    global _CONFIGURED
    logger = logging.getLogger("easylora")

    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console = RichHandler(rich_tracebacks=True, show_path=False)
    console.setLevel(level)
    logger.addHandler(console)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "logs.jsonl"
        jsonl = JsonlHandler(log_path)
        jsonl.setLevel(level)
        logger.addHandler(jsonl)

    _CONFIGURED = True
    return logger


class JsonlHandler(logging.Handler):
    """Logging handler that appends JSON lines to a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": time.time(),
            "level": record.levelname,
            "name": record.name,
            "msg": self.format(record),
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        super().close()


def init_wandb(config: dict, project: str = "easylora", run_name: str | None = None) -> None:
    """Initialise a Weights & Biases run (optional — only if wandb is installed)."""
    try:
        import wandb
    except ImportError:
        logging.getLogger(__name__).warning(
            "wandb is not installed; skipping W&B integration. "
            "Install with: pip install easylora[wandb]"
        )
        return

    wandb.init(project=project, name=run_name, config=config)
