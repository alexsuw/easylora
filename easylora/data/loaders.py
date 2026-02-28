"""Dataset loading from local files and HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset

from easylora.exceptions import EasyLoRAConfigError

if TYPE_CHECKING:
    from easylora.config import DataConfig

logger = logging.getLogger(__name__)

_EXT_FORMAT: dict[str, str] = {
    ".json": "json",
    ".jsonl": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".arrow": "arrow",
}


_FORMAT_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "alpaca": ["instruction", "output"],
    "chatml": ["messages"],
}


def load_dataset_any(data_config: DataConfig) -> Dataset:
    """Load a dataset from a local file or the HuggingFace Hub.

    Validates that the dataset contains the expected columns for the chosen
    format and warns about empty or very short samples.

    Returns the requested split as a single ``Dataset``.
    """
    if data_config.dataset_path:
        ds = _load_local(data_config)
    elif data_config.dataset_name:
        ds = _load_hub(data_config)
    else:
        raise EasyLoRAConfigError("Provide either dataset_path or dataset_name.")

    _validate_columns(ds, data_config)
    _warn_short_samples(ds, data_config)
    return ds


def _load_local(cfg: DataConfig) -> Dataset:
    path = Path(cfg.dataset_path)  # type: ignore[arg-type]
    if not path.exists():
        raise EasyLoRAConfigError(f"Dataset file not found: {path}")

    ext = path.suffix.lower()
    fmt = _EXT_FORMAT.get(ext)
    if fmt is None:
        raise EasyLoRAConfigError(
            f"Unsupported file extension '{ext}'. Supported: {', '.join(_EXT_FORMAT.keys())}"
        )

    ds = load_dataset(fmt, data_files=str(path), split=cfg.split)
    logger.info("Loaded local dataset %s (%d rows)", path, len(ds))
    return ds  # type: ignore[return-value]


def _load_hub(cfg: DataConfig) -> Dataset:
    kwargs: dict = {"path": cfg.dataset_name, "split": cfg.split}
    if cfg.subset:
        kwargs["name"] = cfg.subset
    ds = load_dataset(**kwargs)
    logger.info("Loaded HF dataset %s/%s (%d rows)", cfg.dataset_name, cfg.split, len(ds))
    return ds  # type: ignore[return-value]


def _validate_columns(ds: Dataset, cfg: DataConfig) -> None:
    """Check that required columns exist for the chosen format."""
    required = _FORMAT_REQUIRED_COLUMNS.get(cfg.format)
    if required is None:
        if cfg.text_field not in ds.column_names:
            raise EasyLoRAConfigError(
                f"Dataset is missing text_field '{cfg.text_field}'. "
                f"Available columns: {ds.column_names}"
            )
        return

    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise EasyLoRAConfigError(
            f"Format '{cfg.format}' requires columns {required}, "
            f"but these are missing: {missing}. "
            f"Available columns: {ds.column_names}"
        )


def _warn_short_samples(ds: Dataset, cfg: DataConfig) -> None:
    """Log a warning if many samples appear empty or very short."""
    field = cfg.text_field if cfg.format == "raw" else None
    if field is None:
        return
    if field not in ds.column_names:
        return

    short = sum(1 for ex in ds if len(str(ex.get(field, ""))) < 10)
    if short > 0:
        pct = 100 * short / len(ds)
        logger.warning(
            "%d of %d samples (%.1f%%) have fewer than 10 characters in '%s'",
            short,
            len(ds),
            pct,
            field,
        )
