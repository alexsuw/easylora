"""Dataset sampling and token-length analysis for autopilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from easylora.config import DataConfig
from easylora.data.formatting import ALPACA_INPUT_BLOCK, ALPACA_TEMPLATE
from easylora.data.loaders import load_dataset_any


@dataclass
class DatasetProfile:
    """Summarized dataset properties used by autopilot heuristics."""

    examples: int
    sampled_examples: int
    inferred_format: str
    text_field: str
    p50_tokens: int
    p95_tokens: int
    max_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def infer_dataset_format(column_names: list[str]) -> tuple[str, str]:
    """Infer a training format and text field from dataset columns."""
    cols = set(column_names)
    if "messages" in cols:
        return "chatml", "messages"
    if "instruction" in cols and "output" in cols:
        return "alpaca", "instruction"
    if "text" in cols:
        return "raw", "text"

    # Best effort for arbitrary raw text datasets.
    for c in column_names:
        if c not in {"id", "idx"}:
            return "raw", c
    return "raw", "text"


def _build_text(
    example: dict[str, Any],
    inferred_format: str,
    text_field: str,
    tokenizer: Any,
) -> str:
    if inferred_format == "raw":
        return str(example.get(text_field, ""))

    if inferred_format == "alpaca":
        inp = example.get("input", "")
        input_block = ALPACA_INPUT_BLOCK.format(input=inp) if inp else ""
        return ALPACA_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input_block=input_block,
            output=example.get("output", ""),
        )

    messages = example.get("messages", [])
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )  # type: ignore[no-any-return]
    except Exception:
        # Fallback if tokenizer does not define a chat template.
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
            if isinstance(m, dict)
        )


def _percentile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    idx = round((len(sorted_values) - 1) * p)
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def analyze_dataset(
    dataset: str,
    *,
    tokenizer: Any,
    split: str = "train",
    subset: str | None = None,
    max_samples: int = 512,
) -> DatasetProfile:
    """Load and sample dataset; infer format and token-length distribution."""
    source_path = Path(dataset)
    cfg = DataConfig(
        dataset_path=dataset if source_path.exists() else None,
        dataset_name=dataset if not source_path.exists() else None,
        split=split,
        subset=subset,
        format="raw",
        text_field="text",
        max_seq_len=2048,
        val_split_ratio=0.0,
    )
    ds = load_dataset_any(cfg)
    inferred_format, text_field = infer_dataset_format(ds.column_names)
    sample = _sample_dataset(ds, max_samples=max_samples)
    lengths = _token_lengths(sample, inferred_format, text_field, tokenizer)
    lengths_sorted = sorted(lengths)
    return DatasetProfile(
        examples=len(ds),
        sampled_examples=len(sample),
        inferred_format=inferred_format,
        text_field=text_field,
        p50_tokens=_percentile(lengths_sorted, 0.50),
        p95_tokens=_percentile(lengths_sorted, 0.95),
        max_tokens=max(lengths_sorted) if lengths_sorted else 0,
    )


def _sample_dataset(ds: Dataset, *, max_samples: int) -> Dataset:
    if len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=42).select(range(max_samples))


def _token_lengths(
    ds: Dataset,
    inferred_format: str,
    text_field: str,
    tokenizer: Any,
) -> list[int]:
    lengths: list[int] = []
    for ex in ds:
        if not isinstance(ex, dict):
            continue
        text = _build_text(ex, inferred_format, text_field, tokenizer)
        ids = tokenizer(text, truncation=False, padding=False).get("input_ids", [])
        lengths.append(len(ids))
    return lengths
