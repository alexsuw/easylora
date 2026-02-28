"""Dataset formatting for different instruction/chat templates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from datasets import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from easylora.config import DataConfig

logger = logging.getLogger(__name__)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)

ALPACA_INPUT_BLOCK = "### Input:\n{input}\n\n"


def format_examples(
    dataset: Dataset,
    data_config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """Tokenise a dataset according to the chosen format.

    Supported formats:
    - ``"raw"``:    uses ``data_config.text_field`` as pre-formatted text.
    - ``"alpaca"``: expects ``instruction``, optional ``input``, ``output`` columns.
    - ``"chatml"``: expects a ``messages`` column with role/content dicts.

    Returns a new ``Dataset`` with ``input_ids``, ``attention_mask``, and
    ``labels`` columns ready for causal-LM training.
    """
    fmt = data_config.format

    if fmt == "raw":
        fn = _build_raw_mapper(tokenizer, data_config)
    elif fmt == "alpaca":
        fn = _build_alpaca_mapper(tokenizer, data_config)
    elif fmt == "chatml":
        fn = _build_chatml_mapper(tokenizer, data_config)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    tokenized = dataset.map(
        fn,
        batched=False,
        remove_columns=dataset.column_names,
        desc=f"Formatting ({fmt})",
    )
    logger.info(
        "Formatted %d examples with format=%s, max_seq_len=%d",
        len(tokenized),
        fmt,
        data_config.max_seq_len,
    )
    return tokenized


def _build_raw_mapper(
    tokenizer: PreTrainedTokenizer,
    cfg: DataConfig,
):
    """Tokenise a single text field."""
    max_len = cfg.max_seq_len
    field = cfg.text_field

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        enc = tokenizer(
            example[field],
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    return _map


def _build_alpaca_mapper(
    tokenizer: PreTrainedTokenizer,
    cfg: DataConfig,
):
    """Format instruction/input/output into Alpaca template, masking the prompt in labels."""
    max_len = cfg.max_seq_len

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        inp = example.get("input", "")
        input_block = ALPACA_INPUT_BLOCK.format(input=inp) if inp else ""

        full_text = ALPACA_TEMPLATE.format(
            instruction=example["instruction"],
            input_block=input_block,
            output=example["output"],
        )

        prompt_text = ALPACA_TEMPLATE.format(
            instruction=example["instruction"],
            input_block=input_block,
            output="",
        )

        full_enc = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_len, padding=False)

        labels = full_enc["input_ids"].copy()
        prompt_len = len(prompt_enc["input_ids"])
        # Mask prompt tokens so loss is only on the response
        labels[:prompt_len] = [-100] * prompt_len

        full_enc["labels"] = labels
        return full_enc

    return _map


def _build_chatml_mapper(
    tokenizer: PreTrainedTokenizer,
    cfg: DataConfig,
):
    """Apply the tokenizer's chat template, masking everything except the final assistant turn."""
    max_len = cfg.max_seq_len

    def _map(example: dict[str, Any]) -> dict[str, Any]:
        messages = example["messages"]

        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Build a prefix up to the last assistant message for label masking
        prompt_messages = messages[:-1]
        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = ""

        full_enc = tokenizer(full_text, truncation=True, max_length=max_len, padding=False)
        labels = full_enc["input_ids"].copy()

        if prompt_text:
            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_len, padding=False)
            prompt_len = len(prompt_enc["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

        full_enc["labels"] = labels
        return full_enc

    return _map
