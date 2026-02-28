"""Data collator for causal language modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer


@dataclass
class CausalLMCollator:
    """Pads a batch of tokenised examples for causal LM training.

    Handles ``input_ids``, ``attention_mask``, and ``labels``.
    Labels are padded with -100 so the loss ignores padding tokens.
    """

    tokenizer: PreTrainedTokenizer
    max_seq_len: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = self.max_seq_len or max(len(f["input_ids"]) for f in features)

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for f in features:
            ids = f["input_ids"][:max_len]
            mask = f.get("attention_mask", [1] * len(ids))[:max_len]
            labs = f.get("labels", ids.copy())[:max_len]

            pad_len = max_len - len(ids)
            input_ids_batch.append(ids + [pad_id] * pad_len)
            attention_mask_batch.append(mask + [0] * pad_len)
            labels_batch.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }
