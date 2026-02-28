"""Perplexity computation for causal language models."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    max_seq_len: int = 2048,
    *,
    stride: int | None = None,
    max_samples: int | None = None,
) -> float:
    """Compute perplexity of *model* on *dataset* using a sliding window.

    Args:
        model: A causal language model (with or without LoRA adapters).
        tokenizer: Corresponding tokenizer.
        dataset: Tokenised dataset with ``input_ids`` column.
        max_seq_len: Maximum context window.
        stride: Sliding window stride; defaults to ``max_seq_len // 2``.
        max_samples: Limit number of examples evaluated.

    Returns:
        Perplexity (float).
    """
    if stride is None:
        stride = max_seq_len // 2

    model.eval()
    device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    samples = dataset.select(range(min(max_samples, len(dataset)))) if max_samples else dataset

    for example in samples:
        input_ids = example["input_ids"]
        seq_len = len(input_ids)

        for begin in range(0, seq_len, stride):
            end = min(begin + max_seq_len, seq_len)
            ids = torch.tensor(input_ids[begin:end], dtype=torch.long, device=device).unsqueeze(0)

            target_len = end - begin
            if begin > 0:
                # Overlap region: only score tokens beyond the stride
                target_len = min(end - begin, stride)

            labels = ids.clone()
            # Mask tokens that were already scored in the previous window
            if begin > 0:
                labels[:, : ids.shape[1] - target_len] = -100

            outputs = model(input_ids=ids, labels=labels)
            nll = outputs.loss.item() * target_len
            total_nll += nll
            total_tokens += target_len

            if end == seq_len:
                break

    if total_tokens == 0:
        logger.warning("No tokens evaluated — returning inf perplexity.")
        return float("inf")

    ppl = math.exp(total_nll / total_tokens)
    logger.info("Perplexity: %.4f (%d tokens)", ppl, total_tokens)
    return ppl
