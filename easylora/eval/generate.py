"""Text generation helper for sanity-checking fine-tuned models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

# TODO: Support vision-language models and diffusers (out of scope for MVP).


def generate_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    **gen_kwargs: Any,
) -> list[str]:
    """Generate text completions for a list of prompts.

    Intended for quick sanity checks — not optimised for throughput.

    Args:
        model: A causal language model (with or without adapters).
        tokenizer: Corresponding tokenizer.
        prompts: List of input strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        do_sample: Whether to sample (vs. greedy).
        **gen_kwargs: Extra keyword arguments forwarded to ``model.generate()``.

    Returns:
        List of generated text strings (prompt excluded).
    """
    model.eval()
    device = next(model.parameters()).device
    results: list[str] = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)

    logger.info("Generated %d samples", len(results))
    return results
