"""Build HuggingFace TrainingArguments and Trainer from easylora config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import Trainer, TrainingArguments

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel
    from transformers import PreTrainedTokenizer, TrainerCallback

    from easylora.config import TrainConfig
    from easylora.data.collator import CausalLMCollator

logger = logging.getLogger(__name__)


def _resolve_bf16() -> bool:
    """Return True if bf16 is available on the current device."""
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return False
    return False


def build_training_args(config: TrainConfig) -> TrainingArguments:
    """Map easylora TrainConfig fields to HuggingFace TrainingArguments."""
    use_bf16 = _resolve_bf16()

    args = TrainingArguments(
        output_dir=config.output.output_dir,
        run_name=config.output.run_name,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.grad_accum,
        learning_rate=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        adam_beta1=config.optim.betas[0],
        adam_beta2=config.optim.betas[1],
        warmup_ratio=config.optim.warmup_ratio,
        lr_scheduler_type=config.optim.scheduler,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.output.save_total_limit,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy="steps" if config.training.eval_steps else "no",
        save_strategy="steps" if config.training.save_steps else "epoch",
        seed=config.repro.seed,
        data_seed=config.repro.seed,
        report_to="none",
        push_to_hub=config.output.push_to_hub,
        hub_model_id=config.output.hub_repo_id,
        hub_private_repo=config.output.hub_private,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    return args


def build_trainer(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    training_args: TrainingArguments,
    collator: CausalLMCollator,
    callbacks: list[TrainerCallback] | None = None,
) -> Trainer:
    """Construct a HuggingFace Trainer wired up for LoRA training."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
