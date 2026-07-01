"""Minimal Direct Preference Optimization support."""

from __future__ import annotations

from typing import Any

from easylora.config import DPOTrainConfig, RunArtifacts
from easylora.data.loaders import load_dataset_any
from easylora.exceptions import EasyLoRADependencyError
from easylora.lora.adapter import apply_lora, save_adapter
from easylora.utils.hf import load_base_model, load_tokenizer
from easylora.utils.io import ensure_output_dir, save_json
from easylora.utils.seed import set_seed


def train_dpo(config: DPOTrainConfig) -> RunArtifacts:
    """Run a TRL-backed DPO training job with easylora configs and artifacts."""
    try:
        from trl import DPOTrainer
        from trl import DPOConfig as TRLDPOConfig
    except ImportError as exc:
        raise EasyLoRADependencyError(package="trl", feature="DPO alignment") from exc

    set_seed(config.repro.seed, deterministic=config.repro.deterministic)
    output_dir = ensure_output_dir(config.output.output_dir, config.output.allow_overwrite)
    adapter_dir = output_dir / "adapter"
    config_path = save_json(config.model_dump(), output_dir / "dpo_config.json")
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "train_log.jsonl"

    tokenizer = load_tokenizer(config.model)
    model = apply_lora(load_base_model(config.model), config.lora, config.model)
    ref_model = None
    if config.dpo.reference_model:
        ref_model = load_base_model(
            config.model.model_copy(update={"base_model": config.dpo.reference_model})
        )

    dataset_cfg = _as_data_config(config)
    raw_dataset = load_dataset_any(dataset_cfg)
    preference_dataset = raw_dataset.map(
        _build_preference_mapper(config),
        remove_columns=raw_dataset.column_names,
        desc="Formatting DPO preferences",
    )
    train_dataset: Any = preference_dataset
    eval_dataset: Any | None = None
    if config.data.val_split_ratio > 0:
        split = preference_dataset.train_test_split(
            test_size=config.data.val_split_ratio,
            seed=config.repro.seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    args = TRLDPOConfig(
        output_dir=config.output.output_dir,
        run_name=config.output.run_name,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.grad_accum,
        learning_rate=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        warmup_ratio=config.optim.warmup_ratio,
        lr_scheduler_type=config.optim.scheduler,
        max_steps=config.dpo.max_steps if config.dpo.max_steps > 0 else config.training.max_steps,
        logging_steps=config.training.logging_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.output.save_total_limit,
        beta=config.dpo.beta,
        max_prompt_length=config.data.max_prompt_len,
        max_length=config.data.max_seq_len,
        report_to="none",
        remove_unused_columns=False,
        seed=config.repro.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    save_adapter(model, adapter_dir)
    save_json({"status": "complete", "algorithm": "dpo"}, summary_path)
    log_path.touch(exist_ok=True)
    return RunArtifacts(
        adapter_dir=str(adapter_dir),
        config_path=str(config_path),
        log_path=str(log_path),
        summary_path=str(summary_path),
    )


def _as_data_config(config: DPOTrainConfig):
    from easylora.config import DataConfig

    return DataConfig(
        dataset_path=config.data.dataset_path,
        dataset_name=config.data.dataset_name,
        subset=config.data.subset,
        split=config.data.split,
        format="raw",
        text_field=config.data.prompt_field,
        max_seq_len=config.data.max_seq_len,
        val_split_ratio=0.0,
    )


def _build_preference_mapper(config: DPOTrainConfig):
    def _map(example: dict[str, Any]) -> dict[str, str]:
        return {
            "prompt": str(example[config.data.prompt_field]),
            "chosen": str(example[config.data.chosen_field]),
            "rejected": str(example[config.data.rejected_field]),
        }

    return _map
