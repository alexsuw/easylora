"""EasyLoRATrainer — the high-level training facade."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from easylora.config import RunArtifacts, TrainConfig
from easylora.data.collator import CausalLMCollator
from easylora.data.formatting import format_examples
from easylora.data.loaders import load_dataset_any
from easylora.lora.adapter import apply_lora, save_adapter
from easylora.lora.merge import merge_adapter
from easylora.train.callbacks import JsonlLoggingCallback, TrainingSummaryCallback
from easylora.train.loop import build_trainer, build_training_args
from easylora.utils.hf import load_base_model, load_tokenizer
from easylora.utils.io import ensure_output_dir, save_json
from easylora.utils.model_card import generate_model_card
from easylora.utils.seed import set_seed

if TYPE_CHECKING:
    from datasets import Dataset
    from peft import PeftModel

logger = logging.getLogger(__name__)


def _get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _find_last_checkpoint(output_dir: Path) -> str | None:
    """Find the last HF Trainer checkpoint in *output_dir*."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    if checkpoints:
        return str(checkpoints[-1])
    return None


class EasyLoRATrainer:
    """High-level trainer that orchestrates the full LoRA fine-tuning pipeline.

    Usage::

        trainer = EasyLoRATrainer(config)
        artifacts = trainer.fit()
        trainer.merge_and_save("merged_model/")
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._model: PeftModel | None = None
        self._tokenizer = None
        self._train_dataset: Dataset | None = None
        self._eval_dataset: Dataset | None = None
        self._output_dir: Path | None = None

    def fit(self) -> RunArtifacts:
        """Run the full training pipeline: load, format, train, save.

        If a checkpoint from a previous run exists in the output directory and
        ``allow_overwrite`` is set, training resumes from that checkpoint.

        Returns:
            RunArtifacts describing all produced output paths.
        """
        from easylora.logging import setup_logger

        cfg = self.config
        set_seed(cfg.repro.seed, deterministic=cfg.repro.deterministic)

        self._output_dir = ensure_output_dir(
            cfg.output.output_dir, allow_overwrite=cfg.output.allow_overwrite
        )
        setup_logger(output_dir=self._output_dir, run_name=cfg.output.run_name)

        config_path = save_json(cfg.model_dump(), self._output_dir / "train_config.json")
        log_path = self._output_dir / "train_log.jsonl"
        summary_path = self._output_dir / "summary.json"
        adapter_dir = self._output_dir / "adapter"

        logger.info("Starting easylora training run")

        logger.info("Loading tokenizer...")
        self._tokenizer = load_tokenizer(cfg.model)

        logger.info("Loading dataset...")
        raw_dataset = load_dataset_any(cfg.data)

        logger.info("Formatting dataset...")
        formatted = format_examples(raw_dataset, cfg.data, self._tokenizer)

        if cfg.data.val_split_ratio > 0:
            split = formatted.train_test_split(
                test_size=cfg.data.val_split_ratio, seed=cfg.repro.seed
            )
            self._train_dataset = split["train"]
            self._eval_dataset = split["test"]
            train_ds = self._train_dataset
            eval_ds = self._eval_dataset
            logger.info(
                "Split: %d train, %d eval",
                len(train_ds),
                len(eval_ds),
            )
        else:
            self._train_dataset = formatted
            self._eval_dataset = None

        logger.info("Loading base model...")
        base_model = load_base_model(cfg.model)

        logger.info("Applying LoRA...")
        self._model = apply_lora(base_model, cfg.lora, cfg.model)
        assert self._model is not None
        assert self._tokenizer is not None

        if cfg.training.gradient_checkpointing:
            self._model.enable_input_require_grads()

        collator = CausalLMCollator(tokenizer=self._tokenizer, max_seq_len=cfg.data.max_seq_len)

        training_args = build_training_args(cfg)
        callbacks = [
            JsonlLoggingCallback(log_path),
            TrainingSummaryCallback(summary_path),
        ]

        trainer = build_trainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            training_args=training_args,
            collator=collator,
            callbacks=callbacks,
        )

        resume_from = _find_last_checkpoint(self._output_dir)
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)

        logger.info("Training...")
        trainer.train(resume_from_checkpoint=resume_from)

        logger.info("Saving adapter...")
        save_adapter(self._model, adapter_dir)

        self._save_metadata(self._output_dir)

        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_data = None
        generate_model_card(
            adapter_dir,
            cfg.model.base_model,
            lora_config=cfg.lora.model_dump(),
            summary=summary_data,
        )

        artifacts = RunArtifacts(
            adapter_dir=str(adapter_dir),
            config_path=str(config_path),
            log_path=str(log_path),
            summary_path=str(summary_path),
        )
        logger.info("Training complete. Artifacts: %s", artifacts.model_dump())
        return artifacts

    def _save_metadata(self, output_dir: Path) -> None:
        """Save run metadata (versions, timestamps, git commit)."""
        from importlib.metadata import version as pkg_version

        def _safe_version(pkg: str) -> str:
            try:
                return pkg_version(pkg)
            except Exception:
                return "unknown"

        metadata = {
            "base_model": self.config.model.base_model,
            "dataset": self.config.data.dataset_path or self.config.data.dataset_name,
            "easylora_version": _safe_version("easylora"),
            "torch_version": torch.__version__,
            "transformers_version": _safe_version("transformers"),
            "peft_version": _safe_version("peft"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": _get_git_commit(),
        }
        save_json(metadata, output_dir / "metadata.json")

    def evaluate(self) -> dict:
        """Run perplexity evaluation on the eval split.

        Requires that ``.fit()`` has been called first, or that a model and
        eval dataset are available.
        """
        from easylora.eval.perplexity import compute_perplexity

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call .fit() before .evaluate(), or load a model first.")

        dataset = self._eval_dataset or self._train_dataset
        if dataset is None:
            raise RuntimeError("No dataset available for evaluation.")

        ppl = compute_perplexity(
            self._model, self._tokenizer, dataset, self.config.data.max_seq_len
        )
        return {"perplexity": ppl}

    def save_adapter(self, output_dir: str | Path | None = None) -> Path:
        """Save the current LoRA adapter to disk."""
        if self._model is None:
            raise RuntimeError("No trained model. Call .fit() first.")
        out = Path(output_dir) if output_dir else self._output_dir / "adapter"  # type: ignore[operator]
        return save_adapter(self._model, out)

    def merge_and_save(self, output_dir: str | Path) -> Path:
        """Merge adapter into the base model and save the full merged weights."""
        if self._output_dir is None:
            raise RuntimeError("No training run found. Call .fit() first.")
        adapter_dir = self._output_dir / "adapter"
        return merge_adapter(
            self.config.model.base_model,
            adapter_dir,
            output_dir,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=self.config.model.torch_dtype,
        )
