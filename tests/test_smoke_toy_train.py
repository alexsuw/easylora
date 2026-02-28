"""Smoke test: tiny training run to ensure the full pipeline works end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easylora.config import DataConfig, ModelConfig, OutputConfig, TrainConfig, TrainLoopConfig
from easylora.train.trainer import EasyLoRATrainer

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def tiny_dataset(tmp_path) -> Path:
    """Create a minimal JSONL training file."""
    data_path = tmp_path / "train.jsonl"
    examples = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "A simple test sentence for training."},
        {"text": "LoRA adapters are parameter efficient."},
        {"text": "Fine-tuning language models is fun."},
    ]
    data_path.write_text(
        "\n".join(json.dumps(ex) for ex in examples),
        encoding="utf-8",
    )
    return data_path


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_smoke_training(tmp_path, tiny_dataset):
    """Run 2 training steps on a tiny model and verify outputs."""
    output_dir = tmp_path / "output"

    config = TrainConfig(
        model=ModelConfig(base_model=TINY_MODEL, device_map=None, torch_dtype="fp32"),
        data=DataConfig(
            dataset_path=str(tiny_dataset),
            format="raw",
            max_seq_len=64,
        ),
        training=TrainLoopConfig(
            epochs=1,
            batch_size=2,
            grad_accum=1,
            max_steps=2,
            logging_steps=1,
            gradient_checkpointing=False,
        ),
        output=OutputConfig(output_dir=str(output_dir)),
    )

    trainer = EasyLoRATrainer(config)
    artifacts = trainer.fit()

    assert Path(artifacts.adapter_dir).exists()
    assert Path(artifacts.config_path).exists()
    assert Path(artifacts.summary_path).exists()
    assert Path(artifacts.log_path).exists()

    summary = json.loads(Path(artifacts.summary_path).read_text())
    assert "total_steps" in summary
    assert summary["total_steps"] >= 2
    assert "final_loss" in summary

    adapter_config = Path(artifacts.adapter_dir) / "adapter_config.json"
    assert adapter_config.exists()
