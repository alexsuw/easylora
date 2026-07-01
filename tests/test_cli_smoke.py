"""Smoke tests for CLI dry-run/report paths."""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from easylora.autopilot.dataset_analysis import DatasetProfile
from easylora.autopilot.hardware import HardwareProfile
from easylora.autopilot.heuristics import PlanDecision
from easylora.autopilot.model_analysis import ModelProfile
from easylora.autopilot.planner import AutopilotPlan
from easylora.cli.main import app
from easylora.config import TrainConfig


runner = CliRunner()


def test_init_config_smoke(tmp_path):
    out = tmp_path / "config.yaml"
    result = runner.invoke(app, ["init-config", "--template", "sft-lora", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_train_dry_run_with_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "model": {"base_model": "gpt2"},
                "data": {"dataset_path": "train.jsonl", "format": "raw"},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_autopilot_plan_saves_report(monkeypatch, tmp_path):
    plan = _make_plan(str(tmp_path))
    monkeypatch.setattr("easylora.cli.autopilot_cmd.autopilot_plan", lambda **_kwargs: plan)
    result = runner.invoke(
        app,
        [
            "autopilot",
            "plan",
            "--model",
            "gpt2",
            "--dataset",
            "data.jsonl",
            "--output-dir",
            str(tmp_path),
            "--save-report",
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "autopilot_report.md").exists()


def test_dpo_dry_run(tmp_path):
    cfg_path = tmp_path / "dpo.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "model": {"base_model": "gpt2"},
                "data": {"dataset_path": "prefs.jsonl"},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["align", "dpo", "--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0
    assert "DPO config validated" in result.output


def _make_plan(output_dir: str) -> AutopilotPlan:
    cfg = TrainConfig.model_validate(
        {
            "model": {"base_model": "gpt2"},
            "data": {"dataset_path": "data.jsonl"},
            "output": {"output_dir": output_dir, "allow_overwrite": True},
        }
    )
    return AutopilotPlan(
        config=cfg,
        quality="fast",
        hardware=HardwareProfile(
            python_version="3.11.0",
            platform="test",
            torch_version="2.0.0",
            cuda_available=True,
            gpu_name="T4",
            gpu_vram_gb=16.0,
            bf16_supported=False,
            fp16_supported=True,
            mps_available=False,
            bitsandbytes_available=True,
        ),
        dataset=DatasetProfile(
            examples=10,
            sampled_examples=10,
            inferred_format="raw",
            text_field="text",
            p50_tokens=32,
            p95_tokens=64,
            max_tokens=80,
        ),
        model=ModelProfile(
            model_name="gpt2",
            architecture="GPT2LMHeadModel",
            model_type="gpt2",
            hidden_size=768,
            num_hidden_layers=12,
            context_length=1024,
            estimated_params_b=0.1,
        ),
        decision=PlanDecision(
            use_qlora=False,
            max_seq_len=256,
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            batch_size=2,
            grad_accum=8,
            epochs=1,
            max_steps=-1,
            save_steps=100,
            logging_steps=10,
            estimated_vram_gb=2.5,
            estimated_steps_per_sec=1.0,
            reasons=["test"],
        ),
    )
