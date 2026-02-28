"""Tests for autopilot planner and public API."""

from __future__ import annotations

from pathlib import Path

from easylora.autopilot import api
from easylora.autopilot.dataset_analysis import DatasetProfile
from easylora.autopilot.hardware import HardwareProfile
from easylora.autopilot.heuristics import PlanDecision
from easylora.autopilot.model_analysis import ModelProfile
from easylora.autopilot.planner import AutopilotPlan, plan_autopilot
from easylora.config import RunArtifacts, TrainConfig


def test_planner_produces_valid_trainconfig(monkeypatch):
    monkeypatch.setattr(
        "easylora.autopilot.planner.detect_hardware",
        lambda: HardwareProfile(
            python_version="3.11.0",
            platform="test",
            torch_version="2.0.0",
            cuda_available=True,
            gpu_name="A10",
            gpu_vram_gb=24.0,
            bf16_supported=True,
            fp16_supported=True,
            mps_available=False,
            bitsandbytes_available=True,
        ),
    )
    monkeypatch.setattr(
        "easylora.autopilot.planner.analyze_model",
        lambda _model, trust_remote_code=False: ModelProfile(
            model_name="meta-llama/Llama-3.2-1B",
            architecture="LlamaForCausalLM",
            model_type="llama",
            hidden_size=2048,
            num_hidden_layers=16,
            context_length=8192,
            estimated_params_b=1.0,
        ),
    )
    monkeypatch.setattr(
        "easylora.autopilot.planner.load_tokenizer",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        "easylora.autopilot.planner.analyze_dataset",
        lambda *_args, **_kwargs: DatasetProfile(
            examples=52_000,
            sampled_examples=512,
            inferred_format="alpaca",
            text_field="instruction",
            p50_tokens=250,
            p95_tokens=980,
            max_tokens=1600,
        ),
    )

    plan = plan_autopilot(
        model="meta-llama/Llama-3.2-1B",
        dataset="tatsu-lab/alpaca",
    )
    assert isinstance(plan.config, TrainConfig)
    assert plan.config.data.max_seq_len >= 1024
    assert plan.config.lora.r > 0


def test_autopilot_train_writes_report_and_resolved_config(monkeypatch, tmp_path):
    out = tmp_path / "run"
    plan = _make_test_plan(str(out))

    monkeypatch.setattr("easylora.autopilot.api.autopilot_plan", lambda **_kwargs: plan)

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config

        def fit(self):
            out.mkdir(parents=True, exist_ok=True)
            return RunArtifacts(
                adapter_dir=str(out / "adapter"),
                config_path=str(out / "train_config.json"),
                log_path=str(out / "train_log.jsonl"),
                summary_path=str(out / "summary.json"),
            )

    monkeypatch.setattr(api, "EasyLoRATrainer", _FakeTrainer)

    artifacts = api.autopilot_train(model="x", dataset="y", output_dir=str(out))
    assert Path(artifacts.summary_path).name == "summary.json"
    assert (out / "resolved_config.yaml").exists()
    assert (out / "autopilot_report.json").exists()


def _make_test_plan(output_dir: str) -> AutopilotPlan:
    cfg = TrainConfig.model_validate(
        {
            "model": {"base_model": "hf-internal-testing/tiny-random-LlamaForCausalLM"},
            "data": {"dataset_name": "tatsu-lab/alpaca", "format": "alpaca"},
            "output": {"output_dir": output_dir, "allow_overwrite": True},
        }
    )
    return AutopilotPlan(
        config=cfg,
        quality="balanced",
        hardware=HardwareProfile(
            python_version="3.11.0",
            platform="test",
            torch_version="2.0.0",
            cuda_available=True,
            gpu_name="A10",
            gpu_vram_gb=24.0,
            bf16_supported=True,
            fp16_supported=True,
            mps_available=False,
            bitsandbytes_available=True,
        ),
        dataset=DatasetProfile(
            examples=1000,
            sampled_examples=32,
            inferred_format="alpaca",
            text_field="instruction",
            p50_tokens=200,
            p95_tokens=900,
            max_tokens=1600,
        ),
        model=ModelProfile(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            architecture="LlamaForCausalLM",
            model_type="llama",
            hidden_size=256,
            num_hidden_layers=2,
            context_length=2048,
            estimated_params_b=0.05,
        ),
        decision=PlanDecision(
            use_qlora=False,
            max_seq_len=1024,
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-4,
            batch_size=2,
            grad_accum=8,
            epochs=3,
            max_steps=-1,
            save_steps=100,
            logging_steps=10,
            estimated_vram_gb=8.0,
            estimated_steps_per_sec=1.0,
            reasons=["test"],
        ),
    )
