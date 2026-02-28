"""Tests for config parsing, validation, and loading."""

from __future__ import annotations

import json
import textwrap

import pytest
import yaml
from pydantic import ValidationError

from easylora.config import (
    DataConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    load_config,
)
from easylora.exceptions import EasyLoRAConfigError

# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_base_model_required(self):
        with pytest.raises(ValidationError, match="base_model"):
            ModelConfig()  # type: ignore[call-arg]

    def test_minimal_valid(self):
        cfg = ModelConfig(base_model="gpt2")
        assert cfg.base_model == "gpt2"
        assert cfg.tokenizer is None
        assert cfg.torch_dtype == "auto"

    def test_4bit_and_8bit_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ModelConfig(base_model="gpt2", load_in_4bit=True, load_in_8bit=True)

    def test_4bit_alone_ok(self):
        cfg = ModelConfig(base_model="gpt2", load_in_4bit=True)
        assert cfg.load_in_4bit is True
        assert cfg.load_in_8bit is False


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class TestDataConfig:
    def test_requires_dataset_source(self):
        with pytest.raises(ValidationError, match=r"dataset_path.*dataset_name"):
            DataConfig()

    def test_local_path(self):
        cfg = DataConfig(dataset_path="data.jsonl")
        assert cfg.dataset_path == "data.jsonl"

    def test_hub_name(self):
        cfg = DataConfig(dataset_name="tatsu-lab/alpaca")
        assert cfg.dataset_name == "tatsu-lab/alpaca"

    def test_max_seq_len_bounds(self):
        with pytest.raises(ValidationError):
            DataConfig(dataset_path="d.json", max_seq_len=16)  # below minimum 32


# ---------------------------------------------------------------------------
# OutputConfig
# ---------------------------------------------------------------------------


class TestOutputConfig:
    def test_push_to_hub_requires_repo_id(self):
        with pytest.raises(ValidationError, match="hub_repo_id"):
            OutputConfig(push_to_hub=True)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults_applied(self):
        cfg = TrainConfig(
            model=ModelConfig(base_model="gpt2"),
            data=DataConfig(dataset_path="data.json"),
        )
        assert cfg.lora.r == 16
        assert cfg.optim.lr == 2e-4
        assert cfg.training.epochs == 3
        assert cfg.repro.seed == 42

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            TrainConfig(
                model=ModelConfig(base_model="gpt2"),
                data=DataConfig(dataset_path="d.json"),
                bogus_field="oops",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# load_config from file
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        data = {
            "model": {"base_model": "gpt2"},
            "data": {"dataset_path": "train.jsonl"},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data))
        cfg = load_config(p)
        assert cfg.model.base_model == "gpt2"

    def test_load_json(self, tmp_path):
        data = {
            "model": {"base_model": "gpt2"},
            "data": {"dataset_name": "tatsu-lab/alpaca"},
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(data))
        cfg = load_config(p)
        assert cfg.data.dataset_name == "tatsu-lab/alpaca"

    def test_missing_file(self, tmp_path):
        with pytest.raises(EasyLoRAConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(
            textwrap.dedent("""\
            model:
              base_model: gpt2
            data:
              - this is a list not a dict
        """)
        )
        with pytest.raises(EasyLoRAConfigError, match="validation failed"):
            load_config(p)

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "cfg.toml"
        p.write_text("")
        with pytest.raises(EasyLoRAConfigError, match="Unsupported"):
            load_config(p)
