"""Tests for dataset formatting (alpaca / chatml / raw)."""

from __future__ import annotations

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from easylora.config import DataConfig
from easylora.data.formatting import format_examples


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


# ---------------------------------------------------------------------------
# Raw format
# ---------------------------------------------------------------------------


class TestRawFormat:
    def test_tokenises_text_field(self, tokenizer):
        ds = Dataset.from_dict({"text": ["Hello world", "Goodbye world"]})
        cfg = DataConfig(dataset_path="dummy.json", format="raw", text_field="text", max_seq_len=64)
        result = format_examples(ds, cfg, tokenizer)

        assert "input_ids" in result.column_names
        assert "labels" in result.column_names
        assert "attention_mask" in result.column_names
        assert len(result) == 2

    def test_labels_equal_input_ids(self, tokenizer):
        ds = Dataset.from_dict({"text": ["test"]})
        cfg = DataConfig(dataset_path="dummy.json", format="raw", max_seq_len=64)
        result = format_examples(ds, cfg, tokenizer)
        assert result[0]["input_ids"] == result[0]["labels"]

    def test_truncation(self, tokenizer):
        long_text = "word " * 500
        ds = Dataset.from_dict({"text": [long_text]})
        cfg = DataConfig(dataset_path="dummy.json", format="raw", max_seq_len=32)
        result = format_examples(ds, cfg, tokenizer)
        assert len(result[0]["input_ids"]) <= 32


# ---------------------------------------------------------------------------
# Alpaca format
# ---------------------------------------------------------------------------


class TestAlpacaFormat:
    def test_basic_alpaca(self, tokenizer):
        ds = Dataset.from_dict(
            {
                "instruction": ["Summarize the text."],
                "input": ["The quick brown fox."],
                "output": ["A fox."],
            }
        )
        cfg = DataConfig(dataset_path="dummy.json", format="alpaca", max_seq_len=256)
        result = format_examples(ds, cfg, tokenizer)

        assert len(result) == 1
        labels = result[0]["labels"]
        # Prompt tokens should be masked to -100
        assert -100 in labels
        # At least some tokens should be non-masked (the output)
        assert any(t != -100 for t in labels)

    def test_alpaca_no_input(self, tokenizer):
        ds = Dataset.from_dict(
            {
                "instruction": ["Say hello."],
                "input": [""],
                "output": ["Hello!"],
            }
        )
        cfg = DataConfig(dataset_path="dummy.json", format="alpaca", max_seq_len=256)
        result = format_examples(ds, cfg, tokenizer)
        assert len(result[0]["input_ids"]) > 0


# ---------------------------------------------------------------------------
# ChatML format
# ---------------------------------------------------------------------------


class TestChatMLFormat:
    def test_chatml_basic(self, tokenizer):
        # GPT-2 doesn't have a chat template by default, so set one
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ message['role'] }}: {{ message['content'] }}\n"
                "{% endfor %}"
            )

        ds = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello!"},
                    ]
                ]
            }
        )
        cfg = DataConfig(dataset_path="dummy.json", format="chatml", max_seq_len=256)
        result = format_examples(ds, cfg, tokenizer)

        assert len(result) == 1
        assert "input_ids" in result.column_names
        labels = result[0]["labels"]
        # Prompt (user turn) should be masked
        assert -100 in labels
