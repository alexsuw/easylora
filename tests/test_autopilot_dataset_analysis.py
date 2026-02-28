"""Tests for autopilot dataset analysis."""

from __future__ import annotations

from datasets import Dataset

from easylora.autopilot.dataset_analysis import analyze_dataset, infer_dataset_format


class _FakeTokenizer:
    def __call__(self, text: str, truncation: bool = False, padding: bool = False):
        _ = truncation, padding
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        _ = tokenize, add_generation_prompt
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def test_infer_dataset_format():
    assert infer_dataset_format(["messages"])[0] == "chatml"
    assert infer_dataset_format(["instruction", "output"])[0] == "alpaca"
    assert infer_dataset_format(["text"])[0] == "raw"


def test_analyze_dataset_raw(monkeypatch):
    ds = Dataset.from_dict({"text": ["one two", "one two three four five"]})
    monkeypatch.setattr(
        "easylora.autopilot.dataset_analysis.load_dataset_any",
        lambda _cfg: ds,
    )

    profile = analyze_dataset(
        "dummy-dataset",
        tokenizer=_FakeTokenizer(),
        max_samples=64,
    )

    assert profile.examples == 2
    assert profile.inferred_format == "raw"
    assert profile.text_field == "text"
    assert profile.p95_tokens >= profile.p50_tokens
