"""Tests for structured evaluation reports."""

from __future__ import annotations

import json

from easylora.eval.report import build_eval_report, report_to_markdown, save_eval_report


def test_build_eval_report_contains_metrics():
    report = build_eval_report(
        base_model="gpt2",
        adapter_dir="./adapter",
        dataset="eval.jsonl",
        perplexity=12.34,
        generations=[{"prompt": "Hi", "adapter_output": "Hello"}],
    )
    assert report["metrics"]["perplexity"] == 12.34
    assert report["generations"][0]["prompt"] == "Hi"


def test_save_eval_report_json_and_markdown(tmp_path):
    report = build_eval_report(
        base_model="gpt2",
        adapter_dir="./adapter",
        dataset="eval.jsonl",
        perplexity=1.23,
    )
    json_path = save_eval_report(report, tmp_path / "eval.json")
    md_path = save_eval_report(report, tmp_path / "eval.md")

    assert json.loads(json_path.read_text())["base_model"] == "gpt2"
    assert "# easylora Evaluation Report" in md_path.read_text()
    assert "Perplexity" in report_to_markdown(report)
