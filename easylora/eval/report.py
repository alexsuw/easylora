"""Structured evaluation reports for adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from easylora.utils.io import save_json


def build_eval_report(
    *,
    base_model: str,
    adapter_dir: str,
    dataset: str,
    perplexity: float,
    generations: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serialisable adapter evaluation report."""
    return {
        "base_model": base_model,
        "adapter_dir": adapter_dir,
        "dataset": dataset,
        "metrics": {"perplexity": perplexity},
        "generations": generations or [],
    }


def report_to_markdown(report: dict[str, Any]) -> str:
    """Render an eval report as markdown."""
    lines = [
        "# easylora Evaluation Report",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Base model | {report['base_model']} |",
        f"| Adapter | {report['adapter_dir']} |",
        f"| Dataset | {report['dataset']} |",
        f"| Perplexity | {report['metrics']['perplexity']:.4f} |",
    ]
    generations = report.get("generations") or []
    if generations:
        lines.extend(["", "## Generation samples", ""])
        for item in generations:
            lines.extend(
                [
                    f"### Prompt: `{item['prompt']}`",
                    "",
                    "**Adapter output**",
                    "",
                    item.get("adapter_output", ""),
                    "",
                ]
            )
            base_output = item.get("base_output")
            if base_output is not None:
                lines.extend(["**Base output**", "", base_output, ""])
    return "\n".join(lines) + "\n"


def save_eval_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save an eval report as JSON or Markdown based on the file suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".md", ".markdown"}:
        path.write_text(report_to_markdown(report), encoding="utf-8")
        return path
    return save_json(report, path)
