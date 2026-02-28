"""Generate a model card (README.md) for saved adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_model_card(
    output_dir: str | Path,
    base_model: str,
    lora_config: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    """Write a simple model card README to the adapter output directory.

    Args:
        output_dir: Directory to write the README into.
        base_model: Name/path of the base model.
        lora_config: LoRA configuration dict (r, alpha, etc.).
        summary: Training summary dict (final_loss, steps, etc.).

    Returns:
        Path to the generated README.
    """
    lines = [
        "---",
        "library_name: peft",
        f"base_model: {base_model}",
        "tags:",
        "  - easylora",
        "  - lora",
        "---",
        "",
        f"# LoRA Adapter for {base_model}",
        "",
        "Fine-tuned with [easylora](https://github.com/alexsuw/easylora).",
        "",
    ]

    if lora_config:
        lines.extend(
            [
                "## LoRA Configuration",
                "",
                "| Parameter | Value |",
                "|---|---|",
            ]
        )
        for k, v in lora_config.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    if summary:
        lines.extend(
            [
                "## Training Summary",
                "",
                "| Metric | Value |",
                "|---|---|",
            ]
        )
        for k, v in summary.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.extend(
        [
            "## Usage",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM",
            "",
            f'base = AutoModelForCausalLM.from_pretrained("{base_model}")',
            'model = PeftModel.from_pretrained(base, "<adapter_dir>")',
            "```",
            "",
        ]
    )

    out = Path(output_dir)
    readme_path = out / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    return readme_path
