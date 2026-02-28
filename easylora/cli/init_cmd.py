"""CLI ``init-config`` subcommand — generate starter config files."""

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console

console = Console()

_TEMPLATES: dict[str, dict] = {
    "sft-lora": {
        "model": {
            "base_model": "meta-llama/Llama-3.2-1B",
            "torch_dtype": "auto",
        },
        "data": {
            "dataset_name": "tatsu-lab/alpaca",
            "format": "alpaca",
            "max_seq_len": 2048,
            "val_split_ratio": 0.02,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "auto",
        },
        "optim": {
            "lr": 2e-4,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "grad_accum": 4,
            "logging_steps": 10,
            "save_steps": 500,
            "gradient_checkpointing": True,
        },
        "output": {
            "output_dir": "./output",
            "run_name": "sft-lora-run",
        },
        "repro": {
            "seed": 42,
        },
    },
    "sft-qlora": {
        "model": {
            "base_model": "meta-llama/Llama-3.2-1B",
            "torch_dtype": "auto",
            "load_in_4bit": True,
        },
        "data": {
            "dataset_name": "tatsu-lab/alpaca",
            "format": "alpaca",
            "max_seq_len": 2048,
            "val_split_ratio": 0.02,
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": "auto",
        },
        "optim": {
            "lr": 2e-4,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "grad_accum": 4,
            "logging_steps": 10,
            "save_steps": 500,
            "gradient_checkpointing": True,
        },
        "output": {
            "output_dir": "./output-qlora",
            "run_name": "sft-qlora-run",
        },
        "repro": {
            "seed": 42,
        },
    },
}


def init_config(
    template: Annotated[
        str,
        typer.Option("--template", "-t", help="Template name: sft-lora or sft-qlora"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = Path("easylora_config.yaml"),
) -> None:
    """Generate a starter YAML config file."""
    if template not in _TEMPLATES:
        console.print(
            f"[red]Unknown template '{template}'.[/] Available: {', '.join(_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    if output.exists():
        console.print(f"[red]File already exists:[/] {output}")
        raise typer.Exit(1)

    content = yaml.dump(_TEMPLATES[template], default_flow_style=False, sort_keys=False)
    output.write_text(content, encoding="utf-8")
    console.print(f"[bold green]Config written to {output}[/]")
    console.print(f"\nTo train: easylora train --config {output}")
