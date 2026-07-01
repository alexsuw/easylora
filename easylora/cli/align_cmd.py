"""CLI commands for preference alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from easylora.align import train_dpo
from easylora.config import DPOTrainConfig
from easylora.exceptions import EasyLoRAConfigError

console = Console()
app = typer.Typer(help="Preference alignment commands.", no_args_is_help=True)


@app.command("dpo")
def dpo_cmd(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to DPO YAML/JSON config")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config without training"),
    ] = False,
) -> None:
    """Run DPO preference tuning via TRL."""
    cfg = _load_dpo_config(config)
    if dry_run:
        console.print("[bold cyan]Dry run[/] — DPO config validated successfully.\n")
        console.print(f"  base_model: {cfg.model.base_model}")
        console.print(f"  dataset:    {cfg.data.dataset_path or cfg.data.dataset_name}")
        console.print(f"  beta:       {cfg.dpo.beta}")
        console.print(f"  output:     {cfg.output.output_dir}")
        raise typer.Exit(0)

    artifacts = train_dpo(cfg)
    console.print(
        Panel(
            f"[bold]Adapter:[/] {artifacts.adapter_dir}\n"
            f"[bold]Config:[/]  {artifacts.config_path}\n"
            f"[bold]Summary:[/] {artifacts.summary_path}",
            title="[bold green]DPO complete[/]",
            border_style="green",
        )
    )


def _load_dpo_config(path: Path) -> DPOTrainConfig:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif path.suffix.lower() == ".json":
        import json

        data = json.loads(raw)
    else:
        raise EasyLoRAConfigError("DPO config must be .yaml, .yml, or .json")
    return DPOTrainConfig.model_validate(data)
