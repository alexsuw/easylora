"""CLI ``train`` subcommand."""

from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from easylora.config import load_config
from easylora.train.trainer import EasyLoRATrainer

console = Console()


def train(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML/JSON config file")],
    force: Annotated[bool, typer.Option("--force", help="Allow overwriting output dir")] = False,
    override: Annotated[
        Optional[list[str]],
        typer.Option("--set", help="Override config values as key=value (dot notation)"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config and print settings without training"),
    ] = False,
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print resolved config as YAML and exit"),
    ] = False,
) -> None:
    """Fine-tune a model with LoRA / QLoRA."""
    cfg = load_config(config)

    if force:
        cfg.output.allow_overwrite = True

    if override:
        _apply_overrides(cfg, override)

    if print_config:
        console.print(yaml.dump(cfg.model_dump(), default_flow_style=False, sort_keys=False))
        raise typer.Exit(0)

    if dry_run:
        console.print("[bold cyan]Dry run[/] — config validated successfully.\n")
        console.print(f"  base_model:   {cfg.model.base_model}")
        console.print(f"  dataset:      {cfg.data.dataset_path or cfg.data.dataset_name}")
        console.print(f"  format:       {cfg.data.format}")
        console.print(f"  lora r:       {cfg.lora.r}")
        console.print(f"  lora alpha:   {cfg.lora.alpha}")
        console.print(f"  targets:      {cfg.lora.target_modules}")
        console.print(f"  epochs:       {cfg.training.epochs}")
        console.print(f"  batch_size:   {cfg.training.batch_size}")
        console.print(f"  lr:           {cfg.optim.lr}")
        console.print(f"  output_dir:   {cfg.output.output_dir}")
        console.print(f"  seed:         {cfg.repro.seed}")
        console.print("\n[dim]No training performed.[/dim]")
        raise typer.Exit(0)

    console.print(f"[bold green]easylora train[/] — config: {config}")
    console.print(f"  base_model: {cfg.model.base_model}")
    console.print(f"  output_dir: {cfg.output.output_dir}")

    trainer = EasyLoRATrainer(cfg)
    artifacts = trainer.fit()

    base_model = cfg.model.base_model
    adapter = artifacts.adapter_dir

    summary = Panel(
        f"[bold]Adapter:[/]  {artifacts.adapter_dir}\n"
        f"[bold]Config:[/]   {artifacts.config_path}\n"
        f"[bold]Log:[/]      {artifacts.log_path}\n"
        f"[bold]Summary:[/]  {artifacts.summary_path}\n"
        f"\n[dim]Next steps:[/dim]\n"
        f"  easylora eval -m {base_model} -a {adapter} -d <eval_data>\n"
        f"  easylora merge -m {base_model} -a {adapter} -o ./merged",
        title="[bold green]Training complete[/]",
        border_style="green",
    )
    console.print(summary)


def _apply_overrides(cfg: object, overrides: list[str]) -> None:
    """Apply dot-notation overrides like ``model.base_model=foo``."""
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise typer.BadParameter(f"Unknown config key: {key}")
            obj = getattr(obj, part)
        field = parts[-1]
        if not hasattr(obj, field):
            raise typer.BadParameter(f"Unknown config key: {key}")

        current = getattr(obj, field)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")  # type: ignore[assignment]
        elif isinstance(current, int):
            value = int(value)  # type: ignore[assignment]
        elif isinstance(current, float):
            value = float(value)  # type: ignore[assignment]

        setattr(obj, field, value)
