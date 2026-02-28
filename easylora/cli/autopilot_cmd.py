"""CLI subcommands for autopilot planning."""

from __future__ import annotations

from typing import Annotated, Optional, cast

import typer
import yaml
from rich.console import Console

from easylora.autopilot.api import autopilot_plan
from easylora.autopilot.presets import AutopilotQuality

console = Console()
app = typer.Typer(
    help="Autopilot helpers for planning and no-config training.",
    no_args_is_help=True,
)


@app.command("plan")
def plan_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="Base model ID/path")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset path or HF name")],
    quality: Annotated[
        str,
        typer.Option("--quality", help="Autopilot quality preset: fast|balanced|high"),
    ] = "balanced",
    output_dir: Annotated[
        str,
        typer.Option("--output-dir", help="Output directory for resolved config"),
    ] = "./output",
    subset: Annotated[
        Optional[str],
        typer.Option("--subset", help="Dataset subset/config name"),
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Dataset split")] = "train",
    print_config: Annotated[
        bool,
        typer.Option("--print-config", help="Print resolved TrainConfig YAML"),
    ] = False,
) -> None:
    """Dry-run autopilot planning with transparent strategy output."""
    if quality not in {"fast", "balanced", "high"}:
        raise typer.BadParameter("--quality must be one of: fast, balanced, high.")
    quality_value = cast(AutopilotQuality, quality)

    plan = autopilot_plan(
        model=model,
        dataset=dataset,
        quality=quality_value,
        output_dir=output_dir,
        subset=subset,
        split=split,
    )
    console.print("\n[bold cyan]AUTOPILOT PLAN[/]\n")
    for line in plan.to_pretty_lines()[2:]:
        console.print(line)

    console.print("\n[bold]Reasoning:[/]")
    for reason in plan.decision.reasons:
        console.print(f"  - {reason}")

    if print_config:
        console.print("\n[bold]Resolved TrainConfig:[/]")
        console.print(
            yaml.dump(plan.config.model_dump(), default_flow_style=False, sort_keys=False)
        )
