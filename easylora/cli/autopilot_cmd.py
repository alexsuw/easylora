"""CLI subcommands for autopilot planning."""

from __future__ import annotations

from typing import Annotated, Optional, cast

import typer
import yaml
from rich.console import Console
from rich.table import Table

from easylora.autopilot.api import autopilot_plan, save_autopilot_report
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
    save_report: Annotated[
        bool,
        typer.Option("--save-report", help="Write autopilot_report.md to --output-dir"),
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
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Decision")
    table.add_column("Value")
    table.add_row("Model", plan.model.model_name)
    table.add_row("Dataset examples", f"{plan.dataset.examples:,}")
    table.add_row("Format", plan.dataset.inferred_format)
    table.add_row("GPU", plan.hardware.gpu_name or "CPU/MPS")
    table.add_row("Strategy", "QLoRA" if plan.decision.use_qlora else "LoRA")
    table.add_row("Seq length", str(plan.decision.max_seq_len))
    table.add_row("Batch / accum", f"{plan.decision.batch_size} / {plan.decision.grad_accum}")
    table.add_row("Learning rate", f"{plan.decision.learning_rate:.2e}")
    table.add_row("Estimated VRAM", f"~{plan.decision.estimated_vram_gb} GB")
    console.print(table)

    console.print("\n[bold]Reasoning:[/]")
    for reason in plan.decision.reasons:
        console.print(f"  - {reason}")
    warnings = plan.warnings()
    if warnings:
        console.print("\n[bold yellow]Warnings:[/]")
        for warning in warnings:
            console.print(f"  - {warning}")

    if print_config:
        console.print("\n[bold]Resolved TrainConfig:[/]")
        console.print(
            yaml.dump(plan.config.model_dump(), default_flow_style=False, sort_keys=False)
        )
    if save_report:
        report_path = save_autopilot_report(plan, output_dir)
        console.print(f"\n[bold green]Markdown report written to {report_path}[/]")
