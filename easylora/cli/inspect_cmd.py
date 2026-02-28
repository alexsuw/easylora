"""CLI ``inspect-targets`` subcommand — show recommended LoRA target modules."""

from typing import Annotated

import typer
from rich.console import Console

console = Console()


def inspect_targets(
    model: Annotated[str, typer.Option("--model", "-m", help="HF model ID or local path")],
    trust_remote_code: Annotated[
        bool, typer.Option("--trust-remote-code", help="Trust remote code")
    ] = False,
) -> None:
    """Show recommended LoRA target modules for a model."""
    from transformers import AutoConfig

    from easylora.lora.targets import resolve_target_modules_from_config

    console.print(f"[bold cyan]Inspecting targets for:[/] {model}\n")

    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception as exc:
        console.print(f"[red]Failed to load model config:[/] {exc}")
        raise typer.Exit(1) from None

    arch = None
    if hasattr(config, "architectures") and config.architectures:
        arch = config.architectures[0]
    model_type = getattr(config, "model_type", None)

    console.print(f"  Architecture:  {arch or 'unknown'}")
    console.print(f"  Model type:    {model_type or 'unknown'}")

    targets, source = resolve_target_modules_from_config(config)

    console.print(f"  Detection:     {source}")
    console.print(f"\n[bold green]Recommended target_modules:[/] {targets}")
    console.print(f"\nTo use in config:\n  lora:\n    target_modules: {targets}")
