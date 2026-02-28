"""CLI ``doctor`` subcommand — environment diagnostics."""

import platform
import sys

from rich.console import Console
from rich.table import Table

from easylora.autopilot.hardware import detect_hardware


def doctor() -> None:
    """Print environment diagnostics for debugging."""
    console = Console()
    table = Table(title="easylora doctor", show_header=True, header_style="bold")
    table.add_column("Component", style="cyan")
    table.add_column("Value")

    vi = sys.version_info
    table.add_row("Python", f"{vi.major}.{vi.minor}.{vi.micro}")
    table.add_row("Platform", platform.platform())

    try:
        from importlib.metadata import version as pkg_version

        table.add_row("easylora", pkg_version("easylora"))
    except Exception:
        table.add_row("easylora", "[red]not installed[/red]")

    try:
        import torch

        table.add_row("PyTorch", torch.__version__)
        hw = detect_hardware()
        table.add_row("CUDA available", str(hw.cuda_available))
        if hw.cuda_available:
            table.add_row("CUDA version", torch.version.cuda or "N/A")
            table.add_row("GPU", hw.gpu_name or "N/A")
            table.add_row("bf16 (CUDA)", str(hw.bf16_supported))
        if hw.mps_available:
            table.add_row("MPS available", "True")
    except ImportError:
        table.add_row("PyTorch", "[red]not installed[/red]")

    _add_version_row(table, "transformers")
    _add_version_row(table, "peft")
    _add_version_row(table, "accelerate")
    _add_version_row(table, "datasets")

    try:
        import bitsandbytes  # noqa: F401

        _add_version_row(table, "bitsandbytes")
    except ImportError:
        table.add_row("bitsandbytes", "[dim]not installed (needed for QLoRA)[/dim]")

    try:
        import wandb  # noqa: F401

        _add_version_row(table, "wandb")
    except ImportError:
        table.add_row("wandb", "[dim]not installed (optional)[/dim]")

    console.print(table)


def _add_version_row(table: Table, package: str) -> None:
    try:
        from importlib.metadata import version as pkg_version

        table.add_row(package, pkg_version(package))
    except Exception:
        table.add_row(package, "[red]not installed[/red]")
