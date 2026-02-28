"""CLI ``eval`` and ``merge`` subcommands."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

console = Console()


def eval_cmd(
    base_model: Annotated[str, typer.Option("--base-model", "-m", help="Base model ID or path")],
    adapter_dir: Annotated[Path, typer.Option("--adapter-dir", "-a", help="Path to saved adapter")],
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="Dataset path or HF name")],
    max_samples: Annotated[
        Optional[int], typer.Option("--max-samples", help="Max samples to evaluate")
    ] = None,
    max_seq_len: Annotated[int, typer.Option("--max-seq-len", help="Max sequence length")] = 2048,
    prompts: Annotated[
        Optional[list[str]],
        typer.Option("--prompt", "-p", help="Prompts for generation sanity check"),
    ] = None,
) -> None:
    """Evaluate a LoRA adapter (perplexity + optional generation)."""
    from easylora.config import DataConfig, ModelConfig
    from easylora.data.formatting import format_examples
    from easylora.data.loaders import load_dataset_any
    from easylora.eval.generate import generate_samples
    from easylora.eval.perplexity import compute_perplexity
    from easylora.lora.adapter import load_adapter
    from easylora.utils.hf import load_tokenizer

    console.print(f"[bold cyan]easylora eval[/] — model: {base_model}, adapter: {adapter_dir}")

    model_cfg = ModelConfig(base_model=base_model)
    tokenizer = load_tokenizer(model_cfg)
    model = load_adapter(base_model, adapter_dir)

    is_local = Path(dataset).exists()
    data_cfg = DataConfig(
        dataset_path=dataset if is_local else None,
        dataset_name=dataset if not is_local else None,
        max_seq_len=max_seq_len,
    )
    raw = load_dataset_any(data_cfg)
    formatted = format_examples(raw, data_cfg, tokenizer)

    ppl = compute_perplexity(model, tokenizer, formatted, max_seq_len, max_samples=max_samples)
    console.print(f"\n[bold]Perplexity:[/] {ppl:.4f}")

    if prompts:
        console.print("\n[bold]Generation samples:[/]")
        outputs = generate_samples(model, tokenizer, prompts)
        for p, o in zip(prompts, outputs, strict=True):
            console.print(f"  [dim]Prompt:[/] {p}")
            console.print(f"  [green]Output:[/] {o}\n")


def merge(
    base_model: Annotated[str, typer.Option("--base-model", "-m", help="Base model ID or path")],
    adapter_dir: Annotated[Path, typer.Option("--adapter-dir", "-a", help="Path to saved adapter")],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", "-o", help="Where to save merged model")
    ],
    trust_remote_code: Annotated[
        bool, typer.Option("--trust-remote-code", help="Trust remote code")
    ] = False,
) -> None:
    """Merge LoRA adapter into the base model and save full weights."""
    from easylora.lora.merge import merge_adapter

    console.print(f"[bold magenta]easylora merge[/] — {base_model} + {adapter_dir} -> {output_dir}")
    merge_adapter(
        base_model,
        adapter_dir,
        output_dir,
        trust_remote_code=trust_remote_code,
    )
    console.print(f"[bold green]Merged model saved to {output_dir}[/]")
