"""easylora CLI entrypoint.

Usage::

    easylora train --config config.yaml
    easylora eval --base-model ... --adapter-dir ... --dataset ...
    easylora merge --base-model ... --adapter-dir ... --output-dir ...
    easylora doctor
    easylora inspect-targets --model ...
    easylora init-config --template sft-lora
"""

import typer

from easylora.cli.doctor_cmd import doctor
from easylora.cli.eval_cmd import eval_cmd, merge
from easylora.cli.init_cmd import init_config
from easylora.cli.inspect_cmd import inspect_targets
from easylora.cli.train_cmd import train

app = typer.Typer(
    name="easylora",
    help="Batteries-included LoRA / QLoRA fine-tuning toolkit.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

app.command("train")(train)
app.command("eval")(eval_cmd)
app.command("merge")(merge)
app.command("doctor")(doctor)
app.command("inspect-targets")(inspect_targets)
app.command("init-config")(init_config)

if __name__ == "__main__":
    app()
