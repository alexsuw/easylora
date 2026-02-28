"""Custom TrainerCallbacks for JSONL logging and run summary."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class JsonlLoggingCallback(TrainerCallback):
    """Appends training metrics as JSON lines to a log file.

    Written each time the Trainer logs (controlled by ``logging_steps``).
    """

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "ts": time.time(),
            **{k: v for k, v in logs.items() if isinstance(v, (int, float, str, bool))},
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._file.close()


class TrainingSummaryCallback(TrainerCallback):
    """Captures end-of-training metrics and writes a summary JSON file."""

    def __init__(self, summary_path: str | Path) -> None:
        self._path = Path(summary_path)
        self._start_time: float | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.time()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        elapsed = time.time() - (self._start_time or time.time())

        log_history = state.log_history or []
        final_loss: float | None = None
        for entry in reversed(log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

        summary = {
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "final_loss": final_loss,
            "runtime_seconds": round(elapsed, 2),
            "best_metric": state.best_metric,
        }

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
