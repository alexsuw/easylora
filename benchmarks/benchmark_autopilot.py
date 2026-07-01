"""Benchmark Autopilot estimates against optional short training runs."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from easylora import autopilot_plan, autopilot_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--quality", default="fast", choices=["fast", "balanced", "high"])
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/result.json"))
    parser.add_argument("--run-training", action="store_true", help="Run a short training job")
    parser.add_argument("--max-analysis-samples", type=int, default=512)
    args = parser.parse_args()

    plan = autopilot_plan(
        model=args.model,
        dataset=args.dataset,
        quality=args.quality,
        subset=args.subset,
        split=args.split,
        output_dir=str(args.output.parent / "run"),
        max_analysis_samples=args.max_analysis_samples,
        allow_overwrite=True,
    )
    result: dict[str, Any] = {
        "model": args.model,
        "dataset": args.dataset,
        "quality": args.quality,
        "versions": _versions(),
        "plan": plan.report(),
        "estimated_vram_gb": plan.decision.estimated_vram_gb,
        "estimated_steps_per_sec": plan.decision.estimated_steps_per_sec,
        "actual_peak_vram_gb": None,
        "actual_steps_per_sec": None,
    }

    if args.run_training:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        artifacts = autopilot_train(
            model=args.model,
            dataset=args.dataset,
            quality=args.quality,
            subset=args.subset,
            split=args.split,
            output_dir=str(args.output.parent / "run"),
            max_analysis_samples=args.max_analysis_samples,
            allow_overwrite=True,
        )
        elapsed = time.perf_counter() - start
        result["training_elapsed_sec"] = round(elapsed, 3)
        result["artifacts"] = artifacts.model_dump()
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            result["actual_peak_vram_gb"] = round(peak, 3)
        result["actual_steps_per_sec"] = _read_steps_per_sec(Path(artifacts.summary_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Benchmark written to {args.output}")


def _versions() -> dict[str, str]:
    packages = ["easylora", "torch", "transformers", "peft", "datasets", "accelerate"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except Exception:
            versions[package] = "unknown"
    return versions


def _read_steps_per_sec(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("train_steps_per_second", "steps_per_second", "steps_per_sec"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


if __name__ == "__main__":
    main()
