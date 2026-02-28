"""Heuristic rules that resolve an autopilot training strategy."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from easylora.autopilot.dataset_analysis import DatasetProfile
from easylora.autopilot.hardware import HardwareProfile
from easylora.autopilot.model_analysis import ModelProfile
from easylora.autopilot.presets import QualityPreset


@dataclass
class PlanDecision:
    use_qlora: bool
    max_seq_len: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    grad_accum: int
    epochs: int
    max_steps: int
    save_steps: int
    logging_steps: int
    estimated_vram_gb: float
    estimated_steps_per_sec: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_plan(
    hardware: HardwareProfile,
    dataset: DatasetProfile,
    model: ModelProfile,
    preset: QualityPreset,
) -> PlanDecision:
    """Resolve concrete hyperparameters from profiles + selected quality preset."""
    reasons: list[str] = []
    params_b = model.estimated_params_b or 1.0
    vram = hardware.gpu_vram_gb or 0.0

    use_qlora = (
        (params_b >= 4.0 and hardware.cuda_available)
        or (vram > 0 and vram <= 16)
        or not hardware.cuda_available
    )
    if use_qlora:
        reasons.append("Selected QLoRA for memory safety on current hardware/model scale.")
        if not hardware.bitsandbytes_available:
            reasons.append(
                "bitsandbytes was not detected; training may fail until it is installed."
            )
    else:
        reasons.append("Selected LoRA because hardware has enough memory for full-precision base load.")

    max_seq_len = _resolve_seq_len(dataset.p95_tokens, model.context_length)
    reasons.append(f"Set max_seq_len={max_seq_len} from dataset p95 token length={dataset.p95_tokens}.")

    lora_r = _resolve_lora_rank(preset.base_lora_rank, params_b)
    lora_alpha = lora_r * 2
    learning_rate = _resolve_learning_rate(preset.base_lr, use_qlora, params_b)
    reasons.append(f"Set LoRA rank/alpha to r={lora_r}, alpha={lora_alpha}.")
    reasons.append(f"Set learning rate to {learning_rate:.2e}.")

    batch_size, grad_accum = _resolve_batching(
        vram_gb=vram,
        seq_len=max_seq_len,
        params_b=params_b,
        target_effective_batch=preset.target_effective_batch,
    )
    reasons.append(
        f"Set micro-batch={batch_size} and grad_accum={grad_accum} "
        f"(effective batch={batch_size * grad_accum})."
    )

    epochs, max_steps = _resolve_duration(dataset.examples, preset)
    if max_steps > 0:
        reasons.append(f"Large dataset detected; capped run by max_steps={max_steps}.")
    else:
        reasons.append(f"Set epochs={epochs} from quality preset '{preset.name}'.")

    estimated_vram = _estimate_vram_gb(use_qlora, params_b, max_seq_len, batch_size)
    estimated_steps = _estimate_steps_per_second(params_b, max_seq_len, batch_size, use_qlora)
    save_steps = _resolve_save_steps(dataset.examples, batch_size, grad_accum)

    return PlanDecision(
        use_qlora=use_qlora,
        max_seq_len=max_seq_len,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_accum=grad_accum,
        epochs=epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        logging_steps=10,
        estimated_vram_gb=estimated_vram,
        estimated_steps_per_sec=estimated_steps,
        reasons=reasons,
    )


def _resolve_seq_len(p95: int, context_limit: int | None) -> int:
    buckets = [256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
    target = max(256, min(8192, p95))
    chosen = next((b for b in buckets if b >= target), 8192)
    if context_limit:
        chosen = min(chosen, context_limit)
    return max(256, chosen)


def _resolve_lora_rank(base_rank: int, params_b: float) -> int:
    if params_b >= 10:
        return max(8, base_rank // 2)
    if params_b >= 4:
        return base_rank
    if params_b < 1:
        return base_rank * 2
    return base_rank


def _resolve_learning_rate(base_lr: float, use_qlora: bool, params_b: float) -> float:
    lr = base_lr
    if use_qlora:
        lr *= 1.1
    if params_b >= 10:
        lr *= 0.7
    if params_b < 1:
        lr *= 1.2
    return float(f"{lr:.2e}")


def _resolve_batching(
    *,
    vram_gb: float,
    seq_len: int,
    params_b: float,
    target_effective_batch: int,
) -> tuple[int, int]:
    if vram_gb <= 0:
        micro = 1
    elif vram_gb <= 10:
        micro = 1
    elif vram_gb <= 16:
        micro = 2
    elif vram_gb <= 24:
        micro = 4
    else:
        micro = 8

    if seq_len > 2048:
        micro = max(1, micro // 2)
    if params_b > 10:
        micro = max(1, micro // 2)

    grad_accum = max(1, math.ceil(target_effective_batch / micro))
    return micro, grad_accum


def _resolve_duration(dataset_size: int, preset: QualityPreset) -> tuple[int, int]:
    if dataset_size > 100_000:
        return 1, preset.max_steps_large_dataset
    if dataset_size > 30_000:
        return 1, int(preset.max_steps_large_dataset * 0.7)
    return preset.default_epochs, -1


def _resolve_save_steps(dataset_size: int, batch_size: int, grad_accum: int) -> int:
    effective = max(1, batch_size * grad_accum)
    approx_steps = max(1, dataset_size // effective)
    return max(100, min(1000, approx_steps // 5))


def _estimate_vram_gb(use_qlora: bool, params_b: float, seq_len: int, batch_size: int) -> float:
    base = max(2.0, params_b * (1.6 if use_qlora else 3.8))
    activation = (seq_len / 1024) * batch_size * 0.9
    return round(base + activation, 1)


def _estimate_steps_per_second(
    params_b: float,
    seq_len: int,
    batch_size: int,
    use_qlora: bool,
) -> float:
    throughput = 8.0 / max(0.2, params_b)
    throughput *= 1024 / max(256, seq_len)
    throughput *= batch_size
    if use_qlora:
        throughput *= 0.85
    return round(max(0.05, throughput), 2)
