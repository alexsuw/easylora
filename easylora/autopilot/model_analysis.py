"""Model config inspection for autopilot decisions."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from transformers import AutoConfig


@dataclass
class ModelProfile:
    """Model metadata used by autopilot heuristics."""

    model_name: str
    architecture: str
    model_type: str
    hidden_size: int | None
    num_hidden_layers: int | None
    context_length: int | None
    estimated_params_b: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_model(model: str, *, trust_remote_code: bool = False) -> ModelProfile:
    """Inspect HF config and return compact model profile."""
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    archs = getattr(cfg, "architectures", None)
    arch = archs[0] if isinstance(archs, list) and archs else "unknown"
    model_type = getattr(cfg, "model_type", "unknown")
    hidden_size = _pick_int(cfg, "hidden_size", "d_model", "n_embd")
    num_hidden_layers = _pick_int(cfg, "num_hidden_layers", "n_layer")
    context_length = _pick_int(
        cfg,
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
    )
    estimated_params_b = estimate_params_billion(model, hidden_size, num_hidden_layers)
    return ModelProfile(
        model_name=model,
        architecture=arch,
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        context_length=context_length,
        estimated_params_b=estimated_params_b,
    )


def estimate_params_billion(
    model_name: str,
    hidden_size: int | None,
    num_hidden_layers: int | None,
) -> float | None:
    """Estimate model size in billions using name hints and fallback heuristic."""
    name_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    if name_match:
        return float(name_match.group(1))

    million_match = re.search(r"(\d+(?:\.\d+)?)\s*[mM]\b", model_name)
    if million_match:
        return round(float(million_match.group(1)) / 1000, 3)

    if hidden_size and num_hidden_layers:
        # Rough decoder-only estimate: ~12 * hidden^2 * layers parameters.
        params = 12 * (hidden_size**2) * num_hidden_layers
        return round(params / 1_000_000_000, 3)
    return None


def _pick_int(cfg: object, *names: str) -> int | None:
    for name in names:
        value = getattr(cfg, name, None)
        if isinstance(value, int):
            return value
    return None
