"""Hardware/environment detection for autopilot planning."""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass

import torch


@dataclass
class HardwareProfile:
    """Runtime hardware profile used by autopilot heuristics."""

    python_version: str
    platform: str
    torch_version: str
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    bf16_supported: bool
    fp16_supported: bool
    mps_available: bool
    bitsandbytes_available: bool

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareProfile:
    """Collect a concise hardware profile for planning decisions."""
    cuda_available = torch.cuda.is_available()
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    bf16_supported = False
    fp16_supported = False
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    )

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gpu_vram_gb = round(props.total_memory / (1024**3), 2)
        bf16_supported = torch.cuda.is_bf16_supported()
        fp16_supported = True

    try:
        import bitsandbytes  # noqa: F401

        bitsandbytes_available = True
    except Exception:
        bitsandbytes_available = False

    vi = sys.version_info
    return HardwareProfile(
        python_version=f"{vi.major}.{vi.minor}.{vi.micro}",
        platform=platform.platform(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
        mps_available=mps_available,
        bitsandbytes_available=bitsandbytes_available,
    )
