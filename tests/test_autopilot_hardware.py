"""Tests for autopilot hardware detection."""

from __future__ import annotations

from types import SimpleNamespace

from easylora.autopilot import hardware as hw


def test_detect_hardware_cpu(monkeypatch):
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(hw, "platform", SimpleNamespace(platform=lambda: "test-platform"))
    monkeypatch.setattr(hw.torch, "__version__", "2.0.0")

    profile = hw.detect_hardware()

    assert profile.cuda_available is False
    assert profile.gpu_name is None
    assert profile.gpu_vram_gb is None
    assert profile.fp16_supported is False


def test_detect_hardware_cuda(monkeypatch):
    monkeypatch.setattr(hw.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hw.torch.cuda, "get_device_name", lambda _idx: "TestGPU")
    monkeypatch.setattr(
        hw.torch.cuda,
        "get_device_properties",
        lambda _idx: SimpleNamespace(total_memory=16 * (1024**3)),
    )
    monkeypatch.setattr(hw.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(hw.torch, "__version__", "2.0.0")

    profile = hw.detect_hardware()

    assert profile.cuda_available is True
    assert profile.gpu_name == "TestGPU"
    assert profile.gpu_vram_gb == 16.0
    assert profile.bf16_supported is True
    assert profile.fp16_supported is True
