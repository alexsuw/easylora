"""Custom exceptions for easylora with actionable error messages."""

from __future__ import annotations


class EasyLoRAError(Exception):
    """Base exception for all easylora errors."""


class EasyLoRAConfigError(EasyLoRAError):
    """Raised when a configuration is invalid or incomplete."""


class EasyLoRADependencyError(EasyLoRAError):
    """Raised when a required optional dependency is missing.

    Provides install instructions in the error message.
    """

    def __init__(self, package: str, feature: str, install_extra: str | None = None) -> None:
        install_cmd = (
            f"pip install easylora[{install_extra}]" if install_extra else f"pip install {package}"
        )
        super().__init__(
            f"'{package}' is required for {feature} but is not installed. "
            f"Install it with: {install_cmd}"
        )


class EasyLoRATrainingError(EasyLoRAError):
    """Raised when an error occurs during training."""


class EasyLoRAIOError(EasyLoRAError):
    """Raised for file I/O errors (e.g., output directory conflicts)."""
