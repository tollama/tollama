"""Error types for TiDE runner adapter."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional runner dependencies are unavailable."""


class UnsupportedModelError(ValueError):
    """Raised when a model cannot be mapped to this runner."""


class AdapterInputError(ValueError):
    """Raised when canonical request data cannot be transformed for inference."""
