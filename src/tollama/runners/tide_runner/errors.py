"""Error types for TiDE runner adapter."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional TiDE dependencies are not installed."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported model name."""


class AdapterInputError(ValueError):
    """Raised when canonical request payload cannot be mapped to TiDE runtime inputs."""
