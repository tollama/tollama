"""Shared TimeMixer-runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional TimeMixer runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported TimeMixer model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the TimeMixer adapter."""
