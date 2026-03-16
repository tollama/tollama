"""Shared ForecastPFN-runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional ForecastPFN runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported ForecastPFN model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the ForecastPFN adapter."""
