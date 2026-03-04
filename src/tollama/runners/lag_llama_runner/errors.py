"""Shared Lag-Llama runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional Lag-Llama runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported Lag-Llama model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the Lag-Llama adapter."""
