"""Shared torch-runner adapter error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional torch runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported torch runner model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for one adapter implementation."""
