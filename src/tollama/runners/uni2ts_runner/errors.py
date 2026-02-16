"""Shared Uni2TS-runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional Uni2TS runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported Uni2TS model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the Uni2TS adapter."""
