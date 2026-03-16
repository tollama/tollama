"""Shared Timer-runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional Timer runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported Timer model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the Timer adapter."""
