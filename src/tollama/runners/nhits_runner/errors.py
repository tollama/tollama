"""Shared N-HiTS runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional N-HiTS dependencies are missing."""


class NotImplementedRunnerError(RuntimeError):
    """Raised when placeholder paths are reached before full N-HiTS support."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the N-HiTS adapter."""


class UnsupportedModelError(ValueError):
    """Raised when model metadata/source is incompatible with N-HiTS adapter."""
