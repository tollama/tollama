"""Shared N-BEATSx runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional N-BEATSx dependencies are missing."""


class NotImplementedRunnerError(RuntimeError):
    """Raised when placeholder paths are reached before full N-BEATSx support."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the N-BEATSx adapter."""
