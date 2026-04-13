"""Shared error hierarchy for tollama components."""

from __future__ import annotations


class TollamaError(Exception):
    """Base error for tollama-specific failures."""


class TollamaRuntimeError(TollamaError, RuntimeError):
    """Base class for runtime failures."""


class TollamaValueError(TollamaError, ValueError):
    """Base class for validation and shape failures."""


class RunnerDependencyMissingError(TollamaRuntimeError):
    """Raised when optional runner dependencies are unavailable."""


class RunnerUnsupportedModelError(TollamaValueError):
    """Raised when model metadata cannot be served by the selected runner."""


class RunnerAdapterInputError(TollamaValueError):
    """Raised when canonical request data cannot be transformed for inference."""


class RunnerAdapterRuntimeError(TollamaRuntimeError):
    """Raised when a runner fails after request validation succeeds."""


class RunnerNotImplementedError(TollamaRuntimeError):
    """Raised when placeholder runner code paths are invoked."""


__all__ = [
    "RunnerAdapterInputError",
    "RunnerAdapterRuntimeError",
    "RunnerDependencyMissingError",
    "RunnerNotImplementedError",
    "RunnerUnsupportedModelError",
    "TollamaError",
    "TollamaRuntimeError",
    "TollamaValueError",
]
