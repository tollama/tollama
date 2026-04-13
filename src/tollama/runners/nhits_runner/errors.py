"""Shared N-HiTS runner error types."""

from __future__ import annotations

from tollama.core.errors import (
    RunnerAdapterInputError as AdapterInputError,
    RunnerAdapterRuntimeError as AdapterRuntimeError,
    RunnerDependencyMissingError as DependencyMissingError,
    RunnerNotImplementedError as NotImplementedRunnerError,
    RunnerUnsupportedModelError as UnsupportedModelError,
)

__all__ = [
    "AdapterInputError",
    "AdapterRuntimeError",
    "DependencyMissingError",
    "NotImplementedRunnerError",
    "UnsupportedModelError",
]
