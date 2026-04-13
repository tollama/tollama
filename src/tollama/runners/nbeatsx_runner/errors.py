"""Shared N-BEATSx runner error types."""

from __future__ import annotations

from tollama.core.errors import (
    RunnerAdapterInputError as AdapterInputError,
    RunnerDependencyMissingError as DependencyMissingError,
    RunnerNotImplementedError as NotImplementedRunnerError,
    RunnerUnsupportedModelError as UnsupportedModelError,
)

__all__ = [
    "AdapterInputError",
    "DependencyMissingError",
    "NotImplementedRunnerError",
    "UnsupportedModelError",
]
