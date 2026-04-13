"""Shared N-HiTS runner error types."""

from __future__ import annotations

from tollama.core.errors import (
    RunnerAdapterInputError as AdapterInputError,
)
from tollama.core.errors import (
    RunnerAdapterRuntimeError as AdapterRuntimeError,
)
from tollama.core.errors import (
    RunnerDependencyMissingError as DependencyMissingError,
)
from tollama.core.errors import (
    RunnerNotImplementedError as NotImplementedRunnerError,
)
from tollama.core.errors import (
    RunnerUnsupportedModelError as UnsupportedModelError,
)

__all__ = [
    "AdapterInputError",
    "AdapterRuntimeError",
    "DependencyMissingError",
    "NotImplementedRunnerError",
    "UnsupportedModelError",
]
