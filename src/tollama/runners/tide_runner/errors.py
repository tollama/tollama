"""Error types for TiDE runner adapter."""

from __future__ import annotations

from tollama.core.errors import (
    RunnerAdapterInputError as AdapterInputError,
    RunnerDependencyMissingError as DependencyMissingError,
    RunnerUnsupportedModelError as UnsupportedModelError,
)

__all__ = ["AdapterInputError", "DependencyMissingError", "UnsupportedModelError"]
