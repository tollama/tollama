"""Error types for TiDE runner adapter."""

from __future__ import annotations

from tollama.core.errors import (
    RunnerAdapterInputError as AdapterInputError,
)
from tollama.core.errors import (
    RunnerDependencyMissingError as DependencyMissingError,
)
from tollama.core.errors import (
    RunnerUnsupportedModelError as UnsupportedModelError,
)

__all__ = ["AdapterInputError", "DependencyMissingError", "UnsupportedModelError"]
