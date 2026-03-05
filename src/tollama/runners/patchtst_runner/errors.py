"""Error types for PatchTST runner adapter."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional PatchTST runner dependencies are missing."""


class UnsupportedModelError(RuntimeError):
    """Raised when model metadata is insufficient for PatchTST loading."""


class AdapterInputError(ValueError):
    """Raised when canonical request payload cannot be mapped to PatchTST input."""
