"""Shared TimesFM-runner error types."""

from __future__ import annotations


class DependencyMissingError(RuntimeError):
    """Raised when optional TimesFM runner dependencies are missing."""


class UnsupportedModelError(ValueError):
    """Raised when a request targets an unsupported TimesFM model."""


class AdapterInputError(ValueError):
    """Raised when request input is invalid for the TimesFM adapter."""
