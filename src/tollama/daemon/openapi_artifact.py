"""Helpers for generating stable OpenAPI artifacts."""

from __future__ import annotations

import json
from typing import Any


def canonicalize_openapi_schema(value: Any) -> Any:
    """Return a deterministic representation for artifact export and diffing."""
    if isinstance(value, dict):
        return {key: canonicalize_openapi_schema(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        canonical_items = [canonicalize_openapi_schema(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return value
