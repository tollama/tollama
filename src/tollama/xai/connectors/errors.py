"""
tollama.xai.connectors.errors — Formal error taxonomy for connector failures.

This module defines the canonical error type constants used across all
connectors.  Each constant maps to a specific HTTP status range or failure
mode and carries a ``retryable`` flag that the TrustRouter uses to decide
fallback policy.

Error classification table:

    Error Type     | HTTP Status  | Retryable | Router Action
    ─────────────────────────────────────────────────────────────
    auth           | 401, 403     | No        | Degraded result, no fallback
    schema         | 422          | No        | Degraded result, no fallback
    not_found      | 404          | No        | Degraded result, no fallback
    rate_limit     | 429          | Yes       | Retry → heuristic fallback
    internal       | 5xx          | Yes       | Retry → heuristic fallback
    network        | timeout/conn | Yes       | Retry → heuristic fallback
    unknown        | other        | No        | Degraded result
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ErrorCategory:
    """Immutable descriptor for an error classification."""

    error_type: str
    retryable: bool
    description: str


# ── canonical categories ─────────────────────────────────────

AUTH = ErrorCategory(
    error_type="auth",
    retryable=False,
    description="Authentication or authorisation failure (401/403).",
)

SCHEMA = ErrorCategory(
    error_type="schema",
    retryable=False,
    description="Request schema mismatch or validation error (422).",
)

NOT_FOUND = ErrorCategory(
    error_type="not_found",
    retryable=False,
    description="Requested resource does not exist (404).",
)

RATE_LIMIT = ErrorCategory(
    error_type="rate_limit",
    retryable=True,
    description="Rate limit exceeded — retry after backoff (429).",
)

INTERNAL = ErrorCategory(
    error_type="internal",
    retryable=True,
    description="Remote server error — likely transient (5xx).",
)

NETWORK = ErrorCategory(
    error_type="network",
    retryable=True,
    description="Network-level failure (timeout, connection refused).",
)

UNKNOWN = ErrorCategory(
    error_type="unknown",
    retryable=False,
    description="Unclassified error.",
)

# ── lookup helper ────────────────────────────────────────────

_ALL_CATEGORIES: dict[str, ErrorCategory] = {
    c.error_type: c
    for c in [AUTH, SCHEMA, NOT_FOUND, RATE_LIMIT, INTERNAL, NETWORK, UNKNOWN]
}

ALL_ERROR_TYPES: frozenset[str] = frozenset(_ALL_CATEGORIES)


def classify(error_type: str) -> ErrorCategory:
    """Return the ``ErrorCategory`` for a given error_type string.

    Falls back to ``UNKNOWN`` for unrecognised types.
    """
    return _ALL_CATEGORIES.get(error_type, UNKNOWN)


def is_retryable(error_type: str) -> bool:
    """Convenience: check if an error_type is retryable."""
    return classify(error_type).retryable


__all__ = [
    "AUTH",
    "ALL_ERROR_TYPES",
    "ErrorCategory",
    "INTERNAL",
    "NETWORK",
    "NOT_FOUND",
    "RATE_LIMIT",
    "SCHEMA",
    "UNKNOWN",
    "classify",
    "is_retryable",
]
