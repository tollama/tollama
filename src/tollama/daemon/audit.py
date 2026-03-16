"""Audit logging for security-sensitive daemon operations.

Records pull, unload, config-change, and gate-decision actions to a
dedicated ``tollama.audit`` logger. The structured JSON formatter from
:mod:`tollama.daemon.logging_config` automatically includes the request-ID
when available.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from .logging_config import get_request_id

_audit_logger = logging.getLogger("tollama.audit")


def _base_entry(*, action: str, **kwargs: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "action": action,
        "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    request_id = get_request_id()
    if request_id is not None:
        entry["request_id"] = request_id
    entry.update({k: v for k, v in kwargs.items() if v is not None})
    return entry


def audit_pull(*, model: str, result: str, source_ip: str | None = None) -> None:
    """Log a model pull operation."""
    _audit_logger.info(
        "audit: pull model=%s result=%s",
        model,
        result,
        extra=_base_entry(action="pull", model=model, result=result, source_ip=source_ip),
    )


def audit_unload(*, model: str, family: str, source_ip: str | None = None) -> None:
    """Log a model unload operation."""
    _audit_logger.info(
        "audit: unload model=%s family=%s",
        model,
        family,
        extra=_base_entry(action="unload", model=model, family=family, source_ip=source_ip),
    )


def audit_config_change(
    *, key: str, old_value: Any = None, new_value: Any = None, source_ip: str | None = None,
) -> None:
    """Log a configuration change."""
    _audit_logger.info(
        "audit: config_change key=%s",
        key,
        extra=_base_entry(
            action="config_change",
            key=key,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            source_ip=source_ip,
        ),
    )


def audit_gate_decision(
    *, model: str, decision: str, trust_score: float | None = None,
    source_ip: str | None = None,
) -> None:
    """Log a trust-gate decision."""
    _audit_logger.info(
        "audit: gate_decision model=%s decision=%s trust_score=%s",
        model,
        decision,
        trust_score,
        extra=_base_entry(
            action="gate_decision",
            model=model,
            decision=decision,
            trust_score=trust_score,
            source_ip=source_ip,
        ),
    )
