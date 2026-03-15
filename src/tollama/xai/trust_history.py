"""
tollama.xai.trust_history — Persistent trust score history for trend analysis.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_log = logging.getLogger(__name__)

_MAX_HISTORY_PER_DOMAIN = 500


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class TrustHistoryRecord(BaseModel):
    """A single trust analysis result snapshot."""

    agent_name: str
    domain: str
    trust_score: float = Field(ge=0.0, le=1.0)
    risk_category: str
    calibration_status: str | None = None
    recorded_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrustHistoryStats(BaseModel):
    """Aggregated stats for a domain's trust history."""

    domain: str
    record_count: int
    mean_score: float
    min_score: float
    max_score: float
    latest_score: float
    latest_risk: str
    trend: str  # "improving", "declining", "stable"


class TrustHistoryTracker:
    """In-memory + persistent tracker for trust score history."""

    def __init__(self, max_per_domain: int = _MAX_HISTORY_PER_DOMAIN):
        self._max = max_per_domain
        self._history: dict[str, deque[TrustHistoryRecord]] = {}

    def record(
        self,
        agent_name: str,
        domain: str,
        trust_score: float,
        risk_category: str,
        calibration_status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if domain not in self._history:
            self._history[domain] = deque(maxlen=self._max)
        self._history[domain].append(
            TrustHistoryRecord(
                agent_name=agent_name,
                domain=domain,
                trust_score=trust_score,
                risk_category=risk_category,
                calibration_status=calibration_status,
                metadata=metadata or {},
            )
        )

    @property
    def domains(self) -> list[str]:
        return sorted(self._history.keys())

    def get_history(
        self,
        domain: str,
        limit: int = 50,
    ) -> list[TrustHistoryRecord]:
        records = self._history.get(domain)
        if not records:
            return []
        return list(records)[-limit:]

    def get_stats(self, domain: str) -> TrustHistoryStats:
        records = self._history.get(domain)
        if not records:
            return TrustHistoryStats(
                domain=domain,
                record_count=0,
                mean_score=0.0,
                min_score=0.0,
                max_score=0.0,
                latest_score=0.0,
                latest_risk="GREEN",
                trend="stable",
            )
        scores = [r.trust_score for r in records]
        latest = records[-1]
        trend = self._compute_trend(scores)
        return TrustHistoryStats(
            domain=domain,
            record_count=len(records),
            mean_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
            latest_score=latest.trust_score,
            latest_risk=latest.risk_category,
            trend=trend,
        )

    @staticmethod
    def _compute_trend(scores: list[float]) -> str:
        if len(scores) < 3:
            return "stable"
        recent = scores[-3:]
        older = scores[-6:-3] if len(scores) >= 6 else scores[:3]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        diff = recent_avg - older_avg
        if diff > 0.05:
            return "improving"
        if diff < -0.05:
            return "declining"
        return "stable"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, list[dict[str, Any]]] = {}
        for domain, records in self._history.items():
            data[domain] = [r.model_dump(mode="json") for r in records]
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    @classmethod
    def load(
        cls, path: Path, max_per_domain: int = _MAX_HISTORY_PER_DOMAIN
    ) -> TrustHistoryTracker:
        tracker = cls(max_per_domain=max_per_domain)
        if not path.is_file():
            return tracker
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            _log.warning("Failed to load trust history from %s", path, exc_info=True)
            return tracker
        for domain, records in raw.items():
            dq: deque[TrustHistoryRecord] = deque(maxlen=max_per_domain)
            for rec in records[-max_per_domain:]:
                dq.append(TrustHistoryRecord(**rec))
            tracker._history[domain] = dq
        return tracker


def default_history_path() -> Path:
    from tollama.core.storage import TollamaPaths

    return TollamaPaths.default().base_dir / "xai" / "trust_history.json"


__all__ = [
    "TrustHistoryRecord",
    "TrustHistoryStats",
    "TrustHistoryTracker",
    "default_history_path",
]
