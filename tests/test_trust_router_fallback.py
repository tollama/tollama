"""Tests for TrustRouter fallback and degraded result logic.

Validates Phase 2.3: router fallback policy when remote agents fail.
"""

from __future__ import annotations

from typing import Any

import pytest

from tollama.xai.connectors.protocol import (
    ConnectorError,
    ConnectorFetchError,
)
from tollama.xai.trust_contract import NormalizedTrustResult
from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

# ── stub agents ─────────────────────────────────────────────────────


class _StubAgent:
    """A simple trust agent that returns a fixed result."""

    def __init__(
        self,
        agent_name: str,
        domain: str,
        priority: int = 50,
        trust_score: float = 0.75,
    ) -> None:
        self.agent_name = agent_name
        self.domain = domain
        self.priority = priority
        self._trust_score = trust_score

    def supports(self, context: dict[str, Any]) -> bool:
        return context.get("domain") == self.domain

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "domain": self.domain,
            "trust_score": self._trust_score,
            "risk_category": "GREEN" if self._trust_score >= 0.5 else "RED",
            "calibration_status": "well_calibrated",
            "component_breakdown": {},
            "violations": [],
            "why_trusted": f"Stub agent {self.agent_name}",
            "evidence": {
                "source_type": "stub",
                "source_ids": [],
            },
            "audit": {
                "formula_version": "stub-v1",
                "agent_version": "0.1.0",
            },
        }


class _FailingAgent:
    """Agent that raises a ConnectorFetchError on analyze()."""

    def __init__(
        self,
        agent_name: str,
        domain: str,
        error_type: str = "network",
        retryable: bool = True,
        priority: int = 10,
    ) -> None:
        self.agent_name = agent_name
        self.domain = domain
        self.priority = priority
        self._error_type = error_type
        self._retryable = retryable

    def supports(self, context: dict[str, Any]) -> bool:
        return context.get("domain") == self.domain

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectorFetchError(
            ConnectorError(
                domain=self.domain,
                error_type=self._error_type,
                message=f"Simulated {self._error_type} error",
                retryable=self._retryable,
            )
        )


class _TimeoutAgent:
    """Agent that raises a TimeoutError on analyze()."""

    def __init__(
        self,
        agent_name: str,
        domain: str,
        priority: int = 10,
    ) -> None:
        self.agent_name = agent_name
        self.domain = domain
        self.priority = priority

    def supports(self, context: dict[str, Any]) -> bool:
        return context.get("domain") == self.domain

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("Connection timed out")


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def news_ctx() -> dict[str, Any]:
    return {"domain": "news"}


@pytest.fixture()
def fin_ctx() -> dict[str, Any]:
    return {"domain": "financial_market"}


@pytest.fixture()
def payload() -> dict[str, Any]:
    return {"story_id": "test-123", "source_credibility": 0.8}


# ── helpers ─────────────────────────────────────────────────────────


def _fail(name: str, domain: str, etype: str, retry: bool, pri: int = 10):
    return _FailingAgent(name, domain, etype, retry, pri)


def _stub(name: str, domain: str, pri: int = 50, score: float = 0.75):
    return _StubAgent(name, domain, pri, score)


# ── normal operation ────────────────────────────────────────────────


class TestNormalOperation:
    def test_analyze_returns_result(self, news_ctx, payload):
        reg = TrustAgentRegistry()
        reg.register(_stub("news_agent", "news"))
        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)
        assert result is not None
        assert result.trust_score == 0.75
        assert result.risk_category == "GREEN"


# ── retryable failure → heuristic fallback ──────────────────────────


class TestRetryableFallback:
    def test_network_error_falls_back(self, news_ctx, payload):
        """Network error → should fallback to heuristic."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "network", True, 10))
        reg.register(_stub("heuristic", "news", 50, 0.65))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.agent_name == "heuristic"
        assert result.trust_score == 0.65

    def test_timeout_falls_back(self, news_ctx, payload):
        """TimeoutError should also trigger heuristic fallback."""
        reg = TrustAgentRegistry()
        reg.register(_TimeoutAgent("remote", "news", 10))
        reg.register(_stub("heuristic", "news", 50, 0.60))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.agent_name == "heuristic"
        assert result.trust_score == 0.60

    def test_5xx_error_falls_back(self, fin_ctx, payload):
        """Server errors (retryable) should trigger fallback."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "financial_market", "internal", True, 10))
        reg.register(_stub("heuristic", "financial_market", 50, 0.70))

        router = TrustRouter(reg)
        result = router.analyze(context=fin_ctx, payload=payload)

        assert result is not None
        assert result.agent_name == "heuristic"

    def test_rate_limit_falls_back(self, news_ctx, payload):
        """429 rate limit errors should trigger fallback."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "rate_limit", True, 10))
        reg.register(_stub("heuristic", "news", 50))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.agent_name == "heuristic"


# ── non-retryable → degraded result, no fallback ────────────────────


class TestNonRetryableDegraded:
    def test_auth_error_degraded_no_fallback(self, news_ctx, payload):
        """Auth errors → degraded result, NOT fallback."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "auth", False, 10))
        reg.register(_stub("heuristic", "news", 50))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.trust_score == 0.0
        assert result.risk_category == "RED"
        assert result.agent_name == "remote"
        assert len(result.violations) == 1
        assert "agent_failure:auth" in result.violations[0].name
        assert result.violations[0].severity == "critical"

    def test_schema_error_degraded_no_fallback(self, fin_ctx, payload):
        """Schema mismatch (422) → degraded result, NOT fallback."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "financial_market", "schema", False, 10))
        reg.register(_stub("heuristic", "financial_market", 50))

        router = TrustRouter(reg)
        result = router.analyze(context=fin_ctx, payload=payload)

        assert result is not None
        assert result.trust_score == 0.0
        assert result.risk_category == "RED"
        assert "agent_failure:schema" in result.violations[0].name


# ── no fallback available → degraded result ──────────────────────────


class TestNoFallbackAvailable:
    def test_single_agent_fails_degraded(self, news_ctx, payload):
        """Only agent fails, no fallback → degraded result."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "network", True))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.trust_score == 0.0
        assert result.risk_category == "RED"
        assert result.evidence.source_type == "degraded"

    def test_both_agents_fail_degraded(self, news_ctx, payload):
        """Primary and fallback both fail → degraded result."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "network", True, 10))
        reg.register(_fail("fallback", "news", "internal", True, 50))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert result is not None
        assert result.trust_score == 0.0
        assert result.risk_category == "RED"


# ── degraded result structure ────────────────────────────────────────


class TestDegradedResultStructure:
    def test_valid_normalized_result(self, news_ctx, payload):
        """Degraded result should be a valid NormalizedTrustResult."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "auth", False))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert isinstance(result, NormalizedTrustResult)
        assert result.calibration_status == "poorly_calibrated"
        assert result.audit.formula_version == "degraded-v1"
        assert result.evidence.attributes.get("degraded") is True
        assert result.evidence.attributes.get("error_type") == "auth"

    def test_why_trusted_contains_error_info(self, news_ctx, payload):
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "network", True))

        router = TrustRouter(reg)
        result = router.analyze(context=news_ctx, payload=payload)

        assert "remote" in result.why_trusted
        assert "network" in result.why_trusted


# ── cache interaction with fallback ──────────────────────────────────


class TestCacheWithFallback:
    def test_degraded_result_is_cached(self, news_ctx, payload):
        """Degraded results should be cached."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "auth", False))

        router = TrustRouter(reg, cache_ttl=60.0)
        r1 = router.analyze(context=news_ctx, payload=payload)
        r2 = router.analyze(context=news_ctx, payload=payload)

        assert r1 is not None
        assert r2 is not None
        assert r2.trust_score == 0.0
        stats = router.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_fallback_result_is_cached(self, news_ctx, payload):
        """Fallback results should be cached normally."""
        reg = TrustAgentRegistry()
        reg.register(_fail("remote", "news", "network", True, 10))
        reg.register(_stub("heuristic", "news", 50))

        router = TrustRouter(reg, cache_ttl=60.0)
        r1 = router.analyze(context=news_ctx, payload=payload)
        r2 = router.analyze(context=news_ctx, payload=payload)

        assert r1 is not None
        assert r1.agent_name == "heuristic"
        assert r2 is not None
        stats = router.cache_stats()
        assert stats["hits"] == 1
