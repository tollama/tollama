"""Tests for Phase 8 features — async trust pipeline, caching, history,
record-outcome endpoint, connector health.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from tollama.xai.trust_history import (
    TrustHistoryTracker,
    default_history_path,
)
from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

# ── Fake agent for tests ─────────────────────────────────────────


class _FakeTrustAgent:
    agent_name = "fake_agent"
    domain = "test_domain"
    priority = 10
    call_count = 0

    def supports(self, context: dict[str, Any]) -> bool:
        return context.get("domain") == "test_domain"

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {
            "agent_name": self.agent_name,
            "domain": self.domain,
            "trust_score": 0.85,
            "risk_category": "GREEN",
            "calibration_status": "high_trust",
            "component_breakdown": {
                "accuracy": {
                    "score": 0.9,
                    "weight": 0.5,
                    "label": "Accuracy",
                    "description": "Test accuracy",
                },
            },
            "violations": [],
            "why_trusted": "Test agent",
            "evidence": {
                "source_type": "test",
                "source_ids": ["test-1"],
                "payload_schema": "test_v1",
                "attributes": {},
            },
            "audit": {
                "formula_version": "test-v1",
                "agent_version": "0.1.0",
            },
        }


def _make_router_with_cache(ttl: float = 60.0) -> tuple[TrustRouter, _FakeTrustAgent]:
    registry = TrustAgentRegistry()
    agent = _FakeTrustAgent()
    registry.register(agent)
    router = TrustRouter(registry, cache_ttl=ttl)
    return router, agent


# ── Trust Score Caching ──────────────────────────────────────────


class TestTrustScoreCaching:
    def test_cache_hit_avoids_recompute(self):
        router, agent = _make_router_with_cache(ttl=60.0)
        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}

        result1 = router.analyze(context=ctx, payload=payload)
        result2 = router.analyze(context=ctx, payload=payload)

        assert result1 is not None
        assert result2 is not None
        assert result1.trust_score == result2.trust_score
        assert agent.call_count == 1  # Only one actual call

    def test_different_payload_no_cache_hit(self):
        router, agent = _make_router_with_cache(ttl=60.0)
        ctx = {"domain": "test_domain"}

        router.analyze(context=ctx, payload={"a": 1})
        router.analyze(context=ctx, payload={"b": 2})

        assert agent.call_count == 2

    def test_cache_disabled_when_ttl_zero(self):
        registry = TrustAgentRegistry()
        agent = _FakeTrustAgent()
        registry.register(agent)
        router = TrustRouter(registry, cache_ttl=0.0)

        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}
        router.analyze(context=ctx, payload=payload)
        router.analyze(context=ctx, payload=payload)

        assert agent.call_count == 2

    def test_cache_expires_after_ttl(self):
        router, agent = _make_router_with_cache(ttl=0.05)
        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}

        router.analyze(context=ctx, payload=payload)
        time.sleep(0.06)
        router.analyze(context=ctx, payload=payload)

        assert agent.call_count == 2

    def test_clear_cache(self):
        router, agent = _make_router_with_cache(ttl=60.0)
        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}

        router.analyze(context=ctx, payload=payload)
        cleared = router.clear_cache()
        assert cleared == 1

        router.analyze(context=ctx, payload=payload)
        assert agent.call_count == 2

    def test_cache_key_deterministic(self):
        key1 = TrustRouter._cache_key({"a": 1, "b": 2}, {"x": 10})
        key2 = TrustRouter._cache_key({"b": 2, "a": 1}, {"x": 10})
        assert key1 == key2

    def test_no_agent_returns_none_not_cached(self):
        registry = TrustAgentRegistry()
        router = TrustRouter(registry, cache_ttl=60.0)
        result = router.analyze(context={"domain": "unknown"}, payload={})
        assert result is None
        assert len(router._cache) == 0


# ── Async Trust Pipeline ────────────────────────────────────────


class TestAsyncTrustPipeline:
    def test_analyze_async(self):
        router, agent = _make_router_with_cache(ttl=0.0)
        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}

        result = asyncio.run(router.analyze_async(context=ctx, payload=payload))
        assert result is not None
        assert result.trust_score == 0.85
        assert agent.call_count == 1

    def test_analyze_multi_async(self):
        router, agent = _make_router_with_cache(ttl=0.0)
        ctx = {"domain": "test_domain"}
        payload = {"key": "value"}

        result = asyncio.run(router.analyze_multi_async(context=ctx, payload=payload))
        assert result is not None
        assert result.trust_score == 0.85

    def test_analyze_async_no_agent(self):
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)

        result = asyncio.run(router.analyze_async(context={"domain": "missing"}, payload={}))
        assert result is None


# ── Trust History Tracker ────────────────────────────────────────


class TestTrustHistoryTracker:
    def test_record_and_get_history(self):
        tracker = TrustHistoryTracker()
        tracker.record("agent_a", "financial_market", 0.8, "GREEN")
        tracker.record("agent_a", "financial_market", 0.75, "YELLOW")

        history = tracker.get_history("financial_market")
        assert len(history) == 2
        assert history[0].trust_score == 0.8
        assert history[1].trust_score == 0.75

    def test_domains_property(self):
        tracker = TrustHistoryTracker()
        tracker.record("a", "news", 0.7, "GREEN")
        tracker.record("b", "financial_market", 0.8, "GREEN")

        assert tracker.domains == ["financial_market", "news"]

    def test_get_stats(self):
        tracker = TrustHistoryTracker()
        for score in [0.6, 0.7, 0.8, 0.9]:
            tracker.record("agent", "test", score, "GREEN")

        stats = tracker.get_stats("test")
        assert stats.record_count == 4
        assert stats.mean_score == pytest.approx(0.75)
        assert stats.min_score == 0.6
        assert stats.max_score == 0.9
        assert stats.latest_score == 0.9

    def test_get_stats_empty(self):
        tracker = TrustHistoryTracker()
        stats = tracker.get_stats("missing")
        assert stats.record_count == 0
        assert stats.trend == "stable"

    def test_trend_improving(self):
        tracker = TrustHistoryTracker()
        # Older scores low, recent scores high
        for score in [0.3, 0.3, 0.3, 0.5, 0.7, 0.8]:
            tracker.record("agent", "test", score, "GREEN")

        stats = tracker.get_stats("test")
        assert stats.trend == "improving"

    def test_trend_declining(self):
        tracker = TrustHistoryTracker()
        for score in [0.8, 0.8, 0.8, 0.5, 0.3, 0.2]:
            tracker.record("agent", "test", score, "RED")

        stats = tracker.get_stats("test")
        assert stats.trend == "declining"

    def test_trend_stable(self):
        tracker = TrustHistoryTracker()
        for score in [0.7, 0.71, 0.69, 0.7, 0.71, 0.7]:
            tracker.record("agent", "test", score, "GREEN")

        stats = tracker.get_stats("test")
        assert stats.trend == "stable"

    def test_limit(self):
        tracker = TrustHistoryTracker()
        for i in range(20):
            tracker.record("agent", "test", 0.5, "GREEN")

        history = tracker.get_history("test", limit=5)
        assert len(history) == 5

    def test_max_per_domain(self):
        tracker = TrustHistoryTracker(max_per_domain=5)
        for i in range(10):
            tracker.record("agent", "test", 0.5 + i * 0.01, "GREEN")

        history = tracker.get_history("test")
        assert len(history) == 5

    def test_save_and_load(self, tmp_path: Path):
        tracker = TrustHistoryTracker()
        tracker.record("agent_a", "news", 0.8, "GREEN")
        tracker.record("agent_b", "financial_market", 0.6, "YELLOW")

        path = tmp_path / "history.json"
        tracker.save(path)
        assert path.is_file()

        loaded = TrustHistoryTracker.load(path)
        assert loaded.domains == ["financial_market", "news"]
        assert len(loaded.get_history("news")) == 1
        assert len(loaded.get_history("financial_market")) == 1

    def test_load_missing_file(self, tmp_path: Path):
        tracker = TrustHistoryTracker.load(tmp_path / "nonexistent.json")
        assert tracker.domains == []

    def test_default_history_path(self):
        path = default_history_path()
        assert str(path).endswith("trust_history.json")
        assert "xai" in str(path)

    def test_record_with_metadata(self):
        tracker = TrustHistoryTracker()
        tracker.record(
            "agent",
            "test",
            0.8,
            "GREEN",
            metadata={"request_id": "abc-123"},
        )
        history = tracker.get_history("test")
        assert history[0].metadata == {"request_id": "abc-123"}
