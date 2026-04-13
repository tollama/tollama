"""Phase 10 tests — async client XAI, history persist, cache TTL, e2e pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tollama.daemon.app import create_app
from tollama.xai.trust_history import TrustHistoryTracker
from tollama.xai.trust_router import TrustAgentRegistry, TrustRouter

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _make_stub_agent(
    name: str,
    domain: str,
    score: float = 0.7,
) -> Any:
    """Create a minimal stub trust agent."""
    agent = MagicMock()
    agent.agent_name = name
    agent.domain = domain
    agent.priority = 10

    def _supports(context: dict[str, Any]) -> bool:
        return context.get("domain") == domain

    agent.supports = _supports
    agent.analyze = MagicMock(
        return_value={
            "agent_name": name,
            "domain": domain,
            "trust_score": score,
            "risk_category": "GREEN",
            "calibration_status": "moderate_trust",
            "component_breakdown": {
                "test_component": {
                    "name": "test_component",
                    "weight": 0.5,
                    "score": score,
                    "details": "stub",
                },
            },
            "violations": [],
            "why_trusted": f"Stub agent {name}",
            "evidence": {
                "source_type": "stub",
                "source_ids": [f"{name}-001"],
                "payload_schema": "stub_v1",
                "attributes": {},
            },
            "audit": {
                "formula_version": "stub-v1",
                "agent_version": "0.1.0",
            },
        }
    )
    return agent


# ──────────────────────────────────────────────────────────────
# AsyncTollamaClient XAI methods
# ──────────────────────────────────────────────────────────────


class TestAsyncClientXAIMethods:
    """Verify AsyncTollamaClient has all XAI methods."""

    def test_has_explain_decision(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "explain_decision")
        assert callable(client.explain_decision)

    def test_has_trust_breakdown(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "trust_breakdown")

    def test_has_model_card(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "model_card")

    def test_has_record_outcome(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "record_outcome")

    def test_has_dashboard_history(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "dashboard_history")

    def test_has_connectors_health(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "connectors_health")

    def test_has_cache_stats(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        client = AsyncTollamaClient()
        assert hasattr(client, "cache_stats")


# ──────────────────────────────────────────────────────────────
# Trust history auto-persist
# ──────────────────────────────────────────────────────────────


class TestHistoryAutoPersist:
    def test_persist_history_saves_to_disk(self, tmp_path: Path) -> None:
        tracker = TrustHistoryTracker()
        tracker.record(
            agent_name="financial_market",
            domain="financial_market",
            trust_score=0.8,
            risk_category="GREEN",
        )
        history_path = tmp_path / "history.json"

        registry = TrustAgentRegistry()
        registry.register(_make_stub_agent("financial_market", "financial_market"))

        router = TrustRouter(
            registry,
            history_tracker=tracker,
            history_path=history_path,
        )
        router.persist_history()

        assert history_path.is_file()
        data = json.loads(history_path.read_text())
        assert "financial_market" in data
        assert len(data["financial_market"]) == 1

    def test_auto_persist_saves_history_after_n_calls(
        self,
        tmp_path: Path,
    ) -> None:
        tracker = TrustHistoryTracker()
        history_path = tmp_path / "history.json"

        registry = TrustAgentRegistry()
        agent = _make_stub_agent("financial_market", "financial_market")
        registry.register(agent)

        router = TrustRouter(
            registry,
            auto_persist_every=2,
            history_tracker=tracker,
            history_path=history_path,
        )

        # First analyze — no persist yet
        router.analyze(
            context={"domain": "financial_market"},
            payload={"instrument_id": "test-1"},
        )
        assert not history_path.is_file()

        # Second analyze — triggers auto-persist
        router.analyze(
            context={"domain": "financial_market"},
            payload={"instrument_id": "test-2"},
        )
        assert history_path.is_file()
        data = json.loads(history_path.read_text())
        assert "financial_market" in data

    def test_persist_history_noop_without_tracker(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        # Should not raise
        router.persist_history()


# ──────────────────────────────────────────────────────────────
# Cache TTL configuration
# ──────────────────────────────────────────────────────────────


class TestSetCacheTTL:
    def test_set_cache_ttl_returns_old_value(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry, cache_ttl=30.0)
        old = router.set_cache_ttl(60.0)
        assert old == 30.0
        assert router._cache_ttl == 60.0

    def test_set_ttl_zero_clears_cache(self) -> None:
        registry = TrustAgentRegistry()
        agent = _make_stub_agent("financial_market", "financial_market")
        registry.register(agent)

        router = TrustRouter(registry, cache_ttl=300.0)
        router.analyze(
            context={"domain": "financial_market"},
            payload={"instrument_id": "x"},
        )
        assert len(router._cache) == 1

        router.set_cache_ttl(0.0)
        assert len(router._cache) == 0

    def test_set_ttl_negative_treated_as_zero(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry, cache_ttl=10.0)
        router.set_cache_ttl(-5.0)
        assert router._cache_ttl == 0.0


class TestCacheTTLEndpoint:
    def test_put_cache_ttl(self, client: TestClient) -> None:
        response = client.put(
            "/api/xai/cache/ttl",
            json={"ttl": 120.0},
        )
        assert response.status_code == 200
        body = response.json()
        assert "previous_ttl" in body
        assert body["new_ttl"] == 120.0

    def test_put_cache_ttl_zero_disables(self, client: TestClient) -> None:
        response = client.put(
            "/api/xai/cache/ttl",
            json={"ttl": 0.0},
        )
        assert response.status_code == 200
        assert response.json()["new_ttl"] == 0.0

    def test_put_cache_ttl_rejects_negative(self, client: TestClient) -> None:
        response = client.put(
            "/api/xai/cache/ttl",
            json={"ttl": -1.0},
        )
        assert response.status_code in (400, 422)


# ──────────────────────────────────────────────────────────────
# End-to-end trust pipeline
# ──────────────────────────────────────────────────────────────


class TestEndToEndTrustPipeline:
    """Single test exercising: analyze → history record → calibration
    record → cache hit on replay."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from tollama.xai.trust_agents.calibration import CalibrationTracker

        # Set up tracker + history + paths
        cal_tracker = CalibrationTracker()
        cal_path = tmp_path / "calibration.json"
        history_tracker = TrustHistoryTracker()
        history_path = tmp_path / "history.json"

        registry = TrustAgentRegistry()
        agent = _make_stub_agent(
            "financial_market",
            "financial_market",
            score=0.75,
        )
        registry.register(agent)

        router = TrustRouter(
            registry,
            calibration_tracker=cal_tracker,
            calibration_path=cal_path,
            history_tracker=history_tracker,
            history_path=history_path,
            auto_persist_every=2,
            cache_ttl=300.0,
        )

        context = {"domain": "financial_market"}
        payload = {"instrument_id": "AAPL"}

        # Step 1: First analyze — cache miss, history recorded
        result = router.analyze(context=context, payload=payload)
        assert result is not None
        assert result.trust_score == 0.75
        assert router._cache_misses == 1
        assert router._cache_hits == 0

        # Verify history was recorded
        history = history_tracker.get_history("financial_market")
        assert len(history) == 1
        assert history[0].trust_score == 0.75

        # Step 2: Replay — cache hit
        result2 = router.analyze(context=context, payload=payload)
        assert result2 is not None
        assert router._cache_hits == 1
        assert router._cache_misses == 1

        # Step 3: Record outcome for calibration
        router.record_outcome(
            agent_name="financial_market",
            domain="financial_market",
            predicted_score=0.75,
            actual_outcome=0.80,
        )
        stats = cal_tracker.get_calibration_stats("financial_market")
        assert stats.record_count >= 1

        # Step 4: After enough calls, auto-persist fires
        # We already have 2 analyze calls, so auto_persist_every=2
        # should have triggered. But record_outcome also bumps.
        # Force a third analyze to ensure persistence:
        router.analyze(
            context={"domain": "financial_market"},
            payload={"instrument_id": "MSFT"},
        )
        router.analyze(
            context={"domain": "financial_market"},
            payload={"instrument_id": "GOOG"},
        )
        # Now 4 analyze calls — should have persisted at call 2 and 4.
        assert history_path.is_file()
        persisted = json.loads(history_path.read_text())
        assert "financial_market" in persisted

        # Step 5: Verify cache stats reflect usage
        cs = router.cache_stats()
        assert cs["hits"] >= 1
        assert cs["misses"] >= 1
        assert cs["cached_entries"] >= 1
        assert cs["ttl"] == 300.0

    def test_full_pipeline_via_api(self, client: TestClient) -> None:
        """Exercise the full pipeline through FastAPI endpoints."""
        # 1. Explain decision
        explain_resp = client.post(
            "/api/xai/explain-decision",
            json={
                "forecast_result": {
                    "model": "mock",
                    "forecasts": [{"id": "s1", "mean": [1.0, 2.0]}],
                },
            },
        )
        assert explain_resp.status_code == 200
        assert "explanation_id" in explain_resp.json()

        # 2. Record outcome
        outcome_resp = client.post(
            "/api/xai/record-outcome",
            json={
                "agent_name": "financial_market",
                "domain": "financial_market",
                "predicted_score": 0.8,
                "actual_outcome": 0.75,
            },
        )
        assert outcome_resp.status_code == 200
        assert outcome_resp.json()["status"] == "recorded"

        # 3. Check history
        history_resp = client.post("/api/xai/dashboard/history", json={})
        assert history_resp.status_code == 200

        # 4. Check cache stats
        cache_resp = client.get("/api/xai/cache/stats")
        assert cache_resp.status_code == 200
        assert "hits" in cache_resp.json()

        # 5. Update TTL
        ttl_resp = client.put(
            "/api/xai/cache/ttl",
            json={"ttl": 60.0},
        )
        assert ttl_resp.status_code == 200

        # 6. Check connectors
        conn_resp = client.post("/api/xai/connectors/health", json={})
        assert conn_resp.status_code == 200
        assert "connectors" in conn_resp.json()
