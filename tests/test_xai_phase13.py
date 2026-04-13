"""Phase 13 tests — response models, MCP tools, webhook retry, cache, E2E."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tollama.daemon.app import create_app

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
                "accuracy": {"name": "accuracy", "weight": 0.4, "score": score, "details": "stub"},
                "freshness": {"name": "freshness", "weight": 0.3, "score": 0.9, "details": "stub"},
                "consistency": {
                    "name": "consistency",
                    "weight": 0.3,
                    "score": 0.6,
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
            "audit": {"formula_version": "stub-v1", "agent_version": "0.1.0"},
        }
    )
    return agent


# ──────────────────────────────────────────────────────────────
# 1. response_model= wiring (OpenAPI schema completeness)
# ──────────────────────────────────────────────────────────────


class TestResponseModelWiring:
    def test_gate_decision_endpoint_returns_valid_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "allowed" in body
        assert "gates" in body

    def test_batch_analyze_endpoint_returns_valid_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/batch-analyze",
            json={
                "items": [
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "AAPL"},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "results" in body
        assert "count" in body

    def test_alerts_configure_returns_valid_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/alerts/configure",
            json={
                "thresholds": [{"domain": "financial_market", "min_trust_score": 0.5}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "threshold_count" in body
        assert "thresholds" in body

    def test_alerts_check_returns_valid_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/alerts/check",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "alerts" in body
        assert "alert_count" in body

    def test_trust_attribution_returns_valid_schema(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/trust-attribution",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "attributions" in body
        assert "total_score" in body


# ──────────────────────────────────────────────────────────────
# 2. MCP tool handlers exist and are importable
# ──────────────────────────────────────────────────────────────


class TestMCPToolHandlers:
    def test_gate_decision_tool_importable(self) -> None:
        from tollama.mcp.tools import tollama_gate_decision

        assert callable(tollama_gate_decision)

    def test_batch_analyze_tool_importable(self) -> None:
        from tollama.mcp.tools import tollama_batch_analyze

        assert callable(tollama_batch_analyze)

    def test_alerts_configure_tool_importable(self) -> None:
        from tollama.mcp.tools import tollama_alerts_configure

        assert callable(tollama_alerts_configure)

    def test_alerts_check_tool_importable(self) -> None:
        from tollama.mcp.tools import tollama_alerts_check

        assert callable(tollama_alerts_check)

    def test_mcp_schemas_importable(self) -> None:
        from tollama.mcp.schemas import (
            AlertsCheckToolInput,
            AlertsConfigureToolInput,
            BatchAnalyzeToolInput,
            GateDecisionToolInput,
        )

        assert GateDecisionToolInput is not None
        assert BatchAnalyzeToolInput is not None
        assert AlertsConfigureToolInput is not None
        assert AlertsCheckToolInput is not None


# ──────────────────────────────────────────────────────────────
# 3. Webhook retry with exponential backoff
# ──────────────────────────────────────────────────────────────


class TestWebhookRetry:
    def test_fire_webhook_success(self) -> None:
        from tollama.xai.api import _fire_webhook

        logger = MagicMock()
        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_post.return_value = mock_resp
            result = _fire_webhook("http://example.com/hook", {"test": True}, logger)
        assert result is True
        mock_post.assert_called_once()

    def test_fire_webhook_retries_on_500(self) -> None:
        from tollama.xai.api import _fire_webhook

        logger = MagicMock()
        with patch("httpx.post") as mock_post, patch("time.sleep"):
            mock_resp_500 = MagicMock()
            mock_resp_500.status_code = 500
            mock_resp_200 = MagicMock()
            mock_resp_200.status_code = 200
            mock_post.side_effect = [mock_resp_500, mock_resp_200]
            result = _fire_webhook("http://example.com/hook", {"test": True}, logger)
        assert result is True
        assert mock_post.call_count == 2

    def test_fire_webhook_retries_on_exception(self) -> None:
        from tollama.xai.api import _fire_webhook

        logger = MagicMock()
        with patch("httpx.post") as mock_post, patch("time.sleep"):
            mock_post.side_effect = ConnectionError("refused")
            result = _fire_webhook("http://example.com/hook", {"test": True}, logger)
        assert result is False
        assert mock_post.call_count == 3

    def test_fire_webhook_no_retry_on_4xx(self) -> None:
        from tollama.xai.api import _fire_webhook

        logger = MagicMock()
        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_post.return_value = mock_resp
            result = _fire_webhook("http://example.com/hook", {"test": True}, logger)
        assert result is True  # 4xx is accepted (client error, not retried)
        mock_post.assert_called_once()


# ──────────────────────────────────────────────────────────────
# 4. Cache invalidation API
# ──────────────────────────────────────────────────────────────


class TestCacheInvalidationAPI:
    def test_cache_stats_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/xai/cache/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "hits" in body
        assert "misses" in body
        assert "hit_rate" in body
        assert "cached_entries" in body
        assert "ttl" in body

    def test_cache_ttl_update(self, client: TestClient) -> None:
        resp = client.put("/api/xai/cache/ttl", json={"ttl": 60.0})
        assert resp.status_code == 200
        body = resp.json()
        assert "previous_ttl" in body
        assert "new_ttl" in body
        assert body["new_ttl"] == 60.0

    def test_cache_invalidate(self, client: TestClient) -> None:
        resp = client.delete("/api/xai/cache/invalidate")
        assert resp.status_code == 200
        body = resp.json()
        assert "cleared" in body
        assert isinstance(body["cleared"], int)

    def test_cache_invalidate_returns_count(self, client: TestClient) -> None:
        # Enable caching, run an analysis, then invalidate
        client.put("/api/xai/cache/ttl", json={"ttl": 300.0})
        client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        resp = client.delete("/api/xai/cache/invalidate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleared"] >= 0


# ──────────────────────────────────────────────────────────────
# 5. Native async batch (uses analyze_async)
# ──────────────────────────────────────────────────────────────


class TestNativeAsyncBatch:
    def test_batch_still_works_after_async_refactor(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/batch-analyze",
            json={
                "items": [
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "AAPL"},
                    },
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "MSFT"},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert all(r["status"] == "ok" for r in body["results"])

    def test_batch_no_agent_after_async_refactor(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/batch-analyze",
            json={
                "items": [
                    {"context": {"domain": "nonexistent_domain"}, "payload": {"x": 1}},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["status"] == "no_agent"


# ──────────────────────────────────────────────────────────────
# 6. Client methods for new endpoints
# ──────────────────────────────────────────────────────────────


class TestClientNewMethods:
    def test_sync_client_has_set_cache_ttl(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "set_cache_ttl")
        assert callable(c.set_cache_ttl)

    def test_sync_client_has_invalidate_cache(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "invalidate_cache")
        assert callable(c.invalidate_cache)

    def test_async_client_has_set_cache_ttl(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "set_cache_ttl")

    def test_async_client_has_invalidate_cache(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "invalidate_cache")


# ──────────────────────────────────────────────────────────────
# 7. E2E Integration: Connector → Analysis → Gate → Alert
# ──────────────────────────────────────────────────────────────


class TestE2EFlow:
    def test_full_pipeline_connector_to_alert(self, client: TestClient) -> None:
        """E2E: analyze → gate-decision → alert-check in sequence."""
        # Step 1: Configure alert thresholds
        resp = client.post(
            "/api/xai/alerts/configure",
            json={
                "thresholds": [
                    {
                        "domain": "financial_market",
                        "min_trust_score": 0.99,  # Very high → will trigger
                        "risk_categories": ["RED"],
                        "alert_on_trend": ["declining"],
                    },
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["threshold_count"] == 1

        # Step 2: Run gate-decision (trust analysis + gating)
        gate_resp = client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
                "trust_threshold": 0.5,
            },
        )
        assert gate_resp.status_code == 200
        gate_body = gate_resp.json()
        assert "allowed" in gate_body
        trust_score = gate_body["trust_score"]
        assert trust_score is not None

        # Step 3: Check alerts — should trigger because min_trust_score is 0.99
        alert_resp = client.post(
            "/api/xai/alerts/check",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert alert_resp.status_code == 200
        alert_body = alert_resp.json()
        assert alert_body["trust_result"] is not None
        # The alert should fire since score < 0.99 threshold
        if trust_score < 0.99:
            assert alert_body["alert_count"] >= 1
            assert any("trust_score" in r for r in alert_body["alerts"][0]["reasons"])

    def test_full_pipeline_batch_then_gate(self, client: TestClient) -> None:
        """E2E: batch-analyze → gate-decision for top result."""
        # Step 1: Batch analyze multiple instruments
        batch_resp = client.post(
            "/api/xai/batch-analyze",
            json={
                "items": [
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "AAPL"},
                    },
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "MSFT"},
                    },
                ],
            },
        )
        assert batch_resp.status_code == 200
        results = batch_resp.json()["results"]
        assert len(results) == 2

        # Step 2: Gate decision on same context
        gate_resp = client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert gate_resp.status_code == 200
        assert "allowed" in gate_resp.json()

    def test_full_pipeline_attribution_after_analysis(self, client: TestClient) -> None:
        """E2E: trust-attribution → verify component breakdown."""
        attr_resp = client.post(
            "/api/xai/trust-attribution",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert attr_resp.status_code == 200
        body = attr_resp.json()
        assert len(body["attributions"]) > 0
        assert body["total_score"] > 0
        assert body["top_driver"] is not None

    def test_cache_flow_stats_invalidate(self, client: TestClient) -> None:
        """E2E: enable cache → analyze → check stats → invalidate → verify."""
        # Enable cache
        client.put("/api/xai/cache/ttl", json={"ttl": 300.0})

        # Run analysis
        client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )

        # Check stats
        stats_resp = client.get("/api/xai/cache/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["misses"] >= 1

        # Run same analysis again — should be cached
        client.post(
            "/api/xai/gate-decision",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        stats2 = client.get("/api/xai/cache/stats").json()
        assert stats2["hits"] >= 1

        # Invalidate
        inv_resp = client.delete("/api/xai/cache/invalidate")
        assert inv_resp.status_code == 200
        assert inv_resp.json()["cleared"] >= 1

        # Stats should show 0 cached entries
        stats3 = client.get("/api/xai/cache/stats").json()
        assert stats3["cached_entries"] == 0
