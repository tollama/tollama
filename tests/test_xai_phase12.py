"""Phase 12 tests — async batch, trend alerts, auto-connector, to_dict, response models, CLI."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tollama.daemon.app import create_app
from tollama.xai.trust_contract import (
    NormalizedTrustResult,
    TrustAudit,
    TrustComponent,
    TrustEvidence,
    TrustViolation,
)
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


def _make_trust_result(score: float = 0.7, risk: str = "GREEN") -> NormalizedTrustResult:
    return NormalizedTrustResult(
        agent_name="test_agent",
        domain="financial_market",
        trust_score=score,
        risk_category=risk,
        calibration_status="moderate_trust",
        component_breakdown={
            "accuracy": TrustComponent(name="accuracy", weight=0.4, score=0.8, details="test"),
            "freshness": TrustComponent(name="freshness", weight=0.3, score=0.9, details="test"),
        },
        violations=[
            TrustViolation(name="data_gap", severity="warning", detail="minor gap"),
        ],
        why_trusted="Test agent",
        evidence=TrustEvidence(
            source_type="test",
            source_ids=["test-001"],
            payload_schema="test_v1",
            attributes={},
        ),
        audit=TrustAudit(formula_version="test-v1", agent_version="0.1.0"),
    )


# ──────────────────────────────────────────────────────────────
# 1. NormalizedTrustResult.to_dict() and to_summary()
# ──────────────────────────────────────────────────────────────


class TestNormalizedTrustResultSerialization:
    def test_to_dict_has_all_fields(self) -> None:
        result = _make_trust_result()
        d = result.to_dict()
        assert d["agent_name"] == "test_agent"
        assert d["domain"] == "financial_market"
        assert d["trust_score"] == 0.7
        assert "accuracy" in d["component_breakdown"]
        assert d["component_breakdown"]["accuracy"]["score"] == 0.8
        assert d["component_breakdown"]["accuracy"]["weight"] == 0.4
        assert len(d["violations"]) == 1
        assert d["violations"][0]["name"] == "data_gap"
        assert "evidence" in d
        assert "audit" in d

    def test_to_summary_compact(self) -> None:
        result = _make_trust_result()
        s = result.to_summary()
        assert s["agent_name"] == "test_agent"
        assert s["trust_score"] == 0.7
        assert "component_breakdown" not in s
        assert "evidence" not in s
        assert len(s["violations"]) == 1

    def test_to_dict_empty_violations(self) -> None:
        result = _make_trust_result()
        result.violations = []
        d = result.to_dict()
        assert d["violations"] == []


# ──────────────────────────────────────────────────────────────
# 2. Async batch analysis (uses asyncio.gather)
# ──────────────────────────────────────────────────────────────


class TestAsyncBatchAnalyze:
    def test_batch_uses_to_summary(self, client: TestClient) -> None:
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
        r = body["results"][0]["result"]
        # to_summary() does NOT include component_breakdown or evidence
        assert "component_breakdown" not in r
        assert "evidence" not in r
        assert "trust_score" in r

    def test_batch_concurrent_multiple(self, client: TestClient) -> None:
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
                    {
                        "context": {"domain": "financial_market"},
                        "payload": {"instrument_id": "GOOG"},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert all(r["status"] == "ok" for r in body["results"])


# ──────────────────────────────────────────────────────────────
# 3. Connector auto-discovery
# ──────────────────────────────────────────────────────────────


class TestConnectorAutoDiscovery:
    def test_singleton_router_has_connector_registry(self) -> None:
        app = create_app()
        tr = app.state.trust_router
        assert tr.connector_registry is not None

    def test_analyze_with_auto_connector_mock(self) -> None:
        from tollama.xai.connectors.protocol import ConnectorResult

        registry = TrustAgentRegistry()
        agent = _make_stub_agent("financial_market", "financial_market")
        registry.register(agent)

        connector = MagicMock()
        connector.connector_name = "mock_fin"
        connector.domain = "financial_market"
        connector.supports = MagicMock(return_value=True)
        connector.fetch = MagicMock(
            return_value=ConnectorResult(
                domain="financial_market",
                payload={"instrument_id": "AAPL", "price": 150.0},
                source_id="test-001",
                source_type="mock",
            )
        )

        conn_registry = MagicMock()
        conn_registry.get = MagicMock(return_value=connector)

        router = TrustRouter(registry, connector_registry=conn_registry)
        result = router.analyze_with_auto_connector(
            domain="financial_market",
            identifier="AAPL",
        )
        assert result is not None
        conn_registry.get.assert_called_once()

    def test_analyze_with_auto_connector_no_registry(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = router.analyze_with_auto_connector(
            domain="financial_market",
            identifier="AAPL",
        )
        assert result is None

    def test_analyze_with_auto_connector_no_match(self) -> None:
        registry = TrustAgentRegistry()
        conn_registry = MagicMock()
        conn_registry.get = MagicMock(return_value=None)
        router = TrustRouter(registry, connector_registry=conn_registry)
        result = router.analyze_with_auto_connector(
            domain="nonexistent",
            identifier="X",
        )
        assert result is None


# ──────────────────────────────────────────────────────────────
# 4. Trust trend alerts
# ──────────────────────────────────────────────────────────────


class TestTrustTrendAlerts:
    def test_alert_on_declining_trend(self, client: TestClient) -> None:
        # Configure alert with trend detection
        client.post(
            "/api/xai/alerts/configure",
            json={
                "thresholds": [
                    {
                        "domain": "financial_market",
                        "min_trust_score": 0.01,
                        "risk_categories": [],
                        "alert_on_trend": ["declining"],
                    },
                ],
            },
        )
        # Check — no history yet so trend is "stable", shouldn't trigger
        resp = client.post(
            "/api/xai/alerts/check",
            json={
                "context": {"domain": "financial_market"},
                "payload": {"instrument_id": "AAPL"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["trust_result"] is not None
        # Trend field should be present
        assert "trend" in body["trust_result"]

    def test_alert_threshold_with_trend_field(self, client: TestClient) -> None:
        resp = client.post(
            "/api/xai/alerts/configure",
            json={
                "thresholds": [
                    {
                        "domain": "financial_market",
                        "min_trust_score": 0.5,
                        "alert_on_trend": ["declining", "stable"],
                    },
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["thresholds"][0]["alert_on_trend"] == ["declining", "stable"]


# ──────────────────────────────────────────────────────────────
# 5. OpenAPI response models exist
# ──────────────────────────────────────────────────────────────


class TestResponseModels:
    def test_gate_decision_response_model(self) -> None:
        from tollama.xai.api import GateDecisionResponse

        resp = GateDecisionResponse(
            allowed=True,
            gates=[{"gate": "trust_score", "status": "PASS", "detail": "ok"}],
            risk_category="GREEN",
            trust_score=0.8,
        )
        assert resp.allowed is True

    def test_batch_analyze_response_model(self) -> None:
        from tollama.xai.api import BatchAnalyzeResponse

        resp = BatchAnalyzeResponse(
            results=[{"status": "ok", "result": {"trust_score": 0.7}}],
            count=1,
        )
        assert resp.count == 1

    def test_alert_check_response_model(self) -> None:
        from tollama.xai.api import AlertCheckResponse

        resp = AlertCheckResponse(
            alerts=[],
            alert_count=0,
            trust_result=None,
        )
        assert resp.alert_count == 0

    def test_trust_attribution_response_model(self) -> None:
        from tollama.xai.api import TrustAttributionResponse

        resp = TrustAttributionResponse(
            attributions=[],
            baseline=0.0,
            total_score=0.5,
            top_driver=None,
        )
        assert resp.total_score == 0.5


# ──────────────────────────────────────────────────────────────
# 6. Client methods for new endpoints
# ──────────────────────────────────────────────────────────────


class TestClientMethods:
    def test_sync_client_has_gate_decision(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "gate_decision")
        assert callable(c.gate_decision)

    def test_sync_client_has_batch_analyze(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "batch_analyze")

    def test_sync_client_has_configure_alerts(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "configure_alerts")

    def test_sync_client_has_check_alerts(self) -> None:
        from tollama.client.http import TollamaClient

        c = TollamaClient()
        assert hasattr(c, "check_alerts")

    def test_async_client_has_gate_decision(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "gate_decision")

    def test_async_client_has_batch_analyze(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "batch_analyze")

    def test_async_client_has_configure_alerts(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "configure_alerts")

    def test_async_client_has_check_alerts(self) -> None:
        from tollama.client.http import AsyncTollamaClient

        c = AsyncTollamaClient()
        assert hasattr(c, "check_alerts")
