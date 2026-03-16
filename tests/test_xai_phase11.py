"""Phase 11 tests — singleton router, gating, connector-fed, attribution, batch, alerts."""

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
    name: str, domain: str, score: float = 0.7,
) -> Any:
    """Create a minimal stub trust agent."""
    agent = MagicMock()
    agent.agent_name = name
    agent.domain = domain
    agent.priority = 10

    def _supports(context: dict[str, Any]) -> bool:
        return context.get("domain") == domain

    agent.supports = _supports
    agent.analyze = MagicMock(return_value={
        "agent_name": name,
        "domain": domain,
        "trust_score": score,
        "risk_category": "GREEN",
        "calibration_status": "moderate_trust",
        "component_breakdown": {
            "accuracy": {
                "name": "accuracy",
                "weight": 0.4,
                "score": score,
                "details": "stub",
            },
            "freshness": {
                "name": "freshness",
                "weight": 0.3,
                "score": 0.9,
                "details": "stub",
            },
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
        "audit": {
            "formula_version": "stub-v1",
            "agent_version": "0.1.0",
        },
    })
    return agent


def _make_trust_result(
    score: float = 0.7,
    risk: str = "GREEN",
    violations: list[TrustViolation] | None = None,
) -> NormalizedTrustResult:
    return NormalizedTrustResult(
        agent_name="test_agent",
        domain="financial_market",
        trust_score=score,
        risk_category=risk,
        calibration_status="moderate_trust",
        component_breakdown={
            "accuracy": TrustComponent(
                name="accuracy", weight=0.4, score=0.8, details="test",
            ),
            "freshness": TrustComponent(
                name="freshness", weight=0.3, score=0.9, details="test",
            ),
            "consistency": TrustComponent(
                name="consistency", weight=0.3, score=0.6, details="test",
            ),
        },
        violations=violations or [],
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
# 1. Singleton trust router
# ──────────────────────────────────────────────────────────────


class TestSingletonRouter:
    def test_app_state_has_trust_router(self) -> None:
        app = create_app()
        assert hasattr(app.state, "trust_router")
        assert app.state.trust_router is not None

    def test_singleton_router_has_agents(self) -> None:
        app = create_app()
        agents = app.state.trust_router.registry.agents
        assert len(agents) >= 1


# ──────────────────────────────────────────────────────────────
# 2. Trust-aware auto-execution gating
# ──────────────────────────────────────────────────────────────


class TestGateDecision:
    def test_gate_allows_high_trust(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.8, risk="GREEN")
        gate = router.gate_decision(result, trust_threshold=0.5)
        assert gate["allowed"] is True
        assert all(g["status"] in ("PASS", "WARN") for g in gate["gates"])

    def test_gate_blocks_low_trust(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.3, risk="GREEN")
        gate = router.gate_decision(result, trust_threshold=0.5)
        assert gate["allowed"] is False
        statuses = {g["gate"]: g["status"] for g in gate["gates"]}
        assert statuses["trust_score"] == "BLOCK"

    def test_gate_blocks_red_risk(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.8, risk="RED")
        gate = router.gate_decision(result, trust_threshold=0.5)
        assert gate["allowed"] is False
        statuses = {g["gate"]: g["status"] for g in gate["gates"]}
        assert statuses["risk_category"] == "BLOCK"

    def test_gate_blocks_critical_violations(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(
            score=0.8,
            risk="GREEN",
            violations=[
                TrustViolation(
                    name="data_gap", severity="critical", detail="missing data",
                ),
            ],
        )
        gate = router.gate_decision(result, trust_threshold=0.5)
        assert gate["allowed"] is False

    def test_gate_warns_yellow_risk(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.8, risk="YELLOW")
        gate = router.gate_decision(result, trust_threshold=0.5)
        assert gate["allowed"] is True
        statuses = {g["gate"]: g["status"] for g in gate["gates"]}
        assert statuses["risk_category"] == "WARN"


class TestGateDecisionEndpoint:
    def test_gate_decision_endpoint(self, client: TestClient) -> None:
        resp = client.post("/api/xai/gate-decision", json={
            "context": {"domain": "financial_market"},
            "payload": {"instrument_id": "AAPL"},
            "trust_threshold": 0.3,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "allowed" in body
        assert "gates" in body

    def test_gate_decision_no_agent(self, client: TestClient) -> None:
        resp = client.post("/api/xai/gate-decision", json={
            "context": {"domain": "nonexistent_domain"},
            "payload": {},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["allowed"] is False


# ──────────────────────────────────────────────────────────────
# 3. Connector-fed trust analysis
# ──────────────────────────────────────────────────────────────


class TestConnectorFedAnalysis:
    def test_analyze_with_connector(self) -> None:
        from tollama.xai.connectors.protocol import ConnectorResult

        registry = TrustAgentRegistry()
        agent = _make_stub_agent("financial_market", "financial_market")
        registry.register(agent)
        router = TrustRouter(registry)

        connector = MagicMock()
        connector.connector_name = "test_connector"
        connector.domain = "financial_market"
        connector.fetch = MagicMock(return_value=ConnectorResult(
            domain="financial_market",
            payload={"instrument_id": "AAPL", "price": 150.0},
            source_id="test-001",
            source_type="mock",
        ))

        result = router.analyze_with_connector(
            connector=connector,
            identifier="AAPL",
        )
        assert result is not None
        assert result.trust_score == 0.7
        connector.fetch.assert_called_once()

    def test_analyze_with_connector_failure(self) -> None:
        from tollama.xai.connectors.protocol import (
            ConnectorError,
            ConnectorFetchError,
        )

        registry = TrustAgentRegistry()
        agent = _make_stub_agent("financial_market", "financial_market")
        registry.register(agent)
        router = TrustRouter(registry)

        connector = MagicMock()
        connector.connector_name = "failing_connector"
        connector.domain = "financial_market"
        connector.fetch = MagicMock(side_effect=ConnectorFetchError(
            ConnectorError(
                domain="financial_market",
                error_type="timeout",
                message="request timed out",
            ),
        ))

        result = router.analyze_with_connector(
            connector=connector,
            identifier="AAPL",
        )
        assert result is None


# ──────────────────────────────────────────────────────────────
# 4. SHAP feature attribution for trust
# ──────────────────────────────────────────────────────────────


class TestTrustFeatureAttribution:
    def test_attribution_basic(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.75)
        attr = router.trust_feature_attribution(result)
        assert "attributions" in attr
        assert len(attr["attributions"]) == 3
        assert attr["total_score"] == 0.75
        assert attr["top_driver"] is not None

    def test_attribution_ordered_by_contribution(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.75)
        attr = router.trust_feature_attribution(result)
        contributions = [a["contribution"] for a in attr["attributions"]]
        assert contributions == sorted(contributions, reverse=True)

    def test_attribution_empty_components(self) -> None:
        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        result = _make_trust_result(score=0.5)
        result.component_breakdown = {}
        attr = router.trust_feature_attribution(result)
        assert attr["attributions"] == []
        assert attr.get("top_driver") is None


class TestTrustAttributionEndpoint:
    def test_trust_attribution_endpoint(self, client: TestClient) -> None:
        resp = client.post("/api/xai/trust-attribution", json={
            "context": {"domain": "financial_market"},
            "payload": {"instrument_id": "AAPL"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "attributions" in body
        assert "total_score" in body

    def test_trust_attribution_no_agent(self, client: TestClient) -> None:
        resp = client.post("/api/xai/trust-attribution", json={
            "context": {"domain": "nonexistent"},
            "payload": {},
        })
        assert resp.status_code == 404


# ──────────────────────────────────────────────────────────────
# 5. Batch trust analysis
# ──────────────────────────────────────────────────────────────


class TestBatchAnalyzeEndpoint:
    def test_batch_analyze_single(self, client: TestClient) -> None:
        resp = client.post("/api/xai/batch-analyze", json={
            "items": [
                {
                    "context": {"domain": "financial_market"},
                    "payload": {"instrument_id": "AAPL"},
                },
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["results"][0]["status"] == "ok"

    def test_batch_analyze_multiple(self, client: TestClient) -> None:
        resp = client.post("/api/xai/batch-analyze", json={
            "items": [
                {"context": {"domain": "financial_market"}, "payload": {"instrument_id": "AAPL"}},
                {"context": {"domain": "financial_market"}, "payload": {"instrument_id": "MSFT"}},
                {"context": {"domain": "nonexistent"}, "payload": {}},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert body["results"][0]["status"] == "ok"
        assert body["results"][1]["status"] == "ok"
        assert body["results"][2]["status"] == "no_agent"

    def test_batch_analyze_empty_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/xai/batch-analyze", json={"items": []})
        assert resp.status_code in (400, 422)

    def test_batch_result_has_trust_fields(self, client: TestClient) -> None:
        resp = client.post("/api/xai/batch-analyze", json={
            "items": [
                {"context": {"domain": "financial_market"}, "payload": {"instrument_id": "X"}},
            ],
        })
        body = resp.json()
        r = body["results"][0]["result"]
        assert "trust_score" in r
        assert "risk_category" in r
        assert "why_trusted" in r


# ──────────────────────────────────────────────────────────────
# 6. Trust Alert / Threshold Webhook
# ──────────────────────────────────────────────────────────────


class TestAlertConfiguration:
    def test_configure_alerts(self, client: TestClient) -> None:
        resp = client.post("/api/xai/alerts/configure", json={
            "thresholds": [
                {
                    "domain": "financial_market",
                    "min_trust_score": 0.8,
                    "risk_categories": ["RED"],
                },
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "configured"
        assert body["threshold_count"] == 1

    def test_get_alert_config(self, client: TestClient) -> None:
        # Configure first
        client.post("/api/xai/alerts/configure", json={
            "thresholds": [
                {"domain": "financial_market", "min_trust_score": 0.9},
            ],
        })
        resp = client.get("/api/xai/alerts/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body["threshold_count"] >= 1

    def test_check_alerts_triggers(self, client: TestClient) -> None:
        # Configure a very high threshold that should trigger
        client.post("/api/xai/alerts/configure", json={
            "thresholds": [
                {
                    "domain": "financial_market",
                    "min_trust_score": 0.99,
                    "risk_categories": ["RED", "YELLOW"],
                },
            ],
        })
        resp = client.post("/api/xai/alerts/check", json={
            "context": {"domain": "financial_market"},
            "payload": {"instrument_id": "AAPL"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert_count"] >= 1
        assert len(body["alerts"]) >= 1
        assert body["trust_result"] is not None

    def test_check_alerts_no_trigger(self, client: TestClient) -> None:
        # Configure a very low threshold that shouldn't trigger
        client.post("/api/xai/alerts/configure", json={
            "thresholds": [
                {
                    "domain": "financial_market",
                    "min_trust_score": 0.01,
                    "risk_categories": [],
                },
            ],
        })
        resp = client.post("/api/xai/alerts/check", json={
            "context": {"domain": "financial_market"},
            "payload": {"instrument_id": "AAPL"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["alert_count"] == 0

    def test_check_alerts_no_matching_domain(self, client: TestClient) -> None:
        client.post("/api/xai/alerts/configure", json={
            "thresholds": [
                {"domain": "nonexistent", "min_trust_score": 0.99},
            ],
        })
        resp = client.post("/api/xai/alerts/check", json={
            "context": {"domain": "financial_market"},
            "payload": {"instrument_id": "AAPL"},
        })
        assert resp.status_code == 200
        assert resp.json()["alert_count"] == 0
