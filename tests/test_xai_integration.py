"""Integration tests for XAI endpoints via FastAPI TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tollama.daemon.app import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


# ──────────────────────────────────────────────────────────────
# /api/xai/explain-decision
# ──────────────────────────────────────────────────────────────


class TestExplainDecision:
    def test_returns_200_with_minimal_forecast(self, client: TestClient) -> None:
        payload = {
            "forecast_result": {
                "model": "mock",
                "forecasts": [{"id": "s1", "mean": [1.0, 2.0]}],
            },
        }
        response = client.post("/api/xai/explain-decision", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "explanation_id" in body
        assert "version" in body

    def test_returns_error_on_missing_forecast_result(self, client: TestClient) -> None:
        response = client.post("/api/xai/explain-decision", json={})
        assert response.status_code in (400, 422)

    def test_with_time_series_data(self, client: TestClient) -> None:
        payload = {
            "forecast_result": {
                "model": "mock",
                "forecasts": [{"id": "s1", "mean": [1.0, 2.0]}],
            },
            "time_series_data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
        response = client.post("/api/xai/explain-decision", json=payload)
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────
# /api/xai/trust-breakdown
# ──────────────────────────────────────────────────────────────


class TestTrustBreakdown:
    def test_returns_200_with_valid_input(self, client: TestClient) -> None:
        payload = {
            "trust_score": 0.75,
            "metrics": {"brier_score": 0.15, "log_loss": 0.3, "ece": 0.05},
        }
        response = client.post("/api/xai/trust-breakdown", json=payload)
        assert response.status_code == 200

    def test_returns_error_on_missing_required_fields(self, client: TestClient) -> None:
        response = client.post("/api/xai/trust-breakdown", json={})
        assert response.status_code in (400, 422)


# ──────────────────────────────────────────────────────────────
# /api/xai/forecast-decompose
# ──────────────────────────────────────────────────────────────


class TestForecastDecompose:
    def test_returns_200_with_data(self, client: TestClient) -> None:
        payload = {
            "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
        response = client.post("/api/xai/forecast-decompose", json=payload)
        assert response.status_code == 200

    def test_returns_error_on_missing_data(self, client: TestClient) -> None:
        response = client.post("/api/xai/forecast-decompose", json={})
        assert response.status_code in (400, 422)


# ──────────────────────────────────────────────────────────────
# /api/xai/model-card
# ──────────────────────────────────────────────────────────────


class TestModelCard:
    def test_returns_200_with_model_info(self, client: TestClient) -> None:
        payload = {
            "model_info": {
                "name": "mock",
                "version": "1.0",
                "description": "Test model",
            },
        }
        response = client.post("/api/xai/model-card", json=payload)
        assert response.status_code == 200

    def test_markdown_format_returns_content_key(self, client: TestClient) -> None:
        payload = {
            "model_info": {
                "name": "mock",
                "version": "1.0",
                "description": "Test model",
            },
            "format": "markdown",
        }
        response = client.post("/api/xai/model-card", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["format"] == "markdown"
        assert "content" in body


# ──────────────────────────────────────────────────────────────
# /api/xai/decision-report
# ──────────────────────────────────────────────────────────────


class TestDecisionReport:
    def test_returns_200_with_explanation(self, client: TestClient) -> None:
        payload = {
            "explanation": {
                "explanation_id": "test-001",
                "version": "3.8",
                "input_explanation": {},
                "plan_explanation": {},
                "decision_policy_explanation": {},
            },
        }
        response = client.post("/api/xai/decision-report", json=payload)
        assert response.status_code == 200

    def test_markdown_format(self, client: TestClient) -> None:
        payload = {
            "explanation": {
                "explanation_id": "test-001",
                "version": "3.8",
                "input_explanation": {},
                "plan_explanation": {},
                "decision_policy_explanation": {},
            },
            "format": "markdown",
        }
        response = client.post("/api/xai/decision-report", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["format"] == "markdown"
        assert "content" in body


# ──────────────────────────────────────────────────────────────
# /api/xai/record-outcome
# ──────────────────────────────────────────────────────────────


class TestRecordOutcome:
    def test_returns_200_with_valid_input(self, client: TestClient) -> None:
        payload = {
            "agent_name": "financial_market",
            "domain": "financial_market",
            "predicted_score": 0.8,
            "actual_outcome": 0.75,
        }
        response = client.post("/api/xai/record-outcome", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "recorded"
        assert body["agent_name"] == "financial_market"

    def test_returns_error_on_missing_fields(self, client: TestClient) -> None:
        response = client.post("/api/xai/record-outcome", json={})
        assert response.status_code in (400, 422)

    def test_returns_calibration_stats(self, client: TestClient) -> None:
        payload = {
            "agent_name": "news",
            "domain": "news",
            "predicted_score": 0.6,
            "actual_outcome": 0.65,
            "component_scores": {"freshness": 0.7, "source_diversity": 0.5},
        }
        response = client.post("/api/xai/record-outcome", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "calibration_stats" in body


# ──────────────────────────────────────────────────────────────
# /api/xai/dashboard/history
# ──────────────────────────────────────────────────────────────


class TestDashboardHistory:
    def test_returns_200_with_defaults(self, client: TestClient) -> None:
        response = client.post("/api/xai/dashboard/history", json={})
        assert response.status_code == 200
        body = response.json()
        assert "domains" in body

    def test_with_domain_filter(self, client: TestClient) -> None:
        payload = {"domains": ["financial_market"], "limit": 10}
        response = client.post("/api/xai/dashboard/history", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert "financial_market" in body["domains"]

    def test_with_stats_disabled(self, client: TestClient) -> None:
        payload = {"include_stats": False}
        response = client.post("/api/xai/dashboard/history", json=payload)
        assert response.status_code == 200


# ──────────────────────────────────────────────────────────────
# /api/xai/connectors/health
# ──────────────────────────────────────────────────────────────


class TestConnectorsHealth:
    def test_returns_200_with_defaults(self, client: TestClient) -> None:
        response = client.post("/api/xai/connectors/health", json={})
        assert response.status_code == 200
        body = response.json()
        assert "connectors" in body
        assert isinstance(body["connectors"], list)

    def test_with_domain_filter(self, client: TestClient) -> None:
        payload = {"domains": ["financial_market"]}
        response = client.post("/api/xai/connectors/health", json=payload)
        assert response.status_code == 200
        body = response.json()
        for conn in body["connectors"]:
            assert conn["domain"] == "financial_market"


# ──────────────────────────────────────────────────────────────
# /api/xai/cache/stats
# ──────────────────────────────────────────────────────────────


class TestCacheStats:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/xai/cache/stats")
        assert response.status_code == 200
        body = response.json()
        assert "hits" in body
        assert "misses" in body
        assert "hit_rate" in body
        assert "cached_entries" in body
        assert "ttl" in body

    def test_fresh_cache_has_zero_hits(self, client: TestClient) -> None:
        response = client.get("/api/xai/cache/stats")
        body = response.json()
        assert body["hits"] == 0
        assert body["misses"] == 0
        assert body["hit_rate"] == 0.0
