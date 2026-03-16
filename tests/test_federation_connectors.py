"""Federation connector tests — validates live/mock switching and health checks.

Tests the helpers.py env-var based connector registry configuration
and the live connector health_check/fetch behavior.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import httpx
import pytest

from tollama.xai.connectors.helpers import (
    build_default_async_connector_registry,
    build_default_connector_registry,
)
from tollama.xai.connectors.live import (
    HttpFinancialConnector,
    HttpNewsConnector,
)
from tollama.xai.connectors.protocol import ConnectorFetchError


def _find_by_domain(connectors: list, domain: str):
    """Find a connector by domain in a list-based registry."""
    for c in connectors:
        if c.domain == domain:
            return c
    return None


# === Helpers env-var switching ===


class TestConnectorRegistryMockMode:
    """Default mode (TOLLAMA_USE_LIVE_CONNECTORS not set) uses mocks."""

    def test_default_registry_has_mock_connectors(self):
        with patch.dict(os.environ, {}, clear=True):
            registry = build_default_connector_registry()
        domains = {c.domain for c in registry._connectors}
        assert "financial_market" in domains
        assert "news" in domains

    def test_default_registry_mock_financial_is_not_http(self):
        with patch.dict(os.environ, {}, clear=True):
            registry = build_default_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        assert fin is not None
        assert not isinstance(fin, HttpFinancialConnector)


class TestConnectorRegistryLiveMode:
    """TOLLAMA_USE_LIVE_CONNECTORS=true uses live HTTP connectors."""

    def test_live_mode_uses_http_connectors(self):
        env = {
            "TOLLAMA_USE_LIVE_CONNECTORS": "true",
            "NEWS_AGENT_URL": "http://news:8090",
            "FINANCIAL_AGENT_URL": "http://fin:8091",
        }
        with patch.dict(os.environ, env, clear=True):
            registry = build_default_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        news = _find_by_domain(registry._connectors, "news")
        assert isinstance(fin, HttpFinancialConnector)
        assert isinstance(news, HttpNewsConnector)
        assert fin._base_url == "http://fin:8091"
        assert news._base_url == "http://news:8090"

    def test_live_mode_with_api_keys(self):
        env = {
            "TOLLAMA_USE_LIVE_CONNECTORS": "true",
            "NEWS_AGENT_API_KEY": "news-key-123",
            "FINANCIAL_AGENT_API_KEY": "fin-key-456",
        }
        with patch.dict(os.environ, env, clear=True):
            registry = build_default_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        news = _find_by_domain(registry._connectors, "news")
        assert isinstance(fin, HttpFinancialConnector)
        assert fin._api_key == "fin-key-456"
        assert isinstance(news, HttpNewsConnector)
        assert news._api_key == "news-key-123"

    def test_live_mode_other_domains_remain_mock(self):
        env = {"TOLLAMA_USE_LIVE_CONNECTORS": "true"}
        with patch.dict(os.environ, env, clear=True):
            registry = build_default_connector_registry()
        sc = _find_by_domain(registry._connectors, "supply_chain")
        assert sc is not None
        assert not isinstance(sc, HttpFinancialConnector)

    def test_default_urls_when_env_not_set(self):
        env = {"TOLLAMA_USE_LIVE_CONNECTORS": "true"}
        with patch.dict(os.environ, env, clear=True):
            registry = build_default_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        news = _find_by_domain(registry._connectors, "news")
        assert isinstance(fin, HttpFinancialConnector)
        assert fin._base_url == "http://localhost:8091"
        assert isinstance(news, HttpNewsConnector)
        assert news._base_url == "http://localhost:8090"


# === Health Check ===


class TestHealthCheck:
    """Test health_check method on live connectors."""

    def test_financial_health_check_unavailable_when_no_server(self):
        connector = HttpFinancialConnector(base_url="http://localhost:19999", timeout=1.0)
        result = connector.health_check()
        assert result["status"] == "unavailable"
        assert result["latency_ms"] is None

    def test_news_health_check_unavailable_when_no_server(self):
        connector = HttpNewsConnector(base_url="http://localhost:19999", timeout=1.0)
        result = connector.health_check()
        assert result["status"] == "unavailable"
        assert result["latency_ms"] is None


# === Connector fetch with mock HTTP responses ===


class TestFinancialConnectorFetch:
    """Test HttpFinancialConnector.fetch() behavior."""

    def test_successful_fetch(self, httpx_mock):
        payload = {
            "instrument_id": "AAPL",
            "liquidity_depth": 0.85,
            "bid_ask_spread_bps": 5.0,
            "realized_volatility": 0.22,
            "execution_risk": 0.12,
            "data_freshness": 0.95,
        }
        httpx_mock.add_response(
            url="http://test:8091/instruments/AAPL",
            json=payload,
        )
        connector = HttpFinancialConnector(base_url="http://test:8091")
        result = connector.fetch("AAPL", {})
        assert result.domain == "financial_market"
        assert result.payload["instrument_id"] == "AAPL"
        assert result.payload["liquidity_depth"] == 0.85

    def test_timeout_raises_retryable_error(self, httpx_mock):
        httpx_mock.add_exception(httpx.ReadTimeout("timeout"))
        connector = HttpFinancialConnector(base_url="http://test:8091", timeout=0.1)
        with pytest.raises(ConnectorFetchError) as exc_info:
            connector.fetch("AAPL", {})
        assert exc_info.value.error.retryable is True
        assert exc_info.value.error.error_type == "network"

    def test_auth_error_not_retryable(self, httpx_mock):
        httpx_mock.add_response(
            url="http://test:8091/instruments/AAPL",
            status_code=401,
        )
        connector = HttpFinancialConnector(base_url="http://test:8091")
        with pytest.raises(ConnectorFetchError) as exc_info:
            connector.fetch("AAPL", {})
        assert exc_info.value.error.retryable is False
        assert exc_info.value.error.error_type == "auth"

    def test_server_error_retryable(self, httpx_mock):
        httpx_mock.add_response(
            url="http://test:8091/instruments/AAPL",
            status_code=500,
        )
        connector = HttpFinancialConnector(base_url="http://test:8091")
        with pytest.raises(ConnectorFetchError) as exc_info:
            connector.fetch("AAPL", {})
        assert exc_info.value.error.retryable is True
        assert exc_info.value.error.error_type == "internal"


class TestNewsConnectorFetch:
    """Test HttpNewsConnector.fetch() behavior."""

    def test_successful_fetch(self, httpx_mock):
        payload = {
            "story_id": "story-001",
            "source_credibility": 0.85,
            "corroboration": 0.7,
            "contradiction_score": 0.1,
            "propagation_delay_seconds": 120.0,
            "freshness_score": 0.9,
            "novelty": 0.6,
        }
        httpx_mock.add_response(
            url="http://test:8090/stories/story-001",
            json=payload,
        )
        connector = HttpNewsConnector(base_url="http://test:8090")
        result = connector.fetch("story-001", {})
        assert result.domain == "news"
        assert result.payload["story_id"] == "story-001"

    def test_not_found_error(self, httpx_mock):
        httpx_mock.add_response(
            url="http://test:8090/stories/missing",
            status_code=404,
        )
        connector = HttpNewsConnector(base_url="http://test:8090")
        with pytest.raises(ConnectorFetchError) as exc_info:
            connector.fetch("missing", {})
        assert exc_info.value.error.error_type == "not_found"
        assert exc_info.value.error.retryable is False


# === Async registry ===


class TestAsyncRegistry:
    """Test async connector registry configuration."""

    def test_async_live_mode_urls(self):
        env = {
            "TOLLAMA_USE_LIVE_CONNECTORS": "true",
            "NEWS_AGENT_URL": "http://news:8090",
            "FINANCIAL_AGENT_URL": "http://fin:8091",
        }
        with patch.dict(os.environ, env, clear=True):
            registry = build_default_async_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        news = _find_by_domain(registry._connectors, "news")
        assert fin._base_url == "http://fin:8091"
        assert news._base_url == "http://news:8090"

    def test_async_mock_mode_uses_fallback_url(self):
        with patch.dict(os.environ, {}, clear=True):
            registry = build_default_async_connector_registry()
        fin = _find_by_domain(registry._connectors, "financial_market")
        news = _find_by_domain(registry._connectors, "news")
        assert fin._base_url == "http://localhost:8080"
        assert news._base_url == "http://localhost:8080"
