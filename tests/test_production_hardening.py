"""Tests for Phase 4: Production Hardening.

Covers:
- Config & Auth: env-var based timeout/retry/cache settings
- Rate Limit & Cache: domain-specific TTL, rate-limit header extraction
- Error Taxonomy: error classification and retryable checks
- Version Compatibility Matrix: formula/agent version checks
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ── Phase 4.1: Config & Auth ────────────────────────────────


class TestConnectorConfig:
    """Verify env-var based connector configuration."""

    def test_default_timeout(self):
        from tollama.xai.connectors.helpers import _connector_timeout

        with patch.dict("os.environ", {}, clear=False):
            assert _connector_timeout() == 10.0

    def test_custom_timeout(self):
        from tollama.xai.connectors.helpers import _connector_timeout

        with patch.dict("os.environ", {"TOLLAMA_CONNECTOR_TIMEOUT": "30"}):
            assert _connector_timeout() == 30.0

    def test_timeout_minimum_clamp(self):
        from tollama.xai.connectors.helpers import _connector_timeout

        with patch.dict("os.environ", {"TOLLAMA_CONNECTOR_TIMEOUT": "0.1"}):
            assert _connector_timeout() == 1.0

    def test_invalid_timeout_fallback(self):
        from tollama.xai.connectors.helpers import _connector_timeout

        with patch.dict("os.environ", {"TOLLAMA_CONNECTOR_TIMEOUT": "abc"}):
            assert _connector_timeout() == 10.0

    def test_default_max_retries(self):
        from tollama.xai.connectors.helpers import _connector_max_retries

        with patch.dict("os.environ", {}, clear=False):
            assert _connector_max_retries() == 2

    def test_custom_max_retries(self):
        from tollama.xai.connectors.helpers import _connector_max_retries

        with patch.dict("os.environ", {"TOLLAMA_CONNECTOR_MAX_RETRIES": "5"}):
            assert _connector_max_retries() == 5

    def test_zero_retries_allowed(self):
        from tollama.xai.connectors.helpers import _connector_max_retries

        with patch.dict("os.environ", {"TOLLAMA_CONNECTOR_MAX_RETRIES": "0"}):
            assert _connector_max_retries() == 0

    def test_default_retry_delay(self):
        from tollama.xai.connectors.helpers import _connector_retry_base_delay

        with patch.dict("os.environ", {}, clear=False):
            assert _connector_retry_base_delay() == 0.5

    def test_custom_retry_delay(self):
        from tollama.xai.connectors.helpers import _connector_retry_base_delay

        env = {"TOLLAMA_CONNECTOR_RETRY_BASE_DELAY": "2.0"}
        with patch.dict("os.environ", env):
            assert _connector_retry_base_delay() == 2.0


class TestLiveConnectorRetryInit:
    """Verify live connectors accept retry parameters."""

    def test_financial_connector_retry_params(self):
        from tollama.xai.connectors.live import HttpFinancialConnector

        c = HttpFinancialConnector(
            base_url="http://localhost:8091",
            timeout=15.0,
            max_retries=3,
            retry_base_delay=1.0,
        )
        assert c._timeout == 15.0
        assert c._max_retries == 3
        assert c._retry_base_delay == 1.0

    def test_news_connector_retry_params(self):
        from tollama.xai.connectors.live import HttpNewsConnector

        c = HttpNewsConnector(
            base_url="http://localhost:8090",
            timeout=20.0,
            max_retries=0,
            retry_base_delay=0.0,
        )
        assert c._timeout == 20.0
        assert c._max_retries == 0
        assert c._retry_base_delay == 0.0

    def test_financial_connector_defaults(self):
        from tollama.xai.connectors.live import HttpFinancialConnector

        c = HttpFinancialConnector(base_url="http://localhost:8091")
        assert c._max_retries == 2
        assert c._retry_base_delay == 0.5

    def test_news_connector_defaults(self):
        from tollama.xai.connectors.live import HttpNewsConnector

        c = HttpNewsConnector(base_url="http://localhost:8090")
        assert c._max_retries == 2
        assert c._retry_base_delay == 0.5


# ── Phase 4.2: Rate Limit & Cache ───────────────────────────


class TestCacheTTLConfig:
    """Verify domain-specific cache TTL configuration."""

    def test_news_default_ttl(self):
        from tollama.xai.connectors.helpers import cache_ttl_for_domain

        assert cache_ttl_for_domain("news") == 300.0

    def test_financial_default_ttl(self):
        from tollama.xai.connectors.helpers import cache_ttl_for_domain

        assert cache_ttl_for_domain("financial_market") == 60.0

    def test_unknown_domain_ttl(self):
        from tollama.xai.connectors.helpers import cache_ttl_for_domain

        assert cache_ttl_for_domain("supply_chain") == 0.0

    def test_news_env_override(self):
        from tollama.xai.connectors.helpers import cache_ttl_for_domain

        with patch.dict("os.environ", {"TOLLAMA_CACHE_TTL_NEWS": "600"}):
            assert cache_ttl_for_domain("news") == 600.0

    def test_financial_env_override(self):
        from tollama.xai.connectors.helpers import cache_ttl_for_domain

        with patch.dict("os.environ", {"TOLLAMA_CACHE_TTL_FINANCIAL": "30"}):
            assert cache_ttl_for_domain("financial_market") == 30.0


class TestRateLimitHeaderExtraction:
    """Verify rate-limit header parsing from HTTP responses."""

    def test_extract_all_headers(self):
        from unittest.mock import MagicMock

        from tollama.xai.connectors.live import _extract_rate_limit_info

        resp = MagicMock()
        resp.headers = {
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Reset": "1710000000",
            "Retry-After": "30",
        }
        info = _extract_rate_limit_info(resp)
        assert info["remaining"] == 42
        assert info["limit"] == 100
        assert info["reset"] == "1710000000"
        assert info["retry_after_seconds"] == 30

    def test_extract_no_headers(self):
        from unittest.mock import MagicMock

        from tollama.xai.connectors.live import _extract_rate_limit_info

        resp = MagicMock()
        resp.headers = {}
        info = _extract_rate_limit_info(resp)
        assert info == {}

    def test_extract_partial_headers(self):
        from unittest.mock import MagicMock

        from tollama.xai.connectors.live import _extract_rate_limit_info

        resp = MagicMock()
        resp.headers = {"X-RateLimit-Remaining": "5"}
        info = _extract_rate_limit_info(resp)
        assert info == {"remaining": 5}

    def test_non_numeric_retry_after(self):
        from unittest.mock import MagicMock

        from tollama.xai.connectors.live import _extract_rate_limit_info

        resp = MagicMock()
        resp.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}
        info = _extract_rate_limit_info(resp)
        assert info["retry_after"] == "Wed, 21 Oct 2026 07:28:00 GMT"


class TestShouldRetry:
    """Verify retry decision logic."""

    def test_timeout_is_retryable(self):
        import httpx

        from tollama.xai.connectors.live import _should_retry

        exc = httpx.ReadTimeout("timeout")
        assert _should_retry(exc, attempt=0, max_retries=2) is True

    def test_429_is_retryable(self):
        from unittest.mock import MagicMock

        import httpx

        from tollama.xai.connectors.live import _should_retry

        resp = MagicMock()
        resp.status_code = 429
        exc = httpx.HTTPStatusError("rate limit", request=MagicMock(), response=resp)
        assert _should_retry(exc, attempt=0, max_retries=2) is True

    def test_500_is_retryable(self):
        from unittest.mock import MagicMock

        import httpx

        from tollama.xai.connectors.live import _should_retry

        resp = MagicMock()
        resp.status_code = 500
        exc = httpx.HTTPStatusError("server error", request=MagicMock(), response=resp)
        assert _should_retry(exc, attempt=0, max_retries=2) is True

    def test_401_not_retryable(self):
        from unittest.mock import MagicMock

        import httpx

        from tollama.xai.connectors.live import _should_retry

        resp = MagicMock()
        resp.status_code = 401
        exc = httpx.HTTPStatusError("unauthorized", request=MagicMock(), response=resp)
        assert _should_retry(exc, attempt=0, max_retries=2) is False

    def test_max_retries_exceeded(self):
        import httpx

        from tollama.xai.connectors.live import _should_retry

        exc = httpx.ReadTimeout("timeout")
        assert _should_retry(exc, attempt=2, max_retries=2) is False

    def test_generic_exception_not_retryable(self):
        from tollama.xai.connectors.live import _should_retry

        assert _should_retry(ValueError("bad"), attempt=0, max_retries=2) is False


# ── Phase 4.3: Error Taxonomy ──────────────────────────────


class TestErrorTaxonomy:
    """Verify error classification constants and helpers."""

    def test_all_error_types(self):
        from tollama.xai.connectors.errors import ALL_ERROR_TYPES

        expected = {"auth", "schema", "not_found", "rate_limit", "internal", "network", "unknown"}
        assert ALL_ERROR_TYPES == expected

    def test_auth_not_retryable(self):
        from tollama.xai.connectors.errors import AUTH

        assert AUTH.retryable is False
        assert AUTH.error_type == "auth"

    def test_network_retryable(self):
        from tollama.xai.connectors.errors import NETWORK

        assert NETWORK.retryable is True

    def test_rate_limit_retryable(self):
        from tollama.xai.connectors.errors import RATE_LIMIT

        assert RATE_LIMIT.retryable is True

    def test_schema_not_retryable(self):
        from tollama.xai.connectors.errors import SCHEMA

        assert SCHEMA.retryable is False

    def test_classify_known(self):
        from tollama.xai.connectors.errors import classify

        cat = classify("auth")
        assert cat.error_type == "auth"
        assert cat.retryable is False

    def test_classify_unknown_fallback(self):
        from tollama.xai.connectors.errors import UNKNOWN, classify

        assert classify("bizarre_error") is UNKNOWN

    def test_is_retryable_helper(self):
        from tollama.xai.connectors.errors import is_retryable

        assert is_retryable("network") is True
        assert is_retryable("auth") is False
        assert is_retryable("unknown_type") is False

    def test_error_category_frozen(self):
        from tollama.xai.connectors.errors import AUTH

        with pytest.raises(AttributeError):
            AUTH.retryable = True  # type: ignore[misc]

    def test_422_maps_to_schema(self):
        """Verify _map_http_error maps 422 to schema error type."""
        from unittest.mock import MagicMock

        import httpx

        from tollama.xai.connectors.live import _map_http_error

        resp = MagicMock()
        resp.status_code = 422
        resp.reason_phrase = "Unprocessable Entity"
        resp.headers = {}
        request = MagicMock()
        exc = httpx.HTTPStatusError("422", request=request, response=resp)
        err = _map_http_error("news", "test-id", exc)
        assert err.error.error_type == "schema"
        assert err.error.retryable is False


# ── Phase 4.4: Version Compatibility Matrix ─────────────────


class TestVersionCompatibility:
    """Verify version compatibility checking."""

    def test_known_agent_compatible(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="news_agent",
            formula_version="news-v1",
            agent_version="0.1.0",
        )
        assert result.compatible is True
        assert result.message == "OK"

    def test_unknown_formula_version(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="news_agent",
            formula_version="news-v99",
            agent_version="0.1.0",
        )
        assert result.compatible is False
        assert "unsupported" in result.message

    def test_below_min_agent_version(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="financial_agent",
            formula_version="financial-v1",
            agent_version="0.0.9",
        )
        assert result.compatible is False
        assert "below minimum" in result.message

    def test_unknown_agent_is_compatible(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="custom_agent",
            formula_version="custom-v1",
            agent_version="1.0.0",
        )
        assert result.compatible is True

    def test_degraded_version(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="_degraded",
            formula_version="degraded-v1",
            agent_version="0.1.0",
        )
        assert result.compatible is True

    def test_aggregate_version(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="_aggregate",
            formula_version="aggregate-v1",
            agent_version="0.1.0",
        )
        assert result.compatible is True

    def test_semver_parsing(self):
        from tollama.xai.connectors.versions import _parse_semver

        assert _parse_semver("1.2.3") == (1, 2, 3)
        assert _parse_semver("0.1.0") == (0, 1, 0)
        assert _parse_semver("2.0") == (2, 0)
        assert _parse_semver("") == (0,)

    def test_higher_agent_version_ok(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="news_agent",
            formula_version="news-v2",
            agent_version="1.0.0",
        )
        assert result.compatible is True

    def test_result_fields(self):
        from tollama.xai.connectors.versions import check_compatibility

        result = check_compatibility(
            agent_name="news_agent",
            formula_version="news-v1",
            agent_version="0.2.0",
        )
        assert result.agent_name == "news_agent"
        assert result.formula_version == "news-v1"
        assert result.agent_version == "0.2.0"
