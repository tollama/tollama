"""
tollama.xai.connectors.helpers — Factory functions for connector setup.

Environment variables for live connector mode:
    TOLLAMA_USE_LIVE_CONNECTORS  — set to "true" to use live News-Agent / Financial-Agent
    NEWS_AGENT_URL               — News-Agent base URL (default: http://localhost:8090)
    FINANCIAL_AGENT_URL          — Financial-Agent base URL (default: http://localhost:8091)
    NEWS_AGENT_API_KEY           — optional Bearer token for News-Agent
    FINANCIAL_AGENT_API_KEY      — optional Bearer token for Financial-Agent

Timeout and retry configuration:
    TOLLAMA_CONNECTOR_TIMEOUT    — per-request timeout in seconds (default: 10.0)
    TOLLAMA_CONNECTOR_MAX_RETRIES — max retry attempts for retryable errors (default: 2)
    TOLLAMA_CONNECTOR_RETRY_BASE_DELAY — base delay between retries in seconds (default: 0.5)

Domain-specific cache TTL:
    TOLLAMA_CACHE_TTL_NEWS       — cache TTL for news domain in seconds (default: 300)
    TOLLAMA_CACHE_TTL_FINANCIAL  — cache TTL for financial domain in seconds (default: 60)
"""

from __future__ import annotations

import os

from tollama.xai.connectors.assembler import AsyncPayloadAssembler, PayloadAssembler
from tollama.xai.connectors.registry import AsyncConnectorRegistry, ConnectorRegistry
from tollama.xai.connectors.stubs import (
    MockFinancialConnector,
    MockGeopoliticalConnector,
    MockNewsConnector,
    MockRegulatoryConnector,
    MockSupplyChainConnector,
)

# Default agent URLs
_NEWS_AGENT_URL_DEFAULT = "http://localhost:8090"
_FINANCIAL_AGENT_URL_DEFAULT = "http://localhost:8091"
_FALLBACK_URL_DEFAULT = "http://localhost:8080"

# Default connector settings
_CONNECTOR_TIMEOUT_DEFAULT = 10.0
_CONNECTOR_MAX_RETRIES_DEFAULT = 2
_CONNECTOR_RETRY_BASE_DELAY_DEFAULT = 0.5

# Domain-specific cache TTL defaults (seconds)
CACHE_TTL_NEWS = 300.0  # 5 minutes — news changes less frequently
CACHE_TTL_FINANCIAL = 60.0  # 1 minute — financial data more volatile


def _use_live_connectors() -> bool:
    """Check if live connectors should be used instead of mocks."""
    return os.environ.get(
        "TOLLAMA_USE_LIVE_CONNECTORS", "false",
    ).lower() in ("true", "1", "yes")


def _news_agent_url() -> str:
    return os.environ.get("NEWS_AGENT_URL", _NEWS_AGENT_URL_DEFAULT)


def _financial_agent_url() -> str:
    return os.environ.get("FINANCIAL_AGENT_URL", _FINANCIAL_AGENT_URL_DEFAULT)


def _news_agent_api_key() -> str | None:
    return os.environ.get("NEWS_AGENT_API_KEY")


def _financial_agent_api_key() -> str | None:
    return os.environ.get("FINANCIAL_AGENT_API_KEY")


def _connector_timeout() -> float:
    """Per-request timeout in seconds."""
    raw = os.environ.get("TOLLAMA_CONNECTOR_TIMEOUT", "")
    if raw:
        try:
            return max(float(raw), 1.0)
        except ValueError:
            pass
    return _CONNECTOR_TIMEOUT_DEFAULT


def _connector_max_retries() -> int:
    """Max retry attempts for retryable errors."""
    raw = os.environ.get("TOLLAMA_CONNECTOR_MAX_RETRIES", "")
    if raw:
        try:
            return max(int(raw), 0)
        except ValueError:
            pass
    return _CONNECTOR_MAX_RETRIES_DEFAULT


def _connector_retry_base_delay() -> float:
    """Base delay between retries in seconds."""
    raw = os.environ.get("TOLLAMA_CONNECTOR_RETRY_BASE_DELAY", "")
    if raw:
        try:
            return max(float(raw), 0.0)
        except ValueError:
            pass
    return _CONNECTOR_RETRY_BASE_DELAY_DEFAULT


def cache_ttl_for_domain(domain: str) -> float:
    """Return the configured cache TTL for a given domain.

    Reads from environment variables first, falls back to built-in defaults.
    """
    if domain == "news":
        raw = os.environ.get("TOLLAMA_CACHE_TTL_NEWS", "")
        return float(raw) if raw else CACHE_TTL_NEWS
    if domain == "financial_market":
        raw = os.environ.get("TOLLAMA_CACHE_TTL_FINANCIAL", "")
        return float(raw) if raw else CACHE_TTL_FINANCIAL
    return 0.0


def build_default_connector_registry() -> ConnectorRegistry:
    """Build a registry with connectors.

    When TOLLAMA_USE_LIVE_CONNECTORS=true, news and financial_market domains use
    live HTTP connectors pointing at the external agents.  All other domains
    always use mock connectors.

    Timeout, retry, and cache settings are read from environment variables.
    """
    registry = ConnectorRegistry()
    timeout = _connector_timeout()
    max_retries = _connector_max_retries()
    retry_base_delay = _connector_retry_base_delay()

    if _use_live_connectors():
        from tollama.xai.connectors.live import HttpFinancialConnector, HttpNewsConnector

        registry.register(
            HttpFinancialConnector(
                base_url=_financial_agent_url(),
                api_key=_financial_agent_api_key(),
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
        )
        registry.register(
            HttpNewsConnector(
                base_url=_news_agent_url(),
                api_key=_news_agent_api_key(),
                timeout=timeout,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay,
            )
        )
    else:
        registry.register(MockFinancialConnector())
        registry.register(MockNewsConnector())

    # Other domains always use mocks (no live agents yet)
    registry.register(MockSupplyChainConnector())
    registry.register(MockGeopoliticalConnector())
    registry.register(MockRegulatoryConnector())
    return registry


def build_default_assembler() -> PayloadAssembler:
    """Build an assembler with the default connector registry."""
    return PayloadAssembler(build_default_connector_registry())


def build_default_async_connector_registry() -> AsyncConnectorRegistry:
    """Build an async registry with connectors.

    When TOLLAMA_USE_LIVE_CONNECTORS=true, news and financial_market domains use
    async HTTP connectors pointing at the external agents.  All other domains
    use async HTTP connectors against the fallback URL.
    """
    from tollama.xai.connectors.live import (
        AsyncHttpFinancialConnector,
        AsyncHttpGeopoliticalConnector,
        AsyncHttpNewsConnector,
        AsyncHttpRegulatoryConnector,
        AsyncHttpSupplyChainConnector,
    )

    registry = AsyncConnectorRegistry()
    timeout = _connector_timeout()

    if _use_live_connectors():
        registry.register(
            AsyncHttpFinancialConnector(
                base_url=_financial_agent_url(),
                api_key=_financial_agent_api_key(),
                timeout=timeout,
            )
        )
        registry.register(
            AsyncHttpNewsConnector(
                base_url=_news_agent_url(),
                api_key=_news_agent_api_key(),
                timeout=timeout,
            )
        )
    else:
        registry.register(
            AsyncHttpFinancialConnector(base_url=_FALLBACK_URL_DEFAULT),
        )
        registry.register(
            AsyncHttpNewsConnector(base_url=_FALLBACK_URL_DEFAULT),
        )

    # Other domains always use fallback URL
    registry.register(
        AsyncHttpSupplyChainConnector(base_url=_FALLBACK_URL_DEFAULT),
    )
    registry.register(
        AsyncHttpGeopoliticalConnector(base_url=_FALLBACK_URL_DEFAULT),
    )
    registry.register(
        AsyncHttpRegulatoryConnector(base_url=_FALLBACK_URL_DEFAULT),
    )
    return registry


def build_default_async_assembler() -> AsyncPayloadAssembler:
    """Build an async assembler with the default async connector registry."""
    return AsyncPayloadAssembler(build_default_async_connector_registry())


__all__ = [
    "CACHE_TTL_FINANCIAL",
    "CACHE_TTL_NEWS",
    "build_default_assembler",
    "build_default_async_assembler",
    "build_default_async_connector_registry",
    "build_default_connector_registry",
    "cache_ttl_for_domain",
]
