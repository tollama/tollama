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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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
_TRUE_ENV_VALUES = frozenset({"true", "1", "yes"})
_CACHE_TTL_BY_DOMAIN = {
    "news": ("TOLLAMA_CACHE_TTL_NEWS", CACHE_TTL_NEWS),
    "financial_market": ("TOLLAMA_CACHE_TTL_FINANCIAL", CACHE_TTL_FINANCIAL),
}


@dataclass(frozen=True)
class _LiveConnectorSettings:
    timeout: float
    max_retries: int
    retry_base_delay: float


def _env_float_with_fallback(
    name: str,
    *,
    default: float,
    minimum: float | None = None,
) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(value, minimum) if minimum is not None else value


def _env_int_with_fallback(
    name: str,
    *,
    default: int,
    minimum: int | None = None,
) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum) if minimum is not None else value


def _register_connectors(
    registry: ConnectorRegistry | AsyncConnectorRegistry,
    connectors: Iterable[Any],
) -> None:
    for connector in connectors:
        registry.register(connector)


def _use_live_connectors() -> bool:
    """Check if live connectors should be used instead of mocks."""
    return (
        os.environ.get(
            "TOLLAMA_USE_LIVE_CONNECTORS",
            "false",
        ).lower()
        in _TRUE_ENV_VALUES
    )


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
    return _env_float_with_fallback(
        "TOLLAMA_CONNECTOR_TIMEOUT",
        default=_CONNECTOR_TIMEOUT_DEFAULT,
        minimum=1.0,
    )


def _connector_max_retries() -> int:
    """Max retry attempts for retryable errors."""
    return _env_int_with_fallback(
        "TOLLAMA_CONNECTOR_MAX_RETRIES",
        default=_CONNECTOR_MAX_RETRIES_DEFAULT,
        minimum=0,
    )


def _connector_retry_base_delay() -> float:
    """Base delay between retries in seconds."""
    return _env_float_with_fallback(
        "TOLLAMA_CONNECTOR_RETRY_BASE_DELAY",
        default=_CONNECTOR_RETRY_BASE_DELAY_DEFAULT,
        minimum=0.0,
    )


def _live_connector_settings() -> _LiveConnectorSettings:
    return _LiveConnectorSettings(
        timeout=_connector_timeout(),
        max_retries=_connector_max_retries(),
        retry_base_delay=_connector_retry_base_delay(),
    )


def _sync_primary_connectors() -> tuple[Any, ...]:
    if _use_live_connectors():
        from tollama.xai.connectors.live import HttpFinancialConnector, HttpNewsConnector

        settings = _live_connector_settings()
        return (
            HttpFinancialConnector(
                base_url=_financial_agent_url(),
                api_key=_financial_agent_api_key(),
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                retry_base_delay=settings.retry_base_delay,
            ),
            HttpNewsConnector(
                base_url=_news_agent_url(),
                api_key=_news_agent_api_key(),
                timeout=settings.timeout,
                max_retries=settings.max_retries,
                retry_base_delay=settings.retry_base_delay,
            ),
        )
    return (MockFinancialConnector(), MockNewsConnector())


def _shared_mock_connectors() -> tuple[Any, ...]:
    return (
        MockSupplyChainConnector(),
        MockGeopoliticalConnector(),
        MockRegulatoryConnector(),
    )


def _async_connectors() -> tuple[Any, ...]:
    from tollama.xai.connectors.live import (
        AsyncHttpFinancialConnector,
        AsyncHttpGeopoliticalConnector,
        AsyncHttpNewsConnector,
        AsyncHttpRegulatoryConnector,
        AsyncHttpSupplyChainConnector,
    )

    timeout = _connector_timeout()
    if _use_live_connectors():
        primary_connectors = (
            AsyncHttpFinancialConnector(
                base_url=_financial_agent_url(),
                api_key=_financial_agent_api_key(),
                timeout=timeout,
            ),
            AsyncHttpNewsConnector(
                base_url=_news_agent_url(),
                api_key=_news_agent_api_key(),
                timeout=timeout,
            ),
        )
    else:
        primary_connectors = (
            AsyncHttpFinancialConnector(base_url=_FALLBACK_URL_DEFAULT),
            AsyncHttpNewsConnector(base_url=_FALLBACK_URL_DEFAULT),
        )

    return primary_connectors + (
        AsyncHttpSupplyChainConnector(base_url=_FALLBACK_URL_DEFAULT),
        AsyncHttpGeopoliticalConnector(base_url=_FALLBACK_URL_DEFAULT),
        AsyncHttpRegulatoryConnector(base_url=_FALLBACK_URL_DEFAULT),
    )


def cache_ttl_for_domain(domain: str) -> float:
    """Return the configured cache TTL for a given domain.

    Reads from environment variables first, falls back to built-in defaults.
    """
    ttl_config = _CACHE_TTL_BY_DOMAIN.get(domain)
    if ttl_config is None:
        return 0.0
    env_name, default = ttl_config
    raw = os.environ.get(env_name, "")
    return float(raw) if raw else default


def build_default_connector_registry() -> ConnectorRegistry:
    """Build a registry with connectors.

    When TOLLAMA_USE_LIVE_CONNECTORS=true, news and financial_market domains use
    live HTTP connectors pointing at the external agents.  All other domains
    always use mock connectors.

    Timeout, retry, and cache settings are read from environment variables.
    """
    registry = ConnectorRegistry()
    _register_connectors(registry, _sync_primary_connectors())
    _register_connectors(registry, _shared_mock_connectors())
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
    registry = AsyncConnectorRegistry()
    _register_connectors(registry, _async_connectors())
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
