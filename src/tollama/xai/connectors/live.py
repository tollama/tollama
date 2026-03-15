"""
tollama.xai.connectors.live — HTTP-based live feed connectors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from tollama.xai.connectors.protocol import (
    ConnectorError,
    ConnectorFetchError,
    ConnectorResult,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_freshness(response: httpx.Response) -> float | None:
    """Extract data freshness from response headers if available."""
    age = response.headers.get("Age")
    if age is not None:
        try:
            return float(age)
        except (TypeError, ValueError):
            pass
    x_data_age = response.headers.get("X-Data-Age")
    if x_data_age is not None:
        try:
            return float(x_data_age)
        except (TypeError, ValueError):
            pass
    return None


def _map_http_error(
    domain: str,
    identifier: str,
    exc: httpx.HTTPStatusError,
) -> ConnectorFetchError:
    """Map HTTP status errors to ConnectorFetchError."""
    status = exc.response.status_code
    if status == 401 or status == 403:
        error_type = "auth"
        retryable = False
    elif status == 429:
        error_type = "rate_limit"
        retryable = True
    elif status == 404:
        error_type = "not_found"
        retryable = False
    else:
        error_type = "internal"
        retryable = status >= 500
    return ConnectorFetchError(
        ConnectorError(
            domain=domain,
            source_id=identifier,
            error_type=error_type,
            message=f"HTTP {status}: {exc.response.reason_phrase}",
            retryable=retryable,
            detail={"status_code": status},
        )
    )


class HttpFinancialConnector:
    """HTTP connector for financial market data feeds."""

    connector_name = "http_financial"
    domain = "financial_market"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    f"{self._base_url}/instruments/{identifier}",
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise ConnectorFetchError(
                ConnectorError(
                    domain=self.domain,
                    source_id=identifier,
                    error_type="network",
                    message="Connection timeout",
                    retryable=True,
                )
            )
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(self.domain, identifier, exc)

        data = response.json()
        data.setdefault("instrument_id", identifier)
        return ConnectorResult(
            domain=self.domain,
            payload=data,
            source_id=identifier,
            source_type="equity_market",
            freshness_seconds=_extract_freshness(response),
            metadata={"connector": self.connector_name, "base_url": self._base_url},
        )


class HttpNewsConnector:
    """HTTP connector for news data feeds."""

    connector_name = "http_news"
    domain = "news"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    f"{self._base_url}/stories/{identifier}",
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise ConnectorFetchError(
                ConnectorError(
                    domain=self.domain,
                    source_id=identifier,
                    error_type="network",
                    message="Connection timeout",
                    retryable=True,
                )
            )
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(self.domain, identifier, exc)

        data = response.json()
        data.setdefault("story_id", identifier)
        return ConnectorResult(
            domain=self.domain,
            payload=data,
            source_id=identifier,
            source_type="news_feed",
            freshness_seconds=_extract_freshness(response),
            metadata={"connector": self.connector_name, "base_url": self._base_url},
        )


class HttpSupplyChainConnector:
    """HTTP connector for supply chain data feeds."""

    connector_name = "http_supply_chain"
    domain = "supply_chain"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    f"{self._base_url}/networks/{identifier}",
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise ConnectorFetchError(
                ConnectorError(
                    domain=self.domain,
                    source_id=identifier,
                    error_type="network",
                    message="Connection timeout",
                    retryable=True,
                )
            )
        except httpx.HTTPStatusError as exc:
            raise _map_http_error(self.domain, identifier, exc)

        data = response.json()
        data.setdefault("network_id", identifier)
        return ConnectorResult(
            domain=self.domain,
            payload=data,
            source_id=identifier,
            source_type="supply_chain_iot",
            freshness_seconds=_extract_freshness(response),
            metadata={"connector": self.connector_name, "base_url": self._base_url},
        )


__all__ = [
    "HttpFinancialConnector",
    "HttpNewsConnector",
    "HttpSupplyChainConnector",
]
