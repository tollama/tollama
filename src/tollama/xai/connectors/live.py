"""
tollama.xai.connectors.live — HTTP-based live feed connectors.

Supports configurable timeout, retry with exponential backoff, and
rate-limit header extraction for production resilience.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from tollama.xai.connectors.protocol import (
    ConnectorError,
    ConnectorFetchError,
    ConnectorResult,
)

_log = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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


def _extract_rate_limit_info(
    response: httpx.Response,
) -> dict[str, Any]:
    """Extract rate-limit headers from response.

    Looks for standard ``X-RateLimit-*`` and ``Retry-After`` headers.
    Returns a dict with available fields (empty dict if none found).
    """
    info: dict[str, Any] = {}
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is not None:
        try:
            info["remaining"] = int(remaining)
        except ValueError:
            pass
    limit = response.headers.get("X-RateLimit-Limit")
    if limit is not None:
        try:
            info["limit"] = int(limit)
        except ValueError:
            pass
    reset_at = response.headers.get("X-RateLimit-Reset")
    if reset_at is not None:
        info["reset"] = reset_at
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            info["retry_after_seconds"] = int(retry_after)
        except ValueError:
            info["retry_after"] = retry_after
    return info


def _map_http_error(
    domain: str,
    identifier: str,
    exc: httpx.HTTPStatusError,
) -> ConnectorFetchError:
    """Map HTTP status errors to ConnectorFetchError."""
    status = exc.response.status_code
    if status in (401, 403):
        error_type = "auth"
        retryable = False
    elif status == 422:
        error_type = "schema"
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

    detail: dict[str, Any] = {"status_code": status}
    rate_info = _extract_rate_limit_info(exc.response)
    if rate_info:
        detail["rate_limit"] = rate_info

    return ConnectorFetchError(
        ConnectorError(
            domain=domain,
            source_id=identifier,
            error_type=error_type,
            message=f"HTTP {status}: {exc.response.reason_phrase}",
            retryable=retryable,
            detail=detail,
        )
    )


def _should_retry(
    exc: Exception,
    attempt: int,
    max_retries: int,
) -> bool:
    """Decide if a request should be retried."""
    if attempt >= max_retries:
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


class HttpFinancialConnector:
    """HTTP connector for financial market data feeds.

    Supports configurable timeout, retry with exponential backoff,
    and rate-limit header extraction.
    """

    connector_name = "http_financial"
    domain = "financial_market"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def health_check(self) -> dict[str, Any]:
        """Check health of the remote Financial-Agent service.

        Returns dict with 'status' ('available', 'degraded', 'unavailable')
        and 'latency_ms'.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                start = datetime.now(UTC)
                resp = client.get(
                    f"{self._base_url}/api/v1/financial/health",
                )
                latency_ms = (
                    (datetime.now(UTC) - start).total_seconds() * 1000
                )
                if resp.status_code == 200:
                    threshold = self._timeout * 500
                    status = "degraded" if latency_ms > threshold else "available"
                else:
                    status = "unavailable"
        except (httpx.TimeoutException, httpx.ConnectError):
            return {"status": "unavailable", "latency_ms": None}
        return {"status": status, "latency_ms": round(latency_ms, 1)}

    def fetch(
        self, identifier: str, context: dict[str, Any],
    ) -> ConnectorResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(
                        f"{self._base_url}/instruments/{identifier}",
                        headers=headers,
                    )
                    response.raise_for_status()
                # Success — extract rate-limit info for logging
                rate_info = _extract_rate_limit_info(response)
                if rate_info.get("remaining", 999) < 10:
                    _log.warning(
                        "Financial connector rate limit low: %s",
                        rate_info,
                    )
                data = response.json()
                data.setdefault("instrument_id", identifier)
                return ConnectorResult(
                    domain=self.domain,
                    payload=data,
                    source_id=identifier,
                    source_type="equity_market",
                    freshness_seconds=_extract_freshness(response),
                    metadata={
                        "connector": self.connector_name,
                        "base_url": self._base_url,
                        "attempt": attempt + 1,
                        **({"rate_limit": rate_info} if rate_info else {}),
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if _should_retry(exc, attempt, self._max_retries):
                    delay = self._retry_base_delay * (2 ** attempt)
                    _log.info(
                        "Retrying financial fetch %s (attempt %d, delay %.1fs)",
                        identifier, attempt + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                break

        # Exhausted retries — raise appropriate error
        if isinstance(last_exc, (httpx.TimeoutException, httpx.ConnectError)):
            raise ConnectorFetchError(
                ConnectorError(
                    domain=self.domain,
                    source_id=identifier,
                    error_type="network",
                    message=(
                        "Connection timeout"
                        if isinstance(last_exc, httpx.TimeoutException)
                        else str(last_exc)
                    ),
                    retryable=True,
                )
            )
        if isinstance(last_exc, httpx.HTTPStatusError):
            raise _map_http_error(self.domain, identifier, last_exc)
        raise ConnectorFetchError(  # pragma: no cover
            ConnectorError(
                domain=self.domain,
                source_id=identifier,
                error_type="internal",
                message=str(last_exc),
                retryable=False,
            )
        )


class HttpNewsConnector:
    """HTTP connector for news data feeds.

    Supports configurable timeout, retry with exponential backoff,
    and rate-limit header extraction (NewsAPI 100 req/day limit).
    """

    connector_name = "http_news"
    domain = "news"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def health_check(self) -> dict[str, Any]:
        """Check health of the remote News-Agent service.

        Returns dict with 'status' ('available', 'degraded', 'unavailable')
        and 'latency_ms'.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                start = datetime.now(UTC)
                resp = client.get(
                    f"{self._base_url}/api/v1/news/health",
                )
                latency_ms = (
                    (datetime.now(UTC) - start).total_seconds() * 1000
                )
                if resp.status_code == 200:
                    threshold = self._timeout * 500
                    status = "degraded" if latency_ms > threshold else "available"
                else:
                    status = "unavailable"
        except (httpx.TimeoutException, httpx.ConnectError):
            return {"status": "unavailable", "latency_ms": None}
        return {"status": status, "latency_ms": round(latency_ms, 1)}

    def fetch(
        self, identifier: str, context: dict[str, Any],
    ) -> ConnectorResult:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(
                        f"{self._base_url}/stories/{identifier}",
                        headers=headers,
                    )
                    response.raise_for_status()
                rate_info = _extract_rate_limit_info(response)
                if rate_info.get("remaining", 999) < 10:
                    _log.warning(
                        "News connector rate limit low: %s", rate_info,
                    )
                data = response.json()
                data.setdefault("story_id", identifier)
                return ConnectorResult(
                    domain=self.domain,
                    payload=data,
                    source_id=identifier,
                    source_type="news_feed",
                    freshness_seconds=_extract_freshness(response),
                    metadata={
                        "connector": self.connector_name,
                        "base_url": self._base_url,
                        "attempt": attempt + 1,
                        **({"rate_limit": rate_info} if rate_info else {}),
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if _should_retry(exc, attempt, self._max_retries):
                    delay = self._retry_base_delay * (2 ** attempt)
                    _log.info(
                        "Retrying news fetch %s (attempt %d, delay %.1fs)",
                        identifier, attempt + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                break

        if isinstance(last_exc, (httpx.TimeoutException, httpx.ConnectError)):
            raise ConnectorFetchError(
                ConnectorError(
                    domain=self.domain,
                    source_id=identifier,
                    error_type="network",
                    message=(
                        "Connection timeout"
                        if isinstance(last_exc, httpx.TimeoutException)
                        else str(last_exc)
                    ),
                    retryable=True,
                )
            )
        if isinstance(last_exc, httpx.HTTPStatusError):
            raise _map_http_error(self.domain, identifier, last_exc)
        raise ConnectorFetchError(  # pragma: no cover
            ConnectorError(
                domain=self.domain,
                source_id=identifier,
                error_type="internal",
                message=str(last_exc),
                retryable=False,
            )
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


class HttpGeopoliticalConnector:
    """HTTP connector for geopolitical data feeds."""

    connector_name = "http_geopolitical"
    domain = "geopolitical"

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
                    f"{self._base_url}/regions/{identifier}",
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
        data.setdefault("region_id", identifier)
        return ConnectorResult(
            domain=self.domain,
            payload=data,
            source_id=identifier,
            source_type="country_risk",
            freshness_seconds=_extract_freshness(response),
            metadata={"connector": self.connector_name, "base_url": self._base_url},
        )


class HttpRegulatoryConnector:
    """HTTP connector for regulatory data feeds."""

    connector_name = "http_regulatory"
    domain = "regulatory"

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
                    f"{self._base_url}/jurisdictions/{identifier}",
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
        data.setdefault("jurisdiction_id", identifier)
        return ConnectorResult(
            domain=self.domain,
            payload=data,
            source_id=identifier,
            source_type="compliance",
            freshness_seconds=_extract_freshness(response),
            metadata={"connector": self.connector_name, "base_url": self._base_url},
        )


# ──────────────────────────────────────────────────────────────
# Async Connectors
# ──────────────────────────────────────────────────────────────


async def _async_fetch(
    *,
    base_url: str,
    endpoint: str,
    identifier: str,
    domain: str,
    source_type: str,
    connector_name: str,
    id_key: str,
    api_key: str | None,
    timeout: float,
) -> ConnectorResult:
    """Shared async fetch logic for all async connectors."""
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{base_url}/{endpoint}/{identifier}",
                headers=headers,
            )
            response.raise_for_status()
    except httpx.TimeoutException:
        raise ConnectorFetchError(
            ConnectorError(
                domain=domain,
                source_id=identifier,
                error_type="network",
                message="Connection timeout",
                retryable=True,
            )
        )
    except httpx.HTTPStatusError as exc:
        raise _map_http_error(domain, identifier, exc)

    data = response.json()
    data.setdefault(id_key, identifier)
    return ConnectorResult(
        domain=domain,
        payload=data,
        source_id=identifier,
        source_type=source_type,
        freshness_seconds=_extract_freshness(response),
        metadata={"connector": connector_name, "base_url": base_url},
    )


class AsyncHttpFinancialConnector:
    """Async HTTP connector for financial market data feeds."""

    connector_name = "async_http_financial"
    domain = "financial_market"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url, endpoint="instruments", identifier=identifier,
            domain=self.domain, source_type="equity_market", connector_name=self.connector_name,
            id_key="instrument_id", api_key=self._api_key, timeout=self._timeout,
        )


class AsyncHttpNewsConnector:
    """Async HTTP connector for news data feeds."""

    connector_name = "async_http_news"
    domain = "news"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url, endpoint="stories", identifier=identifier,
            domain=self.domain, source_type="news_feed", connector_name=self.connector_name,
            id_key="story_id", api_key=self._api_key, timeout=self._timeout,
        )


class AsyncHttpSupplyChainConnector:
    """Async HTTP connector for supply chain data feeds."""

    connector_name = "async_http_supply_chain"
    domain = "supply_chain"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url, endpoint="networks", identifier=identifier,
            domain=self.domain, source_type="supply_chain_iot", connector_name=self.connector_name,
            id_key="network_id", api_key=self._api_key, timeout=self._timeout,
        )


class AsyncHttpGeopoliticalConnector:
    """Async HTTP connector for geopolitical data feeds."""

    connector_name = "async_http_geopolitical"
    domain = "geopolitical"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url, endpoint="regions", identifier=identifier,
            domain=self.domain, source_type="country_risk", connector_name=self.connector_name,
            id_key="region_id", api_key=self._api_key, timeout=self._timeout,
        )


class AsyncHttpRegulatoryConnector:
    """Async HTTP connector for regulatory data feeds."""

    connector_name = "async_http_regulatory"
    domain = "regulatory"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url, endpoint="jurisdictions", identifier=identifier,
            domain=self.domain, source_type="compliance", connector_name=self.connector_name,
            id_key="jurisdiction_id", api_key=self._api_key, timeout=self._timeout,
        )


__all__ = [
    "AsyncHttpFinancialConnector",
    "AsyncHttpGeopoliticalConnector",
    "AsyncHttpNewsConnector",
    "AsyncHttpRegulatoryConnector",
    "AsyncHttpSupplyChainConnector",
    "HttpFinancialConnector",
    "HttpGeopoliticalConnector",
    "HttpNewsConnector",
    "HttpRegulatoryConnector",
    "HttpSupplyChainConnector",
]
