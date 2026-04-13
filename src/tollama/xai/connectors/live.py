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


def _map_network_error(
    domain: str,
    identifier: str,
    exc: httpx.RequestError,
) -> ConnectorFetchError:
    """Map network/request failures to ConnectorFetchError."""
    if isinstance(exc, httpx.TimeoutException):
        message = "Connection timeout"
    else:
        message = str(exc) or "Network request failed"

    return ConnectorFetchError(
        ConnectorError(
            domain=domain,
            source_id=identifier,
            error_type="network",
            message=message,
            retryable=True,
        )
    )


def _map_request_exception(
    domain: str,
    identifier: str,
    exc: httpx.RequestError | httpx.HTTPStatusError,
) -> ConnectorFetchError:
    if isinstance(exc, httpx.RequestError):
        return _map_network_error(domain, identifier, exc)
    return _map_http_error(domain, identifier, exc)


def _should_retry(
    exc: Exception,
    attempt: int,
    max_retries: int,
) -> bool:
    """Decide if a request should be retried."""
    if attempt >= max_retries:
        return False
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _build_request_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _build_resource_url(
    base_url: str,
    endpoint: str,
    identifier: str,
) -> str:
    return f"{base_url}/{endpoint}/{identifier}"


def _request_sync_resource(
    *,
    base_url: str,
    endpoint: str,
    identifier: str,
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            _build_resource_url(base_url, endpoint, identifier),
            headers=headers,
        )
        response.raise_for_status()
    return response


async def _request_async_resource(
    *,
    base_url: str,
    endpoint: str,
    identifier: str,
    headers: dict[str, str],
    timeout: float,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            _build_resource_url(base_url, endpoint, identifier),
            headers=headers,
        )
        response.raise_for_status()
    return response


def _build_connector_result(
    *,
    response: httpx.Response,
    domain: str,
    source_id: str,
    source_type: str,
    id_key: str,
    connector_name: str,
    base_url: str,
    metadata_extras: dict[str, Any] | None = None,
) -> ConnectorResult:
    payload = response.json()
    payload.setdefault(id_key, source_id)

    metadata: dict[str, Any] = {
        "connector": connector_name,
        "base_url": base_url,
    }
    if metadata_extras:
        metadata.update(metadata_extras)

    return ConnectorResult(
        domain=domain,
        payload=payload,
        source_id=source_id,
        source_type=source_type,
        freshness_seconds=_extract_freshness(response),
        metadata=metadata,
    )


def _retrying_sync_fetch(
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
    max_retries: int,
    retry_base_delay: float,
    rate_limit_warning_label: str,
    retry_log_label: str,
) -> ConnectorResult:
    headers = _build_request_headers(api_key)

    last_exc: httpx.RequestError | httpx.HTTPStatusError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _request_sync_resource(
                base_url=base_url,
                endpoint=endpoint,
                identifier=identifier,
                headers=headers,
                timeout=timeout,
            )
            rate_info = _extract_rate_limit_info(response)
            if rate_info.get("remaining", 999) < 10:
                _log.warning(
                    "%s rate limit low: %s",
                    rate_limit_warning_label,
                    rate_info,
                )
            return _build_connector_result(
                response=response,
                domain=domain,
                source_id=identifier,
                source_type=source_type,
                id_key=id_key,
                connector_name=connector_name,
                base_url=base_url,
                metadata_extras={
                    "attempt": attempt + 1,
                    **({"rate_limit": rate_info} if rate_info else {}),
                },
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if _should_retry(exc, attempt, max_retries):
                delay = retry_base_delay * (2**attempt)
                _log.info(
                    "Retrying %s fetch %s (attempt %d, delay %.1fs)",
                    retry_log_label,
                    identifier,
                    attempt + 1,
                    delay,
                )
                time.sleep(delay)
                continue
            break

    if last_exc is not None:
        raise _map_request_exception(domain, identifier, last_exc)
    raise ConnectorFetchError(  # pragma: no cover
        ConnectorError(
            domain=domain,
            source_id=identifier,
            error_type="internal",
            message=str(last_exc),
            retryable=False,
        )
    )


def _sync_health_check(
    *,
    base_url: str,
    health_path: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout) as client:
            start = datetime.now(UTC)
            response = client.get(f"{base_url}/{health_path}")
            latency_ms = (datetime.now(UTC) - start).total_seconds() * 1000
    except (httpx.TimeoutException, httpx.ConnectError):
        return {"status": "unavailable", "latency_ms": None}

    if response.status_code == 200:
        threshold = timeout * 500
        status = "degraded" if latency_ms > threshold else "available"
    else:
        status = "unavailable"
    return {"status": status, "latency_ms": round(latency_ms, 1)}


class _HttpConnectorBase:
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


class _RetryingSyncHttpConnectorBase(_HttpConnectorBase):
    connector_name: str
    domain: str
    _endpoint: str
    _source_type: str
    _id_key: str
    _health_path: str
    _rate_limit_warning_label: str
    _retry_log_label: str

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
    ) -> None:
        super().__init__(base_url=base_url, api_key=api_key, timeout=timeout)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def health_check(self) -> dict[str, Any]:
        return _sync_health_check(
            base_url=self._base_url,
            health_path=self._health_path,
            timeout=self._timeout,
        )

    def fetch(
        self,
        identifier: str,
        context: dict[str, Any],
    ) -> ConnectorResult:
        return _retrying_sync_fetch(
            base_url=self._base_url,
            endpoint=self._endpoint,
            identifier=identifier,
            domain=self.domain,
            source_type=self._source_type,
            connector_name=self.connector_name,
            id_key=self._id_key,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
            retry_base_delay=self._retry_base_delay,
            rate_limit_warning_label=self._rate_limit_warning_label,
            retry_log_label=self._retry_log_label,
        )


class HttpFinancialConnector(_RetryingSyncHttpConnectorBase):
    """HTTP connector for financial market data feeds.

    Supports configurable timeout, retry with exponential backoff,
    and rate-limit header extraction.
    """

    connector_name = "http_financial"
    domain = "financial_market"
    _endpoint = "instruments"
    _source_type = "equity_market"
    _id_key = "instrument_id"
    _health_path = "api/v1/financial/health"
    _rate_limit_warning_label = "Financial connector"
    _retry_log_label = "financial"


class HttpNewsConnector(_RetryingSyncHttpConnectorBase):
    """HTTP connector for news data feeds.

    Supports configurable timeout, retry with exponential backoff,
    and rate-limit header extraction (NewsAPI 100 req/day limit).
    """

    connector_name = "http_news"
    domain = "news"
    _endpoint = "stories"
    _source_type = "news_feed"
    _id_key = "story_id"
    _health_path = "api/v1/news/health"
    _rate_limit_warning_label = "News connector"
    _retry_log_label = "news"


def _sync_fetch(
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
    headers = _build_request_headers(api_key)
    try:
        response = _request_sync_resource(
            base_url=base_url,
            endpoint=endpoint,
            identifier=identifier,
            headers=headers,
            timeout=timeout,
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise _map_request_exception(domain, identifier, exc)

    return _build_connector_result(
        response=response,
        domain=domain,
        source_id=identifier,
        source_type=source_type,
        id_key=id_key,
        connector_name=connector_name,
        base_url=base_url,
    )


class _SyncHttpConnectorBase(_HttpConnectorBase):
    connector_name: str
    domain: str
    _endpoint: str
    _source_type: str
    _id_key: str

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return _sync_fetch(
            base_url=self._base_url,
            endpoint=self._endpoint,
            identifier=identifier,
            domain=self.domain,
            source_type=self._source_type,
            connector_name=self.connector_name,
            id_key=self._id_key,
            api_key=self._api_key,
            timeout=self._timeout,
        )


class HttpSupplyChainConnector(_SyncHttpConnectorBase):
    """HTTP connector for supply chain data feeds."""

    connector_name = "http_supply_chain"
    domain = "supply_chain"
    _endpoint = "networks"
    _source_type = "supply_chain_iot"
    _id_key = "network_id"


class HttpGeopoliticalConnector(_SyncHttpConnectorBase):
    """HTTP connector for geopolitical data feeds."""

    connector_name = "http_geopolitical"
    domain = "geopolitical"
    _endpoint = "regions"
    _source_type = "country_risk"
    _id_key = "region_id"


class HttpRegulatoryConnector(_SyncHttpConnectorBase):
    """HTTP connector for regulatory data feeds."""

    connector_name = "http_regulatory"
    domain = "regulatory"
    _endpoint = "jurisdictions"
    _source_type = "compliance"
    _id_key = "jurisdiction_id"


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
    headers = _build_request_headers(api_key)
    try:
        response = await _request_async_resource(
            base_url=base_url,
            endpoint=endpoint,
            identifier=identifier,
            headers=headers,
            timeout=timeout,
        )
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise _map_request_exception(domain, identifier, exc)

    return _build_connector_result(
        response=response,
        domain=domain,
        source_id=identifier,
        source_type=source_type,
        id_key=id_key,
        connector_name=connector_name,
        base_url=base_url,
    )


class _AsyncHttpConnectorBase(_HttpConnectorBase):
    connector_name: str
    domain: str
    _endpoint: str
    _source_type: str
    _id_key: str

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return await _async_fetch(
            base_url=self._base_url,
            endpoint=self._endpoint,
            identifier=identifier,
            domain=self.domain,
            source_type=self._source_type,
            connector_name=self.connector_name,
            id_key=self._id_key,
            api_key=self._api_key,
            timeout=self._timeout,
        )


class AsyncHttpFinancialConnector(_AsyncHttpConnectorBase):
    """Async HTTP connector for financial market data feeds."""

    connector_name = "async_http_financial"
    domain = "financial_market"
    _endpoint = "instruments"
    _source_type = "equity_market"
    _id_key = "instrument_id"


class AsyncHttpNewsConnector(_AsyncHttpConnectorBase):
    """Async HTTP connector for news data feeds."""

    connector_name = "async_http_news"
    domain = "news"
    _endpoint = "stories"
    _source_type = "news_feed"
    _id_key = "story_id"


class AsyncHttpSupplyChainConnector(_AsyncHttpConnectorBase):
    """Async HTTP connector for supply chain data feeds."""

    connector_name = "async_http_supply_chain"
    domain = "supply_chain"
    _endpoint = "networks"
    _source_type = "supply_chain_iot"
    _id_key = "network_id"


class AsyncHttpGeopoliticalConnector(_AsyncHttpConnectorBase):
    """Async HTTP connector for geopolitical data feeds."""

    connector_name = "async_http_geopolitical"
    domain = "geopolitical"
    _endpoint = "regions"
    _source_type = "country_risk"
    _id_key = "region_id"


class AsyncHttpRegulatoryConnector(_AsyncHttpConnectorBase):
    """Async HTTP connector for regulatory data feeds."""

    connector_name = "async_http_regulatory"
    domain = "regulatory"
    _endpoint = "jurisdictions"
    _source_type = "compliance"
    _id_key = "jurisdiction_id"


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
