"""Prometheus metrics helpers for the tollama daemon."""

from __future__ import annotations

import threading
from collections.abc import Callable
from time import perf_counter

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_METRICS_IMPORT_HINT = (
    'prometheus-client is not installed. Install with: pip install "tollama[metrics]"'
)
_PROMETHEUS_CONTENT_TYPE_FALLBACK = "text/plain; version=0.0.4; charset=utf-8"

try:  # pragma: no branch - import-time optional dependency handling
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by monkeypatch in tests
    CONTENT_TYPE_LATEST = _PROMETHEUS_CONTENT_TYPE_FALLBACK
    CollectorRegistry = Counter = Gauge = Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    _PROMETHEUS_AVAILABLE = False

_FORECAST_ENDPOINT_LABELS = {
    "/api/forecast": "api_forecast",
    "/v1/forecast": "v1_forecast",
    "/api/auto-forecast": "api_auto_forecast",
    "/api/compare": "api_compare",
}
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class PrometheusMetrics:
    """Encapsulate all Prometheus collectors used by the daemon."""

    def __init__(
        self,
        *,
        get_loaded_models: Callable[[], int],
        get_runner_restarts: Callable[[], int],
    ) -> None:
        if not _PROMETHEUS_AVAILABLE:  # pragma: no cover - guarded by factory
            raise RuntimeError(_METRICS_IMPORT_HINT)

        self.registry = CollectorRegistry(auto_describe=True)
        self._forecast_requests_total = Counter(
            "tollama_forecast_requests_total",
            "Total forecast-like requests handled by the daemon.",
            labelnames=("endpoint", "status_class"),
            registry=self.registry,
        )
        self._forecast_latency_seconds = Histogram(
            "tollama_forecast_latency_seconds",
            "Latency for forecast-like requests in seconds.",
            labelnames=("endpoint", "status_class"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._models_loaded = Gauge(
            "tollama_models_loaded",
            "Currently loaded models tracked by the daemon.",
            registry=self.registry,
        )
        self._runner_restarts_total = Counter(
            "tollama_runner_restarts_total",
            "Total runner restarts observed since daemon startup.",
            registry=self.registry,
        )
        self._get_loaded_models = get_loaded_models
        self._get_runner_restarts = get_runner_restarts
        self._last_restart_total = 0
        self._restart_lock = threading.Lock()

    def observe_forecast_request(
        self,
        *,
        endpoint: str,
        status_code: int,
        duration_s: float,
    ) -> None:
        """Record request counters and latency for one handled request."""
        status_class = _status_class(status_code)
        duration = duration_s if duration_s >= 0.0 else 0.0
        self._forecast_requests_total.labels(
            endpoint=endpoint,
            status_class=status_class,
        ).inc()
        self._forecast_latency_seconds.labels(
            endpoint=endpoint,
            status_class=status_class,
        ).observe(duration)

    def render_latest(self) -> bytes:
        """Collect current gauge snapshots and render Prometheus payload."""
        loaded_models = max(self._get_loaded_models(), 0)
        self._models_loaded.set(float(loaded_models))
        self._observe_runner_restarts(max(self._get_runner_restarts(), 0))
        if generate_latest is None:  # pragma: no cover - guarded by constructor
            return b""
        return generate_latest(self.registry)

    def _observe_runner_restarts(self, total: int) -> None:
        with self._restart_lock:
            if total < self._last_restart_total:
                # Supervisor lifecycle was reset; treat it as a new baseline.
                self._last_restart_total = total
                return

            delta = total - self._last_restart_total
            if delta > 0:
                self._runner_restarts_total.inc(delta)
            self._last_restart_total = total


class ForecastMetricsMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records latency/counters for forecast-like routes."""

    def __init__(self, app, *, metrics: PrometheusMetrics) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._metrics = metrics

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        endpoint = _FORECAST_ENDPOINT_LABELS.get(request.url.path)
        if endpoint is None:
            return await call_next(request)

        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self._metrics.observe_forecast_request(
                endpoint=endpoint,
                status_code=500,
                duration_s=perf_counter() - started_at,
            )
            raise

        self._metrics.observe_forecast_request(
            endpoint=endpoint,
            status_code=response.status_code,
            duration_s=perf_counter() - started_at,
        )
        return response


def create_prometheus_metrics(
    *,
    get_loaded_models: Callable[[], int],
    get_runner_restarts: Callable[[], int],
) -> PrometheusMetrics | None:
    """Create a metrics collector set when Prometheus support is available."""
    if not _PROMETHEUS_AVAILABLE:
        return None
    return PrometheusMetrics(
        get_loaded_models=get_loaded_models,
        get_runner_restarts=get_runner_restarts,
    )


def metrics_available() -> bool:
    """Return whether ``prometheus_client`` is available in this environment."""
    return _PROMETHEUS_AVAILABLE


def metrics_content_type() -> str:
    """Return content type to use for ``/metrics`` responses."""
    return CONTENT_TYPE_LATEST


def metrics_unavailable_hint() -> str:
    """Return a stable installation hint for optional metrics support."""
    return _METRICS_IMPORT_HINT


def _status_class(status_code: int) -> str:
    if status_code < 100:
        return "0xx"
    if status_code > 599:
        return "6xx"
    return f"{status_code // 100}xx"
