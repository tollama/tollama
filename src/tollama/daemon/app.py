"""FastAPI application for the public tollama HTTP API."""

from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import httpx
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
)
from starlette.middleware.cors import CORSMiddleware

from tollama.a2a import A2AOperationHandlers, A2AServer
from tollama.core.auto_select import AutoSelection, select_auto_models
from tollama.core.config import ConfigFileError, TollamaConfig, load_config
from tollama.core.counterfactual import generate_counterfactual
from tollama.core.ensemble import EnsembleError, merge_forecast_responses
from tollama.core.env_override import set_env_temporarily
from tollama.core.explainability import generate_explanation
from tollama.core.forecast_metrics import compute_forecast_metrics
from tollama.core.hf_pull import (
    OfflineModelUnavailableError,
    PullError,
    pull_snapshot_to_local_dir,
)
from tollama.core.ingest import (
    IngestDependencyError,
    IngestError,
    load_series_inputs_from_bytes,
    load_series_inputs_from_data_url,
)
from tollama.core.modelfile import (
    ModelfileListResponse,
    ModelfileUpsertRequest,
    list_modelfiles,
    load_modelfile,
    merge_modelfile_defaults,
    remove_modelfile,
    write_modelfile,
)
from tollama.core.narratives import (
    build_analysis_narrative,
    build_comparison_narrative,
    build_forecast_narrative,
    build_pipeline_narrative,
)
from tollama.core.pipeline import run_pipeline_analysis
from tollama.core.progressive import ProgressiveStage, build_progressive_stages
from tollama.core.pull_defaults import resolve_effective_pull_defaults
from tollama.core.recommend import recommend_models
from tollama.core.redact import redact_config_dict, redact_env_dict, redact_proxy_url
from tollama.core.registry import ModelCapabilities, ModelSpec, get_model_spec, list_registry_models
from tollama.core.report import build_forecast_report, build_report_narrative, run_report_analysis
from tollama.core.scenario_tree import build_scenario_tree
from tollama.core.scenarios import apply_scenario
from tollama.core.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AutoForecastRequest,
    AutoForecastResponse,
    AutoSelectionInfo,
    AutoSelectionScore,
    CompareError,
    CompareRequest,
    CompareResponse,
    CompareResult,
    CompareSummary,
    CounterfactualRequest,
    CounterfactualResponse,
    ForecastReport,
    ForecastRequest,
    ForecastResponse,
    ForecastTiming,
    GenerateRequest,
    GenerateResponse,
    IngestOptions,
    PipelineRequest,
    PipelineResponse,
    ProgressiveForecastEvent,
    ReportRequest,
    ScenarioTreeRequest,
    ScenarioTreeResponse,
    WhatIfError,
    WhatIfRequest,
    WhatIfResponse,
    WhatIfResult,
    WhatIfSummary,
)
from tollama.core.series_analysis import analyze_series_request
from tollama.core.storage import (
    TollamaPaths,
    install_from_registry,
    list_installed,
    read_manifest,
    remove_model,
    write_manifest,
)
from tollama.core.synthetic import generate_synthetic_series
from tollama.dashboard import create_dashboard_html_router

from .auth import current_key_id, require_api_key
from .covariates import apply_covariate_capabilities, build_covariate_profile, normalize_covariates
from .dashboard_api import create_dashboard_router
from .loaded_models import LoadedModelTracker, parse_keep_alive, to_utc_iso
from .metering import UsageMeter, create_usage_meter, usage_unavailable_hint
from .metrics import (
    ForecastMetricsMiddleware,
    PrometheusMetrics,
    create_prometheus_metrics,
    metrics_content_type,
    metrics_unavailable_hint,
)
from .rate_limiter import TokenBucketRateLimiter, create_rate_limiter_from_env
from .runner_manager import RunnerManager
from .sse import EventStream, EventSubscription, format_sse_event
from .supervisor import (
    RunnerCallError,
    RunnerError,
    RunnerProtocolError,
    RunnerUnavailableError,
)

DEFAULT_FORECAST_TIMEOUT_SECONDS = 300.0
FORECAST_TIMEOUT_ENV_NAME = "TOLLAMA_FORECAST_TIMEOUT_SECONDS"
ALLOW_REMOTE_DATA_URL_ENV_NAME = "TOLLAMA_ALLOW_REMOTE_DATA_URL"
AUTO_ENSEMBLE_MAX_WORKERS = 4
AUTO_FORECAST_MEMBER_TIMEOUT_SECONDS = 10.0
DEFAULT_MODIFIED_AT = "1970-01-01T00:00:00Z"
INFO_ENV_KEYS = (
    "TOLLAMA_HOME",
    "TOLLAMA_HOST",
    FORECAST_TIMEOUT_ENV_NAME,
    ALLOW_REMOTE_DATA_URL_ENV_NAME,
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)
TOKEN_ENV_KEYS = (
    "TOLLAMA_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)
DOCS_PUBLIC_ENV_NAME = "TOLLAMA_DOCS_PUBLIC"
CORS_ORIGINS_ENV_NAME = "TOLLAMA_CORS_ORIGINS"
DASHBOARD_ENABLED_ENV_NAME = "TOLLAMA_DASHBOARD"
DASHBOARD_REQUIRE_AUTH_ENV_NAME = "TOLLAMA_DASHBOARD_REQUIRE_AUTH"

_OPENAPI_TAGS = [
    {
        "name": "system",
        "description": "Version, health, diagnostics, and service metadata.",
    },
    {
        "name": "models",
        "description": "Model registry, install, remove, and lifecycle operations.",
    },
    {
        "name": "forecast",
        "description": "Forecast request execution APIs, including streaming modes.",
    },
    {
        "name": "ingest",
        "description": "CSV/parquet upload endpoints for normalized series ingestion.",
    },
    {
        "name": "analysis",
        "description": "Analyze, compare, report, and scenario-style intelligence APIs.",
    },
    {
        "name": "runtime",
        "description": "Event streaming and usage/metrics observability endpoints.",
    },
    {"name": "modelfiles", "description": "TSModelfile profile CRUD endpoints."},
    {"name": "a2a", "description": "A2A discovery and JSON-RPC message handling endpoints."},
]


@dataclass(frozen=True)
class PullOptions:
    insecure: bool
    offline: bool
    offline_explicit: bool
    local_files_only: bool
    http_proxy: str | None
    http_proxy_explicit: bool
    https_proxy: str | None
    https_proxy_explicit: bool
    no_proxy: str | None
    no_proxy_explicit: bool
    hf_home: str | None
    hf_home_explicit: bool
    max_workers: int
    token: str | None


class ModelPullRequest(BaseModel):
    """Request body for pulling a model into local storage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: StrictStr = Field(min_length=1, description="Model name to pull from the registry.")
    accept_license: StrictBool = Field(
        description="Set true to accept model license terms during pull."
    )


class ModelShowRequest(BaseModel):
    """Request body for the Ollama-compatible show endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1, description="Installed model name to inspect.")


class ModelDeleteRequest(BaseModel):
    """Request body for the Ollama-compatible delete endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1, description="Installed model name to remove.")


class ApiPullRequest(BaseModel):
    """Request body for the Ollama-compatible pull endpoint."""

    model_config = ConfigDict(extra="allow", strict=True)

    model: StrictStr = Field(min_length=1, description="Model name to pull from upstream registry.")
    stream: StrictBool = Field(
        default=True,
        description="When true, stream NDJSON progress events instead of one final payload.",
    )
    accept_license: StrictBool = Field(
        default=False,
        description="Set true to accept model license terms during pull.",
    )
    insecure: StrictBool | None = Field(
        default=None,
        description="Override SSL verification behavior for pull requests.",
    )
    offline: StrictBool | None = Field(
        default=None,
        description="Force offline mode, disallowing remote downloads.",
    )
    local_files_only: StrictBool | None = Field(
        default=None,
        description="Read from local Hugging Face cache only when available.",
    )
    http_proxy: StrictStr | None = Field(default=None, description="HTTP proxy URL override.")
    https_proxy: StrictStr | None = Field(default=None, description="HTTPS proxy URL override.")
    no_proxy: StrictStr | None = Field(
        default=None,
        description="Comma-separated no-proxy patterns.",
    )
    hf_home: StrictStr | None = Field(
        default=None,
        description="Optional HF_HOME directory override used for cache lookup.",
    )
    max_workers: StrictInt | None = Field(
        default=None,
        gt=0,
        description="Maximum pull worker count for parallel transfer stages.",
    )
    token: StrictStr | None = Field(
        default=None,
        description="Hugging Face token override for private or gated model pulls.",
    )


class ConfigProvider:
    """Read-through cache for tollama config with mtime invalidation."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cached_config: TollamaConfig | None = None
        self._cached_path: Path | None = None
        self._cached_mtime_ns: int | None = None

    def get(self) -> TollamaConfig:
        with self._lock:
            paths = TollamaPaths.default()
            config_path = paths.config_path
            cached_path = self._cached_path
            cached_config = self._cached_config

            if not config_path.exists():
                if (
                    cached_path == config_path
                    and self._cached_mtime_ns is None
                    and cached_config is not None
                ):
                    return cached_config
                loaded = TollamaConfig()
                self._cached_config = loaded
                self._cached_path = config_path
                self._cached_mtime_ns = None
                return loaded

            mtime_ns = config_path.stat().st_mtime_ns
            if (
                cached_path == config_path
                and self._cached_mtime_ns == mtime_ns
                and cached_config is not None
            ):
                return cached_config

            loaded = load_config(paths)
            self._cached_config = loaded
            self._cached_path = config_path
            self._cached_mtime_ns = mtime_ns
            return loaded


class ForecastRequestWithKeepAlive(ForecastRequest):
    """Forecast request payload with optional Ollama-compatible keep_alive semantics."""

    keep_alive: StrictStr | StrictInt | StrictFloat | None = Field(
        default=None,
        description="Optional keep-alive policy for loaded model sessions.",
    )


class ApiForecastRequest(ForecastRequestWithKeepAlive):
    """Ollama-compatible forecast request payload."""

    stream: StrictBool = Field(
        default=True,
        description="When true, return NDJSON stream output for forecast events.",
    )


class ValidateResponse(BaseModel):
    """Forecast request validation response payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    valid: StrictBool = Field(description="Whether the request payload passed validation checks.")
    errors: list[StrictStr] = Field(
        default_factory=list,
        description="Validation errors that prevent running the request.",
    )
    warnings: list[StrictStr] = Field(
        default_factory=list,
        description="Non-fatal compatibility warnings emitted during request normalization.",
    )
    suggestions: list[StrictStr] = Field(
        default_factory=list,
        description="Actionable next-step suggestions based on request and model compatibility.",
    )


class VersionResponse(BaseModel):
    """Minimal daemon version response payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: StrictStr = Field(description="Installed tollama package version.")


class HealthResponse(BaseModel):
    """Daemon health probe response payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: StrictStr = Field(description="Service health status value.")


class ModelDeleteResponse(BaseModel):
    """Ollama-compatible delete model response payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    deleted: StrictBool = Field(description="True when requested model metadata was removed.")
    model: StrictStr = Field(description="Name of the removed model.")


class V1ModelDeleteResponse(BaseModel):
    """v1 API delete model response payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    removed: StrictBool = Field(description="True when requested model metadata was removed.")
    name: StrictStr = Field(description="Name of the removed model.")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.started_at = datetime.now(UTC)
    try:
        yield
    finally:
        app.state.runner_manager.stop()
        app.state.loaded_model_tracker.clear()
        dashboard_resource_stack = getattr(app.state, "dashboard_resource_stack", None)
        if isinstance(dashboard_resource_stack, ExitStack):
            dashboard_resource_stack.close()


def create_app(*, runner_manager: RunnerManager | None = None) -> FastAPI:
    """Create a configured FastAPI daemon app."""
    package_version = _resolve_package_version()
    docs_public = _docs_public_enabled()
    docs_dependencies = [] if docs_public else [Depends(require_api_key)]
    app = FastAPI(
        title="tollama daemon",
        version=package_version,
        lifespan=_lifespan,
        dependencies=[Depends(require_api_key)],
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        openapi_tags=_OPENAPI_TAGS,
    )
    app.state.runner_manager = runner_manager or _build_default_runner_manager()
    app.state.loaded_model_tracker = LoadedModelTracker()
    app.state.config_provider = ConfigProvider()
    app.state.started_at = datetime.now(UTC)
    app.state.host_binding = _resolve_host_binding_from_env()
    app.state.prometheus_metrics = _build_prometheus_metrics(app)
    app.state.usage_meter = _build_usage_meter()
    app.state.rate_limiter = _build_rate_limiter()
    app.state.event_stream = EventStream()
    app.state.dashboard_resource_stack = ExitStack()
    app.state.dashboard_index_path = None

    cors_origins = _resolve_cors_origins_from_env()
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    dashboard_enabled = _dashboard_enabled()
    dashboard_require_auth = _dashboard_require_auth_enabled()
    app.state.dashboard_enabled = dashboard_enabled
    app.state.dashboard_require_auth = dashboard_require_auth
    if dashboard_require_auth:

        @app.middleware("http")
        async def _dashboard_static_auth_middleware(
            request: Request,
            call_next: Callable[..., Any],
        ) -> Response:
            if request.url.path.startswith("/dashboard/static"):
                try:
                    require_api_key(request)
                except HTTPException as exc:
                    content = jsonable_encoder(
                        _http_error_body(status_code=exc.status_code, detail=exc.detail),
                        custom_encoder={ValueError: str},
                    )
                    return JSONResponse(
                        status_code=exc.status_code,
                        content=content,
                        headers=exc.headers,
                    )
            return await call_next(request)

    if dashboard_enabled:
        dashboard_static_dir = _resolve_dashboard_static_dir(app)
        app.mount(
            "/dashboard/static",
            StaticFiles(directory=str(dashboard_static_dir)),
            name="dashboard-static",
        )
        app.state.dashboard_index_path = dashboard_static_dir / "index.html"
        app.include_router(
            create_dashboard_html_router(partials_dir=dashboard_static_dir / "partials"),
        )

        @app.get("/dashboard", include_in_schema=False)
        @app.get("/dashboard/{path:path}", include_in_schema=False)
        def dashboard(_path: str = "") -> Response:
            del _path
            index_path = _dashboard_index_path(app)
            return FileResponse(index_path)

    if app.state.prometheus_metrics is not None:
        app.add_middleware(
            ForecastMetricsMiddleware,
            metrics=app.state.prometheus_metrics,
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        detail = jsonable_encoder(
            exc.errors(),
            custom_encoder={ValueError: str},
        )
        return JSONResponse(
            status_code=400,
            content=_http_error_body(status_code=400, detail=detail),
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        _request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        content = jsonable_encoder(
            _http_error_body(status_code=exc.status_code, detail=exc.detail),
            custom_encoder={ValueError: str},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers,
        )

    @app.get(
        "/openapi.json",
        include_in_schema=False,
        dependencies=docs_dependencies,
    )
    def openapi_schema() -> JSONResponse:
        return JSONResponse(app.openapi())

    @app.get(
        "/docs",
        include_in_schema=False,
        dependencies=docs_dependencies,
    )
    def swagger_ui() -> Response:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - Swagger UI",
        )

    @app.get(
        "/docs/oauth2-redirect",
        include_in_schema=False,
        dependencies=docs_dependencies,
    )
    def swagger_ui_redirect() -> Response:
        return get_swagger_ui_oauth2_redirect_html()

    @app.get(
        "/redoc",
        include_in_schema=False,
        dependencies=docs_dependencies,
    )
    def redoc_ui() -> Response:
        return get_redoc_html(
            openapi_url="/openapi.json",
            title=f"{app.title} - ReDoc",
        )

    @app.get(
        "/api/version",
        response_model=VersionResponse,
        tags=["system"],
        summary="Daemon version",
        description="Return the currently running tollama daemon package version.",
    )
    def version() -> VersionResponse:
        return VersionResponse(version=package_version)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        collector = _optional_prometheus_metrics(app)
        if collector is None:
            raise HTTPException(status_code=503, detail=metrics_unavailable_hint())

        _unload_expired_models(app)
        payload = collector.render_latest()
        return Response(content=payload, media_type=metrics_content_type())

    @app.get(
        "/api/usage",
        response_model=dict[str, object],
        tags=["runtime"],
        summary="Usage snapshot",
        description="Return aggregated per-key usage counters when usage metering is enabled.",
    )
    def usage(request: Request) -> dict[str, object]:
        usage_meter = _optional_usage_meter(app)
        if usage_meter is None:
            raise HTTPException(status_code=503, detail=usage_unavailable_hint())
        return usage_meter.snapshot(key_id=current_key_id(request))

    @app.get(
        "/api/events",
        tags=["runtime"],
        summary="Event stream",
        description=(
            "Subscribe to daemon event streams over SSE with optional filters "
            "and heartbeats."
        ),
        responses={
            200: {
                "description": "SSE stream of daemon events.",
                "content": {"text/event-stream": {}},
            }
        },
    )
    def events(request: Request) -> StreamingResponse:
        event_stream = _optional_event_stream(app)
        if event_stream is None:
            raise HTTPException(status_code=503, detail="event streaming is unavailable")

        event_types = _parse_event_type_filter(request)
        heartbeat_seconds = _parse_positive_float_query_param(
            request.query_params.get("heartbeat"),
            key="heartbeat",
            default=15.0,
        )
        max_events = _parse_positive_int_query_param(
            request.query_params.get("max_events"),
            key="max_events",
            default=None,
        )

        subscription = event_stream.subscribe(
            key_id=current_key_id(request),
            event_types=event_types,
        )
        return StreamingResponse(
            _events_stream_lines(
                event_stream=event_stream,
                subscription=subscription,
                heartbeat_seconds=heartbeat_seconds,
                max_events=max_events,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/api/modelfiles",
        response_model=ModelfileListResponse,
        tags=["modelfiles"],
        summary="List TSModelfile profiles",
        description="Return all stored TSModelfile profile names and metadata.",
    )
    def modelfiles() -> ModelfileListResponse:
        return ModelfileListResponse(modelfiles=list_modelfiles())

    @app.get(
        "/api/modelfiles/{name}",
        response_model=dict[str, Any],
        tags=["modelfiles"],
        summary="Get TSModelfile profile",
        description="Return one stored TSModelfile profile by name.",
    )
    def modelfile_show(name: str) -> dict[str, Any]:
        try:
            stored = load_modelfile(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return stored.model_dump(mode="json", exclude_none=True)

    @app.post(
        "/api/modelfiles",
        response_model=dict[str, Any],
        tags=["modelfiles"],
        summary="Create or update TSModelfile profile",
        description="Upsert one TSModelfile profile and return the stored payload.",
    )
    def modelfile_upsert(payload: ModelfileUpsertRequest) -> dict[str, Any]:
        try:
            profile = payload.resolved_profile()
            stored = write_modelfile(payload.name, profile)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return stored.model_dump(mode="json", exclude_none=True)

    @app.delete(
        "/api/modelfiles/{name}",
        response_model=dict[str, Any],
        tags=["modelfiles"],
        summary="Delete TSModelfile profile",
        description="Delete one TSModelfile profile by name.",
    )
    def modelfile_delete(name: str) -> dict[str, Any]:
        try:
            removed = remove_modelfile(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail=f"modelfile {name!r} not found")
        return {"deleted": True, "name": name}

    @app.post(
        "/api/ingest/upload",
        response_model=dict[str, Any],
        tags=["ingest"],
        summary="Upload and ingest series data",
        description="Upload CSV/parquet content and return normalized series payload objects.",
    )
    async def ingest_upload(
        file: UploadFile = File(..., description="CSV/parquet file payload to ingest."),
        format_hint: str | None = Form(
            default=None,
            description="Optional format override such as 'csv' or 'parquet'.",
        ),
        timestamp_column: str | None = Form(
            default=None,
            description="Optional timestamp column name override.",
        ),
        series_id_column: str | None = Form(
            default=None,
            description="Optional series identifier column name override.",
        ),
        target_column: str | None = Form(
            default=None,
            description="Optional target column name override.",
        ),
        freq_column: str | None = Form(
            default=None,
            description="Optional frequency column name override.",
        ),
    ) -> dict[str, Any]:
        file_payload = await file.read()
        filename = file.filename or "upload.csv"
        try:
            ingest_options = IngestOptions.model_validate(
                {
                    "format": format_hint,
                    "timestamp_column": timestamp_column,
                    "series_id_column": series_id_column,
                    "target_column": target_column,
                    "freq_column": freq_column,
                },
            )
            series = load_series_inputs_from_bytes(
                file_payload,
                filename=filename,
                format_hint=ingest_options.format,
                timestamp_column=ingest_options.timestamp_column,
                series_id_column=ingest_options.series_id_column,
                target_column=ingest_options.target_column,
                freq_column=ingest_options.freq_column,
            )
        except IngestDependencyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (IngestError, ValidationError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "filename": filename,
            "series": [item.model_dump(mode="json", exclude_none=True) for item in series],
        }

    @app.post(
        "/api/forecast/upload",
        response_model=ForecastResponse,
        tags=["forecast", "ingest"],
        summary="Forecast from uploaded file",
        description=(
            "Upload file data plus request JSON to run one forecast without "
            "manual preprocessing."
        ),
    )
    async def forecast_upload(
        request: Request,
        payload: str = Form(
            ...,
            description="Forecast request JSON string without `series` or `data_url`.",
        ),
        file: UploadFile = File(
            ...,
            description="CSV/parquet file payload to ingest and forecast.",
        ),
        format_hint: str | None = Form(
            default=None,
            description="Optional format override such as 'csv' or 'parquet'.",
        ),
        timestamp_column: str | None = Form(
            default=None,
            description="Optional timestamp column name override.",
        ),
        series_id_column: str | None = Form(
            default=None,
            description="Optional series identifier column name override.",
        ),
        target_column: str | None = Form(
            default=None,
            description="Optional target column name override.",
        ),
        freq_column: str | None = Form(
            default=None,
            description="Optional frequency column name override.",
        ),
    ) -> ForecastResponse:
        try:
            parsed_payload = _load_json_object(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if "series" in parsed_payload or "data_url" in parsed_payload:
            raise HTTPException(
                status_code=400,
                detail="payload for /api/forecast/upload must not include series or data_url",
            )

        file_payload = await file.read()
        filename = file.filename or "upload.csv"
        try:
            ingest_options = IngestOptions.model_validate(
                {
                    "format": format_hint,
                    "timestamp_column": timestamp_column,
                    "series_id_column": series_id_column,
                    "target_column": target_column,
                    "freq_column": freq_column,
                },
            )
            series = load_series_inputs_from_bytes(
                file_payload,
                filename=filename,
                format_hint=ingest_options.format,
                timestamp_column=ingest_options.timestamp_column,
                series_id_column=ingest_options.series_id_column,
                target_column=ingest_options.target_column,
                freq_column=ingest_options.freq_column,
            )
            parsed_payload["series"] = [item.model_dump(mode="python") for item in series]
            forecast_payload = ForecastRequestWithKeepAlive.model_validate(parsed_payload)
        except IngestDependencyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (IngestError, ValidationError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _execute_forecast(app, payload=forecast_payload, request=request)

    @app.get(
        "/api/info",
        response_model=dict[str, Any],
        tags=["system"],
        summary="Daemon diagnostics",
        description=(
            "Return one comprehensive diagnostics payload with daemon, config, "
            "model, and runtime state."
        ),
    )
    def info() -> dict[str, Any]:
        _unload_expired_models(app)
        started_at = _optional_datetime(getattr(app.state, "started_at", None))
        now = datetime.now(UTC)
        uptime_seconds = _uptime_seconds(started_at=started_at, now=now)
        paths = TollamaPaths.default()
        config_payload = _collect_redacted_config_payload(paths)
        config_defaults = _load_config_or_default(paths)

        return {
            "daemon": {
                "version": package_version,
                "started_at": to_utc_iso(started_at),
                "uptime_seconds": uptime_seconds,
                "host_binding": _optional_nonempty_str(getattr(app.state, "host_binding", None)),
            },
            "paths": {
                "tollama_home": str(paths.base_dir),
                "config_path": str(paths.config_path),
                "config_exists": paths.config_path.exists(),
            },
            "config": config_payload,
            "env": _collect_info_env(),
            "pull_defaults": _collect_pull_defaults(config=config_defaults),
            "models": {
                "installed": _collect_installed_model_entries(paths=paths),
                "loaded": _collect_loaded_model_entries(app),
                "available": _collect_available_model_entries(),
            },
            "runners": app.state.runner_manager.get_all_statuses(),
        }

    @app.get(
        "/api/tags",
        response_model=dict[str, list[dict[str, Any]]],
        tags=["models"],
        summary="List installed model tags",
        description="Return installed models in Ollama-compatible tag list format.",
    )
    def tags() -> dict[str, list[dict[str, Any]]]:
        manifests = list_installed()
        models = [_to_ollama_tag_model(manifest) for manifest in manifests]
        return {"models": models}

    @app.post(
        "/api/show",
        response_model=dict[str, Any],
        tags=["models"],
        summary="Show model metadata",
        description="Return detailed metadata for one installed model.",
    )
    def show(payload: ModelShowRequest) -> dict[str, Any]:
        manifest = _find_installed_manifest(payload.model)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"model {payload.model!r} is not installed")

        name = manifest.get("name")
        model_name = name if isinstance(name, str) else payload.model
        family = manifest.get("family")
        family_name = family if isinstance(family, str) else ""
        source = manifest.get("source")
        resolved = manifest.get("resolved")
        license_data = manifest.get("license")

        return {
            "name": model_name,
            "model": model_name,
            "family": family_name,
            "source": source if isinstance(source, dict) else {},
            "resolved": resolved if isinstance(resolved, dict) else {},
            "digest": _manifest_digest(manifest),
            "size": _manifest_size_bytes(manifest),
            "snapshot_path": _manifest_snapshot_path(manifest),
            "license": license_data if isinstance(license_data, dict) else {},
            "modelfile": "",
            "parameters": "",
        }

    @app.post(
        "/api/pull",
        response_model=None,
        tags=["models"],
        summary="Pull model snapshot",
        description=(
            "Pull one model into local storage. Returns one JSON payload when stream=false, "
            "or NDJSON progress events when stream=true."
        ),
        responses={
            200: {
                "description": "Pull result payload or NDJSON progress stream.",
                "content": {
                    "application/json": {},
                    "application/x-ndjson": {},
                },
            }
        },
    )
    def pull(payload: ApiPullRequest) -> Any:
        try:
            config: TollamaConfig = app.state.config_provider.get()
        except ConfigFileError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        pull_options = _to_pull_options(payload, config=config)
        if not payload.stream:
            try:
                return _pull_model_snapshot(
                    name=payload.model,
                    accept_license=payload.accept_license,
                    pull_options=pull_options,
                )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=502,
                    detail=_friendly_pull_error_message(exc),
                ) from exc

        _validate_pull_request(name=payload.model, accept_license=payload.accept_license)
        return StreamingResponse(
            _pull_stream_lines(
                name=payload.model,
                accept_license=payload.accept_license,
                pull_options=pull_options,
            ),
            media_type="application/x-ndjson",
        )

    @app.delete(
        "/api/delete",
        response_model=ModelDeleteResponse,
        tags=["models"],
        summary="Delete installed model",
        description="Delete one installed model from local storage.",
    )
    def delete(payload: ModelDeleteRequest) -> ModelDeleteResponse:
        _remove_model_or_404(payload.model)
        return ModelDeleteResponse(deleted=True, model=payload.model)

    @app.get(
        "/api/ps",
        response_model=dict[str, list[dict[str, Any]]],
        tags=["models"],
        summary="List loaded models",
        description=(
            "Return currently loaded in-memory model sessions with keep-alive "
            "expiry metadata."
        ),
    )
    def ps() -> dict[str, list[dict[str, Any]]]:
        _unload_expired_models(app)
        tracker: LoadedModelTracker = app.state.loaded_model_tracker
        models = []
        for loaded in tracker.list_models():
            models.append(
                {
                    "name": loaded.name,
                    "model": loaded.model,
                    "expires_at": to_utc_iso(loaded.expires_at),
                    "size": 0,
                    "size_vram": 0,
                    "context_length": 0,
                    "details": {
                        "family": loaded.family,
                    },
                },
            )
        return {"models": models}

    app.include_router(
        create_dashboard_router(
            info_provider=info,
            ps_provider=ps,
            usage_provider=usage,
        )
    )

    @app.get(
        "/v1/health",
        response_model=HealthResponse,
        tags=["system"],
        summary="Health probe",
        description="Return simple liveness status for external health checks.",
    )
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get(
        "/v1/models",
        response_model=dict[str, list[dict[str, Any]]],
        tags=["models"],
        summary="List available and installed models",
        description="Return registry model inventory and currently installed local models.",
    )
    def models() -> dict[str, list[dict[str, Any]]]:
        installed_manifests = list_installed()
        installed_by_name = {
            str(item.get("name")): item
            for item in installed_manifests
            if isinstance(item.get("name"), str)
        }

        available: list[dict[str, Any]] = []
        for spec in list_registry_models():
            manifest = installed_by_name.get(spec.name)
            license_info: dict[str, Any] = {
                "type": spec.license.type,
                "needs_acceptance": spec.license.needs_acceptance,
            }
            if manifest is not None:
                manifest_license = manifest.get("license")
                if isinstance(manifest_license, dict) and "accepted" in manifest_license:
                    license_info["accepted"] = bool(manifest_license["accepted"])
            else:
                license_info["accepted"] = not spec.license.needs_acceptance

            available.append(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "installed": spec.name in installed_by_name,
                    "license": license_info,
                },
            )

        installed = []
        for manifest in installed_manifests:
            manifest_license = manifest.get("license")
            installed.append(
                {
                    "name": manifest.get("name"),
                    "family": manifest.get("family"),
                    "installed": True,
                    "license": _public_license_view(manifest_license),
                },
            )
        return {"available": available, "installed": installed}

    @app.post(
        "/api/validate",
        response_model=ValidateResponse,
        tags=["forecast"],
        summary="Validate forecast request",
        description=(
            "Validate forecast payloads including ingest normalization and "
            "covariate compatibility checks."
        ),
    )
    def validate(payload: Any = Body(...)) -> ValidateResponse:
        if not isinstance(payload, dict):
            return ValidateResponse(
                valid=False,
                errors=[
                    f"field '<root>': payload must be a JSON object (got: {payload!r})",
                ],
                suggestions=[],
            )

        try:
            request = ForecastRequest.model_validate(payload)
        except ValidationError as exc:
            return ValidateResponse(
                valid=False,
                errors=_format_validation_errors(exc.errors()),
                suggestions=[],
            )

        try:
            resolved_request = _prepare_forecast_payload(
                ForecastRequestWithKeepAlive.model_validate(
                    request.model_dump(mode="python", exclude_none=True),
                )
            )
        except FileNotFoundError as exc:
            return ValidateResponse(valid=False, errors=[str(exc)], suggestions=[])
        except IngestDependencyError as exc:
            return ValidateResponse(valid=False, errors=[str(exc)], suggestions=[])
        except (IngestError, ValidationError, ValueError) as exc:
            return ValidateResponse(valid=False, errors=[str(exc)], suggestions=[])

        warnings: list[str] = []
        suggestions: list[str] = []
        model_manifest = _find_installed_manifest(resolved_request.model)
        if model_manifest is None:
            model_family = "unknown"
            model_capabilities = ModelCapabilities()
            warnings.append("model not installed; covariate capability check used defaults")
            suggestions.append(
                f"Install the model first: tollama pull {resolved_request.model}",
            )
        else:
            family = model_manifest.get("family")
            if not isinstance(family, str) or not family:
                return ValidateResponse(
                    valid=False,
                    errors=[f"model {resolved_request.model!r} has invalid family metadata"],
                    warnings=warnings,
                    suggestions=suggestions,
                )
            model_family = family
            model_capabilities = _resolve_model_capabilities(resolved_request.model, model_manifest)

        try:
            normalized_series, normalize_warnings = normalize_covariates(
                resolved_request.series,
                resolved_request.horizon,
            )
        except ValueError as exc:
            return ValidateResponse(
                valid=False,
                errors=[str(exc)],
                warnings=warnings,
                suggestions=_validation_error_suggestions(
                    error_message=str(exc),
                    request=resolved_request,
                    model_capabilities=model_capabilities,
                ),
            )
        warnings.extend(normalize_warnings)

        try:
            _, capability_warnings = apply_covariate_capabilities(
                model_name=resolved_request.model,
                model_family=model_family,
                inputs=normalized_series,
                capabilities=model_capabilities,
                covariates_mode=resolved_request.parameters.covariates_mode,
            )
        except ValueError as exc:
            return ValidateResponse(
                valid=False,
                errors=[str(exc)],
                warnings=warnings,
                suggestions=_validation_error_suggestions(
                    error_message=str(exc),
                    request=resolved_request,
                    model_capabilities=model_capabilities,
                ),
            )
        warnings.extend(capability_warnings)

        suggestions.extend(
            _validation_suggestions(
                request=resolved_request,
                model_manifest=model_manifest,
                model_capabilities=model_capabilities,
                normalized_series=normalized_series,
                warnings=warnings,
            )
        )
        suggestions = _dedupe_preserve_order(suggestions)
        merged_warnings = _merge_warnings(None, warnings) or []
        return ValidateResponse(valid=True, warnings=merged_warnings, suggestions=suggestions)

    @app.post(
        "/v1/models/pull",
        response_model=dict[str, Any],
        tags=["models"],
        summary="Pull model (v1)",
        description="Pull one model into local storage through the stable v1 route.",
    )
    def pull_model(payload: ModelPullRequest) -> dict[str, Any]:
        return _install_model(name=payload.name, accept_license=payload.accept_license)

    @app.delete(
        "/v1/models/{name}",
        response_model=V1ModelDeleteResponse,
        tags=["models"],
        summary="Delete model (v1)",
        description="Delete one installed model through the stable v1 route.",
    )
    def delete_model(name: str) -> V1ModelDeleteResponse:
        _remove_model_or_404(name)
        return V1ModelDeleteResponse(removed=True, name=name)

    @app.post(
        "/api/forecast",
        response_model=None,
        tags=["forecast"],
        summary="Forecast",
        description=(
            "Run one forecast request. Returns one JSON payload when stream=false, "
            "or NDJSON event lines when stream=true."
        ),
        responses={
            200: {
                "description": "Forecast result payload or NDJSON stream payload.",
                "content": {
                    "application/json": {},
                    "application/x-ndjson": {},
                },
            }
        },
    )
    def api_forecast(payload: ApiForecastRequest, request: Request) -> Any:
        response = _execute_forecast(
            app,
            payload=payload,
            request=request,
            extra_exclude={"stream"},
        )
        if not payload.stream:
            return response
        return StreamingResponse(
            _forecast_stream_lines(response),
            media_type="application/x-ndjson",
        )

    @app.post(
        "/api/forecast/progressive",
        response_model=None,
        tags=["forecast"],
        summary="Progressive forecast stream",
        description="Stream staged model-selection and forecast refinement events over SSE.",
        responses={
            200: {
                "description": "SSE stream of progressive forecast events.",
                "content": {"text/event-stream": {}},
            }
        },
    )
    def api_forecast_progressive(
        payload: AutoForecastRequest,
        request: Request,
    ) -> StreamingResponse:
        stages, plan_warnings = _resolve_progressive_stages(payload=payload)
        return StreamingResponse(
            _progressive_forecast_sse_lines(
                app=app,
                payload=payload,
                request=request,
                stages=stages,
                plan_warnings=plan_warnings,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/api/compare",
        response_model=CompareResponse,
        tags=["analysis"],
        summary="Compare models",
        description=(
            "Run one forecast request across multiple models and aggregate "
            "per-model outcomes."
        ),
    )
    def compare(payload: CompareRequest, request: Request) -> CompareResponse:
        results: list[CompareResult] = []
        for model in payload.models:
            forecast_payload = ForecastRequestWithKeepAlive(
                model=model,
                horizon=payload.horizon,
                quantiles=payload.quantiles,
                series=payload.series,
                options=payload.options,
                timeout=payload.timeout,
                keep_alive=payload.keep_alive,
                parameters=payload.parameters,
                response_options=payload.response_options,
            )
            try:
                response = _execute_forecast(app, payload=forecast_payload, request=request)
            except HTTPException as exc:
                results.append(
                    CompareResult(
                        model=model,
                        ok=False,
                        error=CompareError(
                            category=_compare_error_category(exc.status_code),
                            status_code=exc.status_code,
                            message=_compare_error_message(exc),
                        ),
                    ),
                )
                continue
            except Exception as exc:  # noqa: BLE001
                results.append(
                    CompareResult(
                        model=model,
                        ok=False,
                        error=CompareError(
                            category="INTERNAL_ERROR",
                            status_code=500,
                            message=str(exc),
                        ),
                    ),
                )
                continue

            results.append(CompareResult(model=model, ok=True, response=response))

        summary = CompareSummary(
            requested_models=len(payload.models),
            succeeded=sum(1 for item in results if item.ok),
            failed=sum(1 for item in results if not item.ok),
        )
        compare_response = CompareResponse(
            models=list(payload.models),
            horizon=payload.horizon,
            results=results,
            summary=summary,
        )
        if payload.response_options.narrative:
            compare_response = compare_response.model_copy(
                update={"narrative": build_comparison_narrative(response=compare_response)},
            )
        return compare_response

    @app.post(
        "/api/analyze",
        response_model=AnalyzeResponse,
        tags=["analysis"],
        summary="Analyze series",
        description=(
            "Compute descriptive analysis metrics, anomalies, and optional "
            "narrative summaries."
        ),
    )
    def analyze(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
        response = analyze_series_request(payload)
        if payload.response_options.narrative:
            response = response.model_copy(
                update={"narrative": build_analysis_narrative(response=response)},
            )
        key_id = _event_key_id(request)
        _publish_event(
            app=app,
            key_id=key_id,
            event="analysis.complete",
            data={
                "series_count": len(response.results),
                "response": response.model_dump(mode="json", exclude_none=True),
            },
        )
        for result in response.results:
            anomaly_indices = list(result.anomaly_indices)
            if not anomaly_indices:
                continue
            _publish_event(
                app=app,
                key_id=key_id,
                event="anomaly.detected",
                data={
                    "series_id": result.id,
                    "anomaly_count": len(anomaly_indices),
                    "indices": anomaly_indices,
                },
            )
        return response

    @app.post(
        "/api/generate",
        response_model=GenerateResponse,
        tags=["analysis"],
        summary="Generate synthetic series",
        description="Generate synthetic series variations for forecasting experiments and demos.",
    )
    def generate(payload: GenerateRequest) -> GenerateResponse:
        try:
            return generate_synthetic_series(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/counterfactual",
        response_model=CounterfactualResponse,
        tags=["analysis"],
        summary="Counterfactual forecast",
        description=(
            "Estimate counterfactual trajectories and effect deltas after an "
            "intervention index."
        ),
    )
    def counterfactual(payload: CounterfactualRequest, request: Request) -> CounterfactualResponse:
        def _forecast_executor(
            counterfactual_forecast_request: ForecastRequest,
        ) -> ForecastResponse:
            request_payload = ForecastRequestWithKeepAlive.model_validate(
                {
                    **counterfactual_forecast_request.model_dump(mode="python", exclude_none=True),
                    "keep_alive": payload.keep_alive,
                },
            )
            return _execute_forecast(app, payload=request_payload, request=request)

        try:
            return generate_counterfactual(
                payload=payload,
                forecast_executor=_forecast_executor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/scenario-tree",
        response_model=ScenarioTreeResponse,
        tags=["analysis"],
        summary="Scenario tree",
        description="Build branching probabilistic scenario trees from recursive forecast calls.",
    )
    def scenario_tree(payload: ScenarioTreeRequest, request: Request) -> ScenarioTreeResponse:
        def _forecast_executor(tree_forecast_request: ForecastRequest) -> ForecastResponse:
            request_payload = ForecastRequestWithKeepAlive.model_validate(
                {
                    **tree_forecast_request.model_dump(mode="python", exclude_none=True),
                    "keep_alive": payload.keep_alive,
                },
            )
            return _execute_forecast(app, payload=request_payload, request=request)

        try:
            return build_scenario_tree(
                payload=payload,
                forecast_executor=_forecast_executor,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/report",
        response_model=ForecastReport,
        tags=["analysis"],
        summary="Composite forecast report",
        description=(
            "Run analysis, recommendation, and auto-forecast to produce one "
            "composite report payload."
        ),
    )
    def report(payload: ReportRequest, request: Request) -> ForecastReport:
        return _execute_report(app, payload=payload, request=request)

    @app.post(
        "/api/auto-forecast",
        response_model=AutoForecastResponse,
        tags=["forecast"],
        summary="Auto forecast",
        description="Automatically select model(s) and run zero-config forecast orchestration.",
    )
    def auto_forecast(payload: AutoForecastRequest, request: Request) -> AutoForecastResponse:
        return _execute_auto_forecast(app, payload=payload, request=request)

    @app.post(
        "/api/what-if",
        response_model=WhatIfResponse,
        tags=["analysis"],
        summary="What-if scenarios",
        description=(
            "Run baseline plus transformed scenario forecasts and aggregate "
            "scenario outcomes."
        ),
    )
    def what_if(payload: WhatIfRequest, request: Request) -> WhatIfResponse:
        baseline_payload = _what_if_payload_to_forecast_payload(payload=payload)
        baseline_response = _execute_forecast(app, payload=baseline_payload, request=request)

        results: list[WhatIfResult] = []
        for scenario in payload.scenarios:
            try:
                scenario_series = apply_scenario(
                    series=baseline_payload.series,
                    scenario=scenario,
                )
                scenario_payload = ForecastRequestWithKeepAlive.model_validate(
                    baseline_payload.model_copy(update={"series": scenario_series}).model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                )
                scenario_response = _execute_forecast(
                    app,
                    payload=scenario_payload,
                    request=request,
                )
            except HTTPException as exc:
                if not payload.continue_on_error:
                    raise
                results.append(
                    WhatIfResult(
                        scenario=scenario.name,
                        ok=False,
                        error=WhatIfError(
                            category=_what_if_error_category(exc.status_code),
                            status_code=exc.status_code,
                            message=_compare_error_message(exc),
                        ),
                    ),
                )
                continue
            except ValidationError as exc:
                if not payload.continue_on_error:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                results.append(
                    WhatIfResult(
                        scenario=scenario.name,
                        ok=False,
                        error=WhatIfError(
                            category="INVALID_SCENARIO",
                            status_code=400,
                            message=str(exc),
                        ),
                    ),
                )
                continue
            except ValueError as exc:
                if not payload.continue_on_error:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                results.append(
                    WhatIfResult(
                        scenario=scenario.name,
                        ok=False,
                        error=WhatIfError(
                            category="INVALID_SCENARIO",
                            status_code=400,
                            message=str(exc),
                        ),
                    ),
                )
                continue

            results.append(
                WhatIfResult(
                    scenario=scenario.name,
                    ok=True,
                    response=scenario_response,
                ),
            )

        summary = WhatIfSummary(
            requested_scenarios=len(payload.scenarios),
            succeeded=sum(1 for item in results if item.ok),
            failed=sum(1 for item in results if not item.ok),
        )
        return WhatIfResponse(
            model=payload.model,
            horizon=payload.horizon,
            baseline=baseline_response,
            results=results,
            summary=summary,
        )

    @app.post(
        "/api/pipeline",
        response_model=PipelineResponse,
        tags=["analysis"],
        summary="Autonomous pipeline",
        description="Run one autonomous end-to-end analysis/recommendation/forecast pipeline.",
    )
    def pipeline(payload: PipelineRequest, request: Request) -> PipelineResponse:
        return _execute_pipeline(app, payload=payload, request=request)

    @app.post(
        "/v1/forecast",
        response_model=ForecastResponse,
        tags=["forecast"],
        summary="Forecast (v1)",
        description="Run one forecast request through the stable v1 JSON endpoint.",
    )
    def forecast(payload: ForecastRequestWithKeepAlive, request: Request) -> ForecastResponse:
        return _execute_forecast(app, payload=payload, request=request)

    def _a2a_forecast_handler(payload: dict[str, Any], auth_request: Any) -> dict[str, Any]:
        forecast_payload = ForecastRequestWithKeepAlive.model_validate(payload)
        response = _execute_forecast(app, payload=forecast_payload, request=auth_request)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_auto_forecast_handler(payload: dict[str, Any], auth_request: Any) -> dict[str, Any]:
        auto_payload = AutoForecastRequest.model_validate(payload)
        response = _execute_auto_forecast(app, payload=auto_payload, request=auth_request)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_analyze_handler(payload: dict[str, Any], _auth_request: Any) -> dict[str, Any]:
        analyze_payload = AnalyzeRequest.model_validate(payload)
        response = analyze_series_request(analyze_payload)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_generate_handler(payload: dict[str, Any], _auth_request: Any) -> dict[str, Any]:
        generate_payload = GenerateRequest.model_validate(payload)
        response = generate_synthetic_series(generate_payload)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_compare_handler(payload: dict[str, Any], auth_request: Any) -> dict[str, Any]:
        compare_payload = CompareRequest.model_validate(payload)
        response = compare(compare_payload, auth_request)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_what_if_handler(payload: dict[str, Any], auth_request: Any) -> dict[str, Any]:
        what_if_payload = WhatIfRequest.model_validate(payload)
        response = what_if(what_if_payload, auth_request)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_pipeline_handler(payload: dict[str, Any], auth_request: Any) -> dict[str, Any]:
        pipeline_payload = PipelineRequest.model_validate(payload)
        response = _execute_pipeline(app, payload=pipeline_payload, request=auth_request)
        return response.model_dump(mode="json", exclude_none=True)

    def _a2a_recommend_handler(payload: dict[str, Any], _auth_request: Any) -> dict[str, Any]:
        try:
            return recommend_models(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid recommend request: {exc}") from exc

    a2a_server = A2AServer(
        app=app,
        package_version=package_version,
        handlers=A2AOperationHandlers(
            forecast=_a2a_forecast_handler,
            auto_forecast=_a2a_auto_forecast_handler,
            analyze=_a2a_analyze_handler,
            generate=_a2a_generate_handler,
            compare=_a2a_compare_handler,
            what_if=_a2a_what_if_handler,
            pipeline=_a2a_pipeline_handler,
            recommend=_a2a_recommend_handler,
        ),
    )
    app.state.a2a_server = a2a_server

    @app.get(
        "/.well-known/agent-card.json",
        response_model=dict[str, object],
        tags=["a2a"],
        summary="A2A agent card",
        description="Return A2A discovery metadata for the tollama agent endpoint.",
    )
    def a2a_agent_card(request: Request) -> dict[str, object]:
        return a2a_server.agent_card(request=request)

    @app.get("/.well-known/agent.json", include_in_schema=False)
    def a2a_agent_card_legacy(request: Request) -> dict[str, object]:
        return a2a_server.agent_card(request=request)

    @app.post(
        "/a2a",
        response_model=None,
        tags=["a2a"],
        summary="A2A JSON-RPC",
        description="Handle A2A JSON-RPC requests for forecasting and intelligence tools.",
    )
    def a2a_jsonrpc(request: Request, payload: Any = Body(...)) -> Response:
        return a2a_server.handle_jsonrpc(payload=payload, request=request)

    return app


def _execute_forecast(
    app: FastAPI,
    *,
    payload: ForecastRequestWithKeepAlive,
    request: Request | None = None,
    extra_exclude: set[str] | None = None,
) -> ForecastResponse:
    try:
        payload = _prepare_forecast_payload(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IngestDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (IngestError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _enforce_rate_limit(app=app, request=request)
    request_started_at = time.perf_counter()
    _unload_expired_models(app)
    forecast_timeout_seconds = _resolve_forecast_timeout_seconds(payload.timeout)
    request_now = datetime.now(UTC)
    model_manifest = _require_installed_manifest(payload.model)
    model_family = _manifest_family_or_500(model_manifest, payload.model)
    model_local_dir = _manifest_snapshot_path(model_manifest)
    model_source = _manifest_source(model_manifest)
    model_metadata = _manifest_metadata(model_manifest)
    model_capabilities = _resolve_model_capabilities(payload.model, model_manifest)
    key_id = _event_key_id(request)
    _publish_event(
        app=app,
        key_id=key_id,
        event="model.loaded",
        data={
            "model": payload.model,
            "family": model_family,
        },
    )

    try:
        normalized_series, normalize_warnings = normalize_covariates(
            payload.series,
            payload.horizon,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        compatible_series, capability_warnings = apply_covariate_capabilities(
            model_name=payload.model,
            model_family=model_family,
            inputs=normalized_series,
            capabilities=model_capabilities,
            covariates_mode=payload.parameters.covariates_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    forecast_payload = payload.model_copy(update={"series": compatible_series})
    request_warnings = [*normalize_warnings, *capability_warnings]

    try:
        keep_alive_policy = parse_keep_alive(forecast_payload.keep_alive, now=request_now)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runner_payload = forecast_payload.model_copy(
        update={
            "series": [
                series.model_copy(update={"actuals": None})
                for series in forecast_payload.series
            ],
            "parameters": forecast_payload.parameters.model_copy(update={"metrics": None}),
        },
    )

    exclude_fields = {"keep_alive"}
    exclude_fields.add("response_options")
    if extra_exclude is not None:
        exclude_fields.update(extra_exclude)
    params = runner_payload.model_dump(mode="json", exclude_none=True, exclude=exclude_fields)
    params["model_name"] = forecast_payload.model
    params["model_family"] = model_family
    if model_local_dir is not None:
        params["model_local_dir"] = model_local_dir
    if model_source is not None:
        params["model_source"] = model_source
    if model_metadata is not None:
        params["model_metadata"] = model_metadata

    _publish_event(
        app=app,
        key_id=key_id,
        event="forecast.progress",
        data={
            "model": forecast_payload.model,
            "family": model_family,
            "status": "running",
            "series_count": len(forecast_payload.series),
            "horizon": forecast_payload.horizon,
        },
    )

    runner_started_at = time.perf_counter()
    try:
        raw_result = app.state.runner_manager.call(
            family=model_family,
            method="forecast",
            params=params,
            timeout=forecast_timeout_seconds,
        )
    except RunnerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RunnerCallError as exc:
        status_code = 503 if exc.code == "DEPENDENCY_MISSING" else 502
        if exc.code == "BAD_REQUEST":
            status_code = 400
        raise HTTPException(status_code=status_code, detail=_runner_error_detail(exc)) from exc
    except RunnerProtocolError as exc:
        raise HTTPException(status_code=502, detail=_runner_error_detail(exc)) from exc
    runner_roundtrip_ms = _elapsed_ms(runner_started_at)

    try:
        response = ForecastResponse.model_validate(raw_result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"runner returned invalid forecast response: {exc}",
        ) from exc

    metrics_payload, metrics_warnings = compute_forecast_metrics(
        request=forecast_payload,
        response=response,
    )
    if metrics_payload is not None:
        response = response.model_copy(update={"metrics": metrics_payload})

    explanation = generate_explanation(request=forecast_payload, response=response)
    if explanation is not None:
        response = response.model_copy(update={"explanation": explanation})

    if forecast_payload.response_options.narrative:
        narrative = build_forecast_narrative(request=forecast_payload, response=response)
        if narrative is not None:
            response = response.model_copy(update={"narrative": narrative})

    merged_warnings = _merge_warnings(response.warnings, [*request_warnings, *metrics_warnings])
    if merged_warnings:
        response = response.model_copy(update={"warnings": merged_warnings})

    response = _enrich_forecast_observability(
        response=response,
        model_family=model_family,
        runner_roundtrip_ms=runner_roundtrip_ms,
        total_ms=_elapsed_ms(request_started_at),
    )
    _record_usage(
        app=app,
        request=request,
        response=response,
        series_processed=len(forecast_payload.series),
    )
    _publish_event(
        app=app,
        key_id=key_id,
        event="forecast.complete",
        data={
            "model": response.model,
            "family": model_family,
            "series_count": len(response.forecasts),
            "response": response.model_dump(mode="json", exclude_none=True),
        },
    )

    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    if keep_alive_policy.unload_immediately:
        try:
            app.state.runner_manager.unload(
                family=model_family,
                model=forecast_payload.model,
                timeout=forecast_timeout_seconds,
            )
        except RunnerError:
            app.state.runner_manager.stop(family=model_family)
        tracker.unload_runner(model_family)
        return response

    tracker.upsert(
        name=forecast_payload.model,
        model=forecast_payload.model,
        family=model_family,
        runner=model_family,
        expires_at=keep_alive_policy.expires_at,
        device=_extract_usage_device(response.usage),
    )
    return response


def _prepare_forecast_payload(
    payload: ForecastRequestWithKeepAlive,
) -> ForecastRequestWithKeepAlive:
    resolved = _apply_modelfile_defaults(payload)
    if resolved.data_url is None:
        return resolved

    ingest_options = resolved.ingest or IngestOptions()
    allow_remote = _optional_bool_str(_env_or_none(ALLOW_REMOTE_DATA_URL_ENV_NAME)) is True
    series = load_series_inputs_from_data_url(
        resolved.data_url,
        format_hint=ingest_options.format,
        timestamp_column=ingest_options.timestamp_column,
        series_id_column=ingest_options.series_id_column,
        target_column=ingest_options.target_column,
        freq_column=ingest_options.freq_column,
        allow_remote=allow_remote,
    )

    hydrated = resolved.model_copy(
        update={
            "series": series,
            "data_url": None,
            "ingest": None,
        },
    )
    return ForecastRequestWithKeepAlive.model_validate(
        hydrated.model_dump(mode="python", exclude_none=True),
    )


def _apply_modelfile_defaults(
    payload: ForecastRequestWithKeepAlive,
) -> ForecastRequestWithKeepAlive:
    modelfile_name = _optional_nonempty_str(payload.modelfile)
    if modelfile_name is None:
        return payload

    try:
        stored = load_modelfile(modelfile_name)
    except FileNotFoundError:
        raise

    request_payload = payload.model_dump(mode="python")
    merged_payload = merge_modelfile_defaults(
        request_payload=request_payload,
        profile=stored.profile,
        request_fields_set=set(payload.model_fields_set),
    )
    merged_payload["modelfile"] = modelfile_name
    return ForecastRequestWithKeepAlive.model_validate(merged_payload)


def _elapsed_ms(started_at: float) -> float:
    return max((time.perf_counter() - started_at) * 1000.0, 0.0)


def _enrich_forecast_observability(
    *,
    response: ForecastResponse,
    model_family: str,
    runner_roundtrip_ms: float,
    total_ms: float,
) -> ForecastResponse:
    usage: dict[str, Any] = dict(response.usage or {})
    usage.setdefault("runner", f"tollama-{model_family}")
    usage.setdefault("device", "unknown")
    usage.setdefault("peak_memory_mb", _process_peak_memory_mb() or 0.0)

    existing_timing = response.timing or ForecastTiming()
    model_load_ms = _resolve_non_negative_ms(existing_timing.model_load_ms, default=0.0)
    inference_ms = _resolve_non_negative_ms(
        existing_timing.inference_ms,
        default=runner_roundtrip_ms,
    )

    return response.model_copy(
        update={
            "usage": usage,
            "timing": ForecastTiming(
                model_load_ms=model_load_ms,
                inference_ms=inference_ms,
                total_ms=max(total_ms, 0.0),
            ),
        },
    )


def _resolve_non_negative_ms(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        resolved = float(value)
        if resolved >= 0.0:
            return resolved
    return max(default, 0.0)


def _process_peak_memory_mb() -> float | None:
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, ValueError, OSError):
        return None

    if peak <= 0.0:
        return None

    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(peak / divisor, 3)


def _extract_usage_device(usage: dict[str, Any] | None) -> str | None:
    if usage is None:
        return None
    device = usage.get("device")
    if isinstance(device, str) and device:
        return device
    return None


def _enforce_rate_limit(*, app: FastAPI, request: Request | None) -> None:
    limiter = _optional_rate_limiter(app)
    if limiter is None:
        return

    key = current_key_id(request) if request is not None else "anonymous"
    if limiter.allow(key_id=key):
        return
    raise HTTPException(status_code=429, detail="rate limit exceeded")


def _record_usage(
    *,
    app: FastAPI,
    request: Request | None,
    response: ForecastResponse,
    series_processed: int,
) -> None:
    meter = _optional_usage_meter(app)
    if meter is None:
        return

    timing = response.timing
    inference_ms = timing.inference_ms if timing is not None else 0.0
    inference_value = float(inference_ms) if inference_ms is not None else 0.0
    key = current_key_id(request) if request is not None else "anonymous"
    try:
        meter.record_usage(
            key_id=key,
            inference_ms=inference_value,
            series_processed=series_processed,
        )
    except Exception:  # noqa: BLE001
        return


def _execute_auto_forecast(
    app: FastAPI,
    *,
    payload: AutoForecastRequest,
    request: Request | None = None,
) -> AutoForecastResponse:
    _unload_expired_models(app)
    manifests = list_installed()
    installed_by_name = {
        name: manifest
        for manifest in manifests
        if isinstance((name := manifest.get("name")), str)
    }
    installed_models = sorted(installed_by_name)
    if not installed_models:
        raise HTTPException(
            status_code=404,
            detail="no installed models available for auto-forecast",
        )

    fallback_used = False
    fallback_rationale: list[str] = []
    blocked_models: set[str] = set()

    if payload.model is not None:
        explicit_model = payload.model
        explicit_manifest = installed_by_name.get(explicit_model)
        if explicit_manifest is None:
            if not payload.allow_fallback:
                raise HTTPException(
                    status_code=404,
                    detail=f"model {explicit_model!r} is not installed",
                )
            fallback_used = True
            blocked_models.add(explicit_model)
            fallback_rationale.append(
                f"explicit model {explicit_model!r} is not installed; "
                "using fallback auto-selection",
            )
        else:
            explicit_request = _auto_payload_to_forecast_payload(
                payload=payload,
                model=explicit_model,
                default_timeout=AUTO_FORECAST_MEMBER_TIMEOUT_SECONDS,
            )
            try:
                explicit_response = _execute_forecast(
                    app,
                    payload=explicit_request,
                    request=request,
                )
            except HTTPException as exc:
                if not payload.allow_fallback:
                    raise
                fallback_used = True
                blocked_models.add(explicit_model)
                fallback_rationale.append(
                    "explicit model "
                    f"{explicit_model!r} failed ({exc.status_code}): "
                    f"{_compare_error_message(exc)}; "
                    "using fallback auto-selection",
                )
            else:
                return AutoForecastResponse(
                    strategy=payload.strategy,
                    selection=_manual_auto_selection_info(
                        payload=payload,
                        model=explicit_model,
                        manifest=explicit_manifest,
                    ),
                    response=explicit_response,
                )

    candidate_models = [name for name in installed_models if name not in blocked_models]
    if not candidate_models:
        raise HTTPException(
            status_code=404,
            detail="no compatible installed models found for auto-forecast",
        )

    try:
        selection = select_auto_models(
            series=payload.series,
            horizon=payload.horizon,
            strategy=payload.strategy,
            include_models=candidate_models,
            ensemble_top_k=payload.ensemble_top_k,
            allow_restricted_license=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if payload.strategy == "ensemble":
        forecast_response, chosen_model, execution_rationale = _execute_auto_ensemble(
            app,
            payload=payload,
            selection=selection,
            request=request,
        )
        selection_info = _selection_info_from_auto_selection(
            payload=payload,
            selection=selection,
            chosen_model=chosen_model,
            fallback_used=fallback_used or bool(execution_rationale),
            rationale_prefix=[*fallback_rationale, *execution_rationale],
        )
        return AutoForecastResponse(
            strategy=payload.strategy,
            selection=selection_info,
            response=forecast_response,
        )

    forecast_response, chosen_model, execution_rationale = _execute_auto_candidates(
        app,
        payload=payload,
        candidate_models=[item.model for item in selection.ranked_candidates],
        request=request,
    )
    selection_info = _selection_info_from_auto_selection(
        payload=payload,
        selection=selection,
        chosen_model=chosen_model,
        fallback_used=fallback_used or bool(execution_rationale),
        rationale_prefix=[*fallback_rationale, *execution_rationale],
    )
    return AutoForecastResponse(
        strategy=payload.strategy,
        selection=selection_info,
        response=forecast_response,
    )


def _execute_auto_candidates(
    app: FastAPI,
    *,
    payload: AutoForecastRequest,
    candidate_models: list[str],
    request: Request | None = None,
) -> tuple[ForecastResponse, str, list[str]]:
    if not candidate_models:
        raise HTTPException(status_code=404, detail="no candidate models were provided")

    last_error: HTTPException | None = None
    execution_warnings: list[str] = []
    for index, model in enumerate(candidate_models):
        forecast_payload = _auto_payload_to_forecast_payload(
            payload=payload,
            model=model,
            default_timeout=AUTO_FORECAST_MEMBER_TIMEOUT_SECONDS,
        )
        try:
            response = _execute_forecast(app, payload=forecast_payload, request=request)
        except HTTPException as exc:
            last_error = exc
            detail = _compare_error_message(exc)
            execution_warnings.append(
                f"candidate model {model!r} failed ({exc.status_code}): {detail}",
            )
            continue

        if index > 0 and execution_warnings:
            merged_warnings = _merge_warnings(response.warnings, execution_warnings)
            if merged_warnings is not None:
                response = response.model_copy(update={"warnings": merged_warnings})
        return response, model, execution_warnings

    if last_error is not None:
        raise last_error
    raise HTTPException(
        status_code=404,
        detail="no compatible installed models found for auto-forecast",
    )


def _execute_auto_ensemble(
    app: FastAPI,
    *,
    payload: AutoForecastRequest,
    selection: AutoSelection,
    request: Request | None = None,
) -> tuple[ForecastResponse, str, list[str]]:
    execution_warnings: list[str] = []
    successful_by_model: dict[str, ForecastResponse] = {}
    score_by_model = {item.model: item.score for item in selection.ranked_candidates}

    max_workers = min(len(selection.selected_models), AUTO_ENSEMBLE_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _execute_forecast,
                app,
                payload=_auto_payload_to_forecast_payload(
                    payload=payload,
                    model=model,
                    default_timeout=AUTO_FORECAST_MEMBER_TIMEOUT_SECONDS,
                ),
                request=request,
            ): model
            for model in selection.selected_models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                response = future.result()
            except HTTPException as exc:
                detail = _compare_error_message(exc)
                execution_warnings.append(
                    f"ensemble member {model!r} failed ({exc.status_code}): {detail}",
                )
                continue
            except Exception as exc:  # noqa: BLE001
                execution_warnings.append(
                    f"ensemble member {model!r} failed (500): {exc}",
                )
                continue
            successful_by_model[model] = response

    if not successful_by_model:
        raise HTTPException(
            status_code=503,
            detail="all ensemble candidates failed to produce a forecast",
        )

    successful = [
        successful_by_model[model]
        for model in selection.selected_models
        if model in successful_by_model
    ]

    if len(successful) == 1:
        merged_warnings = _merge_warnings(
            successful[0].warnings,
            [
                "ensemble strategy requested but only one model succeeded; "
                "returned single-model forecast",
                *execution_warnings,
            ],
        )
        single = successful[0].model_copy(update={"warnings": merged_warnings})
        return single, successful[0].model, execution_warnings

    try:
        merged = merge_forecast_responses(
            successful,
            weights={
                response.model: max(score_by_model.get(response.model, 1.0), 1.0)
                for response in successful
            },
            method=payload.ensemble_method,
            model_name="ensemble",
        )
    except EnsembleError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    merged_warnings = _merge_warnings(
        merged.warnings,
        [
            "ensemble strategy returns aggregated point forecasts only; quantiles are omitted",
            *execution_warnings,
        ],
    )
    chosen_model = next(
        model for model in selection.selected_models if model in successful_by_model
    )
    return (
        merged.model_copy(update={"warnings": merged_warnings}),
        chosen_model,
        execution_warnings,
    )


def _auto_payload_to_forecast_payload(
    *,
    payload: AutoForecastRequest,
    model: str,
    default_timeout: float | None = None,
) -> ForecastRequestWithKeepAlive:
    timeout = payload.timeout
    if timeout is None:
        timeout = default_timeout

    return ForecastRequestWithKeepAlive(
        model=model,
        horizon=payload.horizon,
        quantiles=payload.quantiles,
        series=payload.series,
        options=payload.options,
        timeout=timeout,
        keep_alive=payload.keep_alive,
        parameters=payload.parameters,
        response_options=payload.response_options,
    )


def _what_if_payload_to_forecast_payload(
    *,
    payload: WhatIfRequest,
) -> ForecastRequestWithKeepAlive:
    return ForecastRequestWithKeepAlive(
        model=payload.model,
        horizon=payload.horizon,
        quantiles=payload.quantiles,
        series=payload.series,
        options=payload.options,
        timeout=payload.timeout,
        keep_alive=payload.keep_alive,
        parameters=payload.parameters,
        response_options=payload.response_options,
    )


def _execute_pipeline(
    app: FastAPI,
    *,
    payload: PipelineRequest,
    request: Request | None = None,
) -> PipelineResponse:
    insights = run_pipeline_analysis(payload)
    analysis_response = insights.analysis
    if payload.response_options.narrative:
        analysis_response = analysis_response.model_copy(
            update={"narrative": build_analysis_narrative(response=analysis_response)},
        )
    warnings: list[str] = []
    pulled_model: str | None = None

    if payload.pull_if_missing and insights.preferred_model is not None:
        if _find_installed_manifest(insights.preferred_model) is None:
            try:
                _install_model(
                    name=insights.preferred_model,
                    accept_license=payload.accept_license,
                )
            except HTTPException as exc:
                warnings.append(
                    "pipeline pull step failed for model "
                    f"{insights.preferred_model!r} ({exc.status_code}): "
                    f"{_compare_error_message(exc)}",
                )
            else:
                pulled_model = insights.preferred_model

    auto_payload = AutoForecastRequest(
        model=payload.model,
        allow_fallback=payload.allow_fallback,
        strategy=payload.strategy,
        ensemble_top_k=payload.ensemble_top_k,
        ensemble_method=payload.ensemble_method,
        horizon=payload.horizon,
        quantiles=payload.quantiles,
        series=payload.series,
        options=payload.options,
        timeout=payload.timeout,
        keep_alive=payload.keep_alive,
        parameters=payload.parameters,
        response_options=payload.response_options,
    )
    auto_forecast = _execute_auto_forecast(app, payload=auto_payload, request=request)

    pipeline_response = PipelineResponse(
        analysis=analysis_response,
        recommendation=insights.recommendation,
        pulled_model=pulled_model,
        auto_forecast=auto_forecast,
        warnings=warnings or None,
    )
    if payload.response_options.narrative:
        pipeline_response = pipeline_response.model_copy(
            update={"narrative": build_pipeline_narrative(response=pipeline_response)},
        )
    return pipeline_response


def _execute_report(
    app: FastAPI,
    *,
    payload: ReportRequest,
    request: Request | None = None,
) -> ForecastReport:
    insights = run_report_analysis(payload)
    analysis_response = insights.analysis
    if payload.response_options.narrative:
        analysis_response = analysis_response.model_copy(
            update={"narrative": build_analysis_narrative(response=analysis_response)},
        )

    warnings: list[str] = []
    if payload.pull_if_missing and insights.preferred_model is not None:
        if _find_installed_manifest(insights.preferred_model) is None:
            try:
                _install_model(
                    name=insights.preferred_model,
                    accept_license=payload.accept_license,
                )
            except HTTPException as exc:
                warnings.append(
                    "report pull step failed for model "
                    f"{insights.preferred_model!r} ({exc.status_code}): "
                    f"{_compare_error_message(exc)}",
                )

    auto_payload = AutoForecastRequest(
        model=payload.model,
        allow_fallback=payload.allow_fallback,
        strategy=payload.strategy,
        ensemble_top_k=payload.ensemble_top_k,
        ensemble_method=payload.ensemble_method,
        horizon=payload.horizon,
        quantiles=payload.quantiles,
        series=payload.series,
        options=payload.options,
        timeout=payload.timeout,
        keep_alive=payload.keep_alive,
        parameters=payload.parameters,
        response_options=payload.response_options,
    )
    auto_forecast = _execute_auto_forecast(app, payload=auto_payload, request=request)

    report = build_forecast_report(
        analysis=analysis_response,
        recommendation=insights.recommendation,
        forecast=auto_forecast,
        include_baseline=payload.include_baseline,
        warnings=warnings,
    )
    if payload.response_options.narrative:
        report = report.model_copy(
            update={"narrative": build_report_narrative(report=report)},
        )
    return report


def _manual_auto_selection_info(
    *,
    payload: AutoForecastRequest,
    model: str,
    manifest: dict[str, Any],
) -> AutoSelectionInfo:
    family = _manifest_family_or_500(manifest, model)
    return AutoSelectionInfo(
        strategy=payload.strategy,
        chosen_model=model,
        selected_models=[model],
        candidates=[
            AutoSelectionScore(
                model=model,
                family=family,
                rank=1,
                score=0.0,
                reasons=["explicit model override applied"],
            )
        ],
        rationale=["explicit model override applied"],
        fallback_used=False,
    )


def _selection_info_from_auto_selection(
    *,
    payload: AutoForecastRequest,
    selection: AutoSelection,
    chosen_model: str,
    fallback_used: bool,
    rationale_prefix: list[str],
) -> AutoSelectionInfo:
    candidates: list[AutoSelectionScore] = []
    chosen_reasons: list[str] = []
    for rank, candidate in enumerate(selection.ranked_candidates, start=1):
        candidates.append(
            AutoSelectionScore(
                model=candidate.model,
                family=candidate.family,
                rank=rank,
                score=float(candidate.score),
                reasons=list(candidate.reasons),
            ),
        )
        if candidate.model == chosen_model:
            chosen_reasons = list(candidate.reasons)

    rationale = _dedupe_messages(
        [
            *rationale_prefix,
            "strategy "
            f"{payload.strategy!r} evaluated {len(selection.ranked_candidates)} "
            "compatible installed models",
            *chosen_reasons,
        ],
    )
    if payload.strategy == "ensemble":
        selected_models = list(selection.selected_models)
    else:
        selected_models = [chosen_model]

    return AutoSelectionInfo(
        strategy=payload.strategy,
        chosen_model=chosen_model,
        selected_models=selected_models,
        candidates=candidates,
        rationale=rationale or ["auto-selection completed"],
        fallback_used=fallback_used,
    )


def _compare_error_category(status_code: int) -> str:
    if status_code == 400:
        return "INVALID_REQUEST"
    if status_code == 404:
        return "MODEL_MISSING"
    if status_code == 503:
        return "RUNNER_UNAVAILABLE"
    if status_code == 502:
        return "RUNNER_ERROR"
    return "DAEMON_ERROR"


def _what_if_error_category(status_code: int) -> str:
    if status_code == 400:
        return "INVALID_REQUEST"
    if status_code == 404:
        return "MODEL_MISSING"
    if status_code == 503:
        return "RUNNER_UNAVAILABLE"
    if status_code == 502:
        return "RUNNER_ERROR"
    return "DAEMON_ERROR"


def _compare_error_message(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, str):
        return detail
    return str(detail)


def _forecast_stream_lines(response: ForecastResponse) -> Iterator[str]:
    yield _to_ndjson_line({"status": "loading model"})
    yield _to_ndjson_line({"status": "running forecast"})
    yield _to_ndjson_line(
        {
            "done": True,
            "response": response.model_dump(mode="json", exclude_none=True),
        },
    )


def _events_stream_lines(
    *,
    event_stream: EventStream,
    subscription: EventSubscription,
    heartbeat_seconds: float,
    max_events: int | None,
) -> Iterator[str]:
    try:
        yield from event_stream.iter_sse_lines(
            subscription=subscription,
            heartbeat_seconds=heartbeat_seconds,
            max_events=max_events,
        )
    finally:
        event_stream.unsubscribe(subscription)


def _progressive_forecast_sse_lines(
    *,
    app: FastAPI,
    payload: AutoForecastRequest,
    request: Request | None,
    stages: tuple[ProgressiveStage, ...],
    plan_warnings: list[str],
) -> Iterator[str]:
    key_id = _event_key_id(request)
    stage_count = len(stages)

    yield "retry: 3000\n\n"
    for warning in plan_warnings:
        warning_payload = {"message": warning}
        yield format_sse_event(event="forecast.warning", data=warning_payload)
        _publish_event(
            app=app,
            key_id=key_id,
            event="forecast.progress",
            data={"status": "warning", **warning_payload},
        )

    for stage in stages:
        is_final_stage = stage.index == stage_count
        selected_event = ProgressiveForecastEvent(
            event="model.selected",
            stage=stage.index,
            strategy=stage.strategy,
            model=stage.model,
            family=stage.family,
            status="selected",
            final=is_final_stage,
        )
        selected_payload = selected_event.model_dump(mode="json", exclude_none=True)
        yield format_sse_event(event=selected_event.event, data=selected_payload)
        _publish_event(
            app=app,
            key_id=key_id,
            event=selected_event.event,
            data=selected_payload,
        )

        running_event = ProgressiveForecastEvent(
            event="forecast.progress",
            stage=stage.index,
            strategy=stage.strategy,
            model=stage.model,
            family=stage.family,
            status="running",
            final=is_final_stage,
        )
        running_payload = running_event.model_dump(mode="json", exclude_none=True)
        yield format_sse_event(event=running_event.event, data=running_payload)
        _publish_event(
            app=app,
            key_id=key_id,
            event=running_event.event,
            data=running_payload,
        )

        forecast_payload = _auto_payload_to_forecast_payload(
            payload=payload,
            model=stage.model,
            default_timeout=AUTO_FORECAST_MEMBER_TIMEOUT_SECONDS,
        )
        try:
            response = _execute_forecast(app, payload=forecast_payload, request=request)
        except HTTPException as exc:
            failed_event = ProgressiveForecastEvent(
                event="forecast.complete",
                stage=stage.index,
                strategy=stage.strategy,
                model=stage.model,
                family=stage.family,
                status="failed",
                final=is_final_stage,
                error=_compare_error_message(exc),
            )
            failed_payload = failed_event.model_dump(mode="json", exclude_none=True)
            yield format_sse_event(event=failed_event.event, data=failed_payload)
            _publish_event(
                app=app,
                key_id=key_id,
                event=failed_event.event,
                data=failed_payload,
            )
            if is_final_stage:
                return
            continue

        completed_event = ProgressiveForecastEvent(
            event="forecast.complete",
            stage=stage.index,
            strategy=stage.strategy,
            model=stage.model,
            family=stage.family,
            status="completed",
            final=is_final_stage,
            response=response,
        )
        completed_payload = completed_event.model_dump(mode="json", exclude_none=True)
        yield format_sse_event(event=completed_event.event, data=completed_payload)
        _publish_event(
            app=app,
            key_id=key_id,
            event=completed_event.event,
            data=completed_payload,
        )


def _resolve_progressive_stages(
    *,
    payload: AutoForecastRequest,
) -> tuple[tuple[ProgressiveStage, ...], list[str]]:
    manifests = list_installed()
    installed_by_name = {
        name: manifest
        for manifest in manifests
        if isinstance((name := manifest.get("name")), str) and name.strip()
    }
    if not installed_by_name:
        raise HTTPException(
            status_code=404,
            detail="no installed models available for progressive forecast",
        )

    available_models = sorted(installed_by_name)
    preferred_model: str | None = None
    plan_warnings: list[str] = []
    if payload.model is not None:
        if payload.model in installed_by_name:
            preferred_model = payload.model
        elif not payload.allow_fallback:
            raise HTTPException(
                status_code=404,
                detail=f"model {payload.model!r} is not installed",
            )
        else:
            plan_warnings.append(
                f"explicit model {payload.model!r} is not installed; "
                "using progressive auto-selection",
            )

    family_by_model: dict[str, str] = {}
    for model_name, manifest in installed_by_name.items():
        family = manifest.get("family")
        if isinstance(family, str) and family.strip():
            family_by_model[model_name] = family.strip()

    try:
        stages = build_progressive_stages(
            series=payload.series,
            horizon=payload.horizon,
            include_models=available_models,
            preferred_model=preferred_model,
            family_by_model=family_by_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return stages, plan_warnings


def _parse_event_type_filter(request: Request) -> set[str] | None:
    raw_values: list[str] = []
    raw_values.extend(request.query_params.getlist("event"))
    events_raw = request.query_params.get("events")
    if events_raw is not None:
        raw_values.append(events_raw)

    normalized = {
        item.strip()
        for chunk in raw_values
        for item in chunk.split(",")
        if item.strip()
    }
    return normalized or None


def _parse_positive_int_query_param(
    raw: str | None,
    *,
    key: str,
    default: int | None,
) -> int | None:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    if not normalized.isdigit():
        raise HTTPException(status_code=400, detail=f"{key} must be a positive integer")
    value = int(normalized)
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{key} must be a positive integer")
    return value


def _parse_positive_float_query_param(raw: str | None, *, key: str, default: float) -> float:
    if raw is None:
        return default
    normalized = raw.strip()
    if not normalized:
        return default
    try:
        value = float(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{key} must be a positive number") from exc
    if value <= 0:
        raise HTTPException(status_code=400, detail=f"{key} must be a positive number")
    return value


def _optional_event_stream(app: FastAPI) -> EventStream | None:
    event_stream = getattr(app.state, "event_stream", None)
    if isinstance(event_stream, EventStream):
        return event_stream
    return None


def _publish_event(
    *,
    app: FastAPI,
    key_id: str | None,
    event: str,
    data: dict[str, Any],
) -> None:
    event_stream = _optional_event_stream(app)
    if event_stream is None:
        return
    try:
        event_stream.publish(key_id=key_id, event=event, data=data)
    except Exception:  # noqa: BLE001
        return


def _event_key_id(request: Any) -> str:
    if request is None:
        return "anonymous"
    try:
        return current_key_id(request)
    except Exception:  # noqa: BLE001
        return "anonymous"


def _to_ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def _load_json_object(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("payload JSON must be an object")
    return payload


def _pull_stream_lines(
    *,
    name: str,
    accept_license: bool,
    pull_options: PullOptions,
) -> Iterator[str]:
    event_queue: Queue[dict[str, Any] | object] = Queue()
    done = object()

    def _run_pull() -> None:
        try:
            result = _pull_model_snapshot(
                name=name,
                accept_license=accept_license,
                pull_options=pull_options,
                progress_cb=event_queue.put,
            )
            event_queue.put(result)
        except HTTPException as exc:
            event_queue.put({"error": {"message": str(exc.detail)}})
        except Exception as exc:  # noqa: BLE001
            event_queue.put({"error": {"message": _friendly_pull_error_message(exc)}})
        finally:
            event_queue.put(done)

    thread = Thread(target=_run_pull, daemon=True)
    thread.start()

    while True:
        item = event_queue.get()
        if item is done:
            break
        if not isinstance(item, dict):
            continue
        yield _to_ndjson_line(item)


def _to_pull_options(payload: ApiPullRequest, *, config: TollamaConfig) -> PullOptions:
    fields_set = payload.model_fields_set
    pull_defaults = config.pull

    insecure, _insecure_explicit = _resolve_pull_bool(
        field_name="insecure",
        payload_value=payload.insecure,
        fields_set=fields_set,
        env_name=None,
        config_value=pull_defaults.insecure,
        hard_default=False,
    )
    offline, offline_explicit = _resolve_pull_bool(
        field_name="offline",
        payload_value=payload.offline,
        fields_set=fields_set,
        env_name="HF_HUB_OFFLINE",
        config_value=pull_defaults.offline,
        hard_default=False,
    )
    local_files_only, _local_files_only_explicit = _resolve_pull_bool(
        field_name="local_files_only",
        payload_value=payload.local_files_only,
        fields_set=fields_set,
        env_name=None,
        config_value=pull_defaults.local_files_only,
        hard_default=False,
    )
    if offline:
        local_files_only = True

    http_proxy, http_proxy_explicit = _resolve_pull_str(
        field_name="http_proxy",
        payload_value=payload.http_proxy,
        fields_set=fields_set,
        env_name="HTTP_PROXY",
        config_value=pull_defaults.http_proxy,
    )
    https_proxy, https_proxy_explicit = _resolve_pull_str(
        field_name="https_proxy",
        payload_value=payload.https_proxy,
        fields_set=fields_set,
        env_name="HTTPS_PROXY",
        config_value=pull_defaults.https_proxy,
    )
    no_proxy, no_proxy_explicit = _resolve_pull_str(
        field_name="no_proxy",
        payload_value=payload.no_proxy,
        fields_set=fields_set,
        env_name="NO_PROXY",
        config_value=pull_defaults.no_proxy,
    )
    hf_home, hf_home_explicit = _resolve_pull_str(
        field_name="hf_home",
        payload_value=payload.hf_home,
        fields_set=fields_set,
        env_name="HF_HOME",
        config_value=pull_defaults.hf_home,
    )

    max_workers = _resolve_pull_int(
        field_name="max_workers",
        payload_value=payload.max_workers,
        fields_set=fields_set,
        config_value=pull_defaults.max_workers,
        hard_default=8,
    )
    token = _resolve_pull_token(payload=payload, fields_set=fields_set)

    return PullOptions(
        insecure=insecure,
        offline=offline,
        offline_explicit=offline_explicit,
        local_files_only=local_files_only,
        http_proxy=http_proxy,
        http_proxy_explicit=http_proxy_explicit,
        https_proxy=https_proxy,
        https_proxy_explicit=https_proxy_explicit,
        no_proxy=no_proxy,
        no_proxy_explicit=no_proxy_explicit,
        hf_home=hf_home,
        hf_home_explicit=hf_home_explicit,
        max_workers=max_workers,
        token=token,
    )


def _resolve_pull_bool(
    *,
    field_name: str,
    payload_value: bool | None,
    fields_set: set[str],
    env_name: str | None,
    config_value: bool | None,
    hard_default: bool,
) -> tuple[bool, bool]:
    if field_name in fields_set and payload_value is not None:
        return bool(payload_value), True

    if env_name is not None:
        env_value = _optional_bool_str(os.environ.get(env_name))
        if env_value is not None:
            return env_value, False

    if config_value is not None:
        return bool(config_value), False

    return hard_default, False


def _resolve_forecast_timeout_seconds(explicit: float | None = None) -> float:
    if explicit is not None and explicit > 0:
        return explicit

    configured = _env_or_none(FORECAST_TIMEOUT_ENV_NAME)
    if configured is None:
        return DEFAULT_FORECAST_TIMEOUT_SECONDS

    try:
        resolved = float(configured)
    except ValueError:
        return DEFAULT_FORECAST_TIMEOUT_SECONDS
    if resolved <= 0:
        return DEFAULT_FORECAST_TIMEOUT_SECONDS
    return resolved


def _resolve_pull_str(
    *,
    field_name: str,
    payload_value: str | None,
    fields_set: set[str],
    env_name: str | None,
    config_value: str | None,
) -> tuple[str | None, bool]:
    if field_name in fields_set:
        return _optional_nonempty_str(payload_value), True

    if env_name is not None:
        env_value = _optional_nonempty_str(os.environ.get(env_name))
        if env_value is not None:
            return env_value, False

    return _optional_nonempty_str(config_value), False


def _resolve_pull_int(
    *,
    field_name: str,
    payload_value: int | None,
    fields_set: set[str],
    config_value: int | None,
    hard_default: int,
) -> int:
    if field_name in fields_set and payload_value is not None:
        return payload_value

    if config_value is not None:
        return config_value

    return hard_default


def _resolve_pull_token(*, payload: ApiPullRequest, fields_set: set[str]) -> str | None:
    if "token" in fields_set:
        return _optional_nonempty_str(payload.token)
    env_token = _optional_nonempty_str(os.environ.get("TOLLAMA_HF_TOKEN"))
    request_token = _optional_nonempty_str(payload.token)
    return request_token or env_token


def _validate_pull_request(*, name: str, accept_license: bool) -> None:
    _prepare_pull_manifest(name=name, accept_license=accept_license)


def _pull_model_snapshot(
    *,
    name: str,
    accept_license: bool,
    pull_options: PullOptions,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    spec, manifest, paths = _prepare_pull_manifest(
        name=name,
        accept_license=accept_license,
    )
    snapshot_existing = _manifest_snapshot_path(manifest)
    if snapshot_existing is not None:
        digest_existing = _manifest_digest(manifest) or "unknown"
        size_existing = _manifest_size_bytes(manifest)
        _emit_pull_event(progress_cb, {"status": "already present", "model": spec.name})
        return {
            "status": "success",
            "model": spec.name,
            "digest": digest_existing,
            "size": size_existing,
        }

    model_dir = paths.model_dir(spec.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_tmp = model_dir / "snapshot.tmp"
    snapshot_dir = model_dir / "snapshot"
    _remove_path(snapshot_tmp)

    env_mapping: dict[str, str | None] = {}
    if pull_options.hf_home is not None or pull_options.hf_home_explicit:
        env_mapping["HF_HOME"] = pull_options.hf_home
    if pull_options.http_proxy is not None or pull_options.http_proxy_explicit:
        env_mapping["HTTP_PROXY"] = pull_options.http_proxy
    if pull_options.https_proxy is not None or pull_options.https_proxy_explicit:
        env_mapping["HTTPS_PROXY"] = pull_options.https_proxy
    if pull_options.no_proxy is not None or pull_options.no_proxy_explicit:
        env_mapping["NO_PROXY"] = pull_options.no_proxy
    if pull_options.offline:
        env_mapping["HF_HUB_OFFLINE"] = "1"
    elif pull_options.offline_explicit:
        env_mapping["HF_HUB_OFFLINE"] = None

    try:
        with set_env_temporarily(env_mapping):
            if spec.source.type == "huggingface":
                commit_hint = _manifest_digest(manifest) or None
                size_hint = _manifest_size_bytes(manifest)
                commit_sha, _snapshot_path, size_bytes = pull_snapshot_to_local_dir(
                    repo_id=spec.source.repo_id,
                    revision=spec.source.revision,
                    local_dir=snapshot_tmp,
                    token=pull_options.token,
                    max_workers=pull_options.max_workers,
                    progress_cb=progress_cb,
                    local_files_only=pull_options.local_files_only,
                    offline=pull_options.offline,
                    insecure=pull_options.insecure,
                    known_commit_sha=commit_hint,
                    known_total_bytes=size_hint,
                )
                _replace_directory(source=snapshot_tmp, destination=snapshot_dir)
            else:
                _emit_pull_event(progress_cb, {"status": "pulling manifest"})
                commit_sha = "local"
                _emit_pull_event(progress_cb, {"status": "resolving digest", "digest": commit_sha})
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                size_bytes = 0

            manifest["resolved"] = {
                "commit_sha": commit_sha,
                "snapshot_path": str(snapshot_dir.resolve()),
            }
            manifest["size_bytes"] = size_bytes
            manifest["pulled_at"] = _utc_now_iso()
            write_manifest(spec.name, manifest, paths=paths)
            return {
                "status": "success",
                "model": spec.name,
                "digest": commit_sha,
                "size": size_bytes,
            }
    except Exception:
        _remove_path(snapshot_tmp)
        raise


def _prepare_pull_manifest(
    *,
    name: str,
    accept_license: bool,
) -> tuple[Any, dict[str, Any], TollamaPaths]:
    try:
        spec = get_model_spec(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    paths = TollamaPaths.default()
    existing = read_manifest(spec.name, paths=paths)
    previously_accepted = _manifest_license_accepted(existing)
    accepted = accept_license or previously_accepted or (not spec.license.needs_acceptance)
    if spec.license.needs_acceptance and not accepted:
        raise HTTPException(
            status_code=409,
            detail=(
                f"model {spec.name!r} requires license acceptance; "
                "pass accept_license=True"
            ),
        )

    resolved_existing = existing.get("resolved") if isinstance(existing, dict) else {}
    resolved_map = resolved_existing if isinstance(resolved_existing, dict) else {}
    installed_at = (
        existing.get("installed_at")
        if isinstance(existing, dict) and isinstance(existing.get("installed_at"), str)
        else _utc_now_iso()
    )
    pulled_at = (
        existing.get("pulled_at")
        if isinstance(existing, dict) and isinstance(existing.get("pulled_at"), str)
        else None
    )
    size_bytes = (
        existing.get("size_bytes")
        if isinstance(existing, dict)
        and isinstance(existing.get("size_bytes"), int)
        and existing.get("size_bytes") >= 0
        else 0
    )
    existing_license = existing.get("license") if isinstance(existing, dict) else {}
    existing_license_map = existing_license if isinstance(existing_license, dict) else {}
    accepted_at = _optional_nonempty_str(existing_license_map.get("accepted_at"))
    if accepted and accepted_at is None:
        accepted_at = _utc_now_iso()

    manifest = {
        "name": spec.name,
        "family": spec.family,
        "source": spec.source.model_dump(mode="json"),
        "resolved": {
            "commit_sha": _optional_nonempty_str(resolved_map.get("commit_sha")),
            "snapshot_path": _optional_nonempty_str(resolved_map.get("snapshot_path")),
        },
        "size_bytes": size_bytes,
        "pulled_at": pulled_at,
        "installed_at": installed_at,
        "license": {
            "type": spec.license.type,
            "needs_acceptance": spec.license.needs_acceptance,
            "accepted": accepted,
            "accepted_at": accepted_at,
        },
    }
    if spec.license.notice is not None:
        manifest["license"]["notice"] = spec.license.notice
    if spec.metadata is not None:
        manifest["metadata"] = spec.metadata
    if spec.capabilities is not None:
        manifest["capabilities"] = spec.capabilities.model_dump(mode="json")
    return spec, manifest, paths


def _runner_error_detail(exc: RunnerCallError | RunnerProtocolError) -> dict[str, Any] | str:
    if isinstance(exc, RunnerCallError):
        detail: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.data is not None:
            detail["data"] = exc.data
        return detail
    return str(exc)


def _http_error_body(*, status_code: int, detail: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"detail": detail}
    hint = _http_error_hint(status_code=status_code, detail=detail)
    if hint is not None:
        payload["hint"] = hint
    return payload


def _http_error_hint(*, status_code: int, detail: Any) -> str | None:
    detail_text = _http_error_detail_text(detail).lower()

    if status_code == 400:
        return "Fix request payload or parameters and retry."

    if status_code == 404:
        if "model" in detail_text and "not installed" in detail_text:
            return (
                "Run `tollama pull <model>` to install. "
                "Use `tollama info --json` to inspect available models."
            )
        return None

    if status_code in {401, 403, 409}:
        if (
            "license" in detail_text
            or "accept_license" in detail_text
            or "accept-license" in detail_text
        ):
            return (
                "Re-run with `--accept-license`, or call "
                "`tollama pull <model> --accept-license`."
            )
        return None

    if status_code in {408, 504}:
        return "Try a smaller series or increase timeout."

    if status_code == 503:
        if "runner unavailable" in detail_text:
            return "Install and start the required runner family, then retry."
        return "Retry after daemon dependencies are available."

    return None


def _http_error_detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
        try:
            return json.dumps(detail, separators=(",", ":"), sort_keys=True)
        except TypeError:
            return str(detail)
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict):
                msg = item.get("msg")
                if isinstance(msg, str) and msg:
                    parts.append(msg)
                    continue
            parts.append(str(item))
        return "; ".join(parts)
    return str(detail)


def _format_validation_errors(errors: list[dict[str, Any]]) -> list[str]:
    formatted: list[str] = []
    for error in errors:
        location = _format_error_location(error.get("loc"))
        message = str(error.get("msg") or "invalid value")
        entry = f"field '{location}': {message}"
        if "input" in error:
            entry = f"{entry} (got: {error.get('input')!r})"
        formatted.append(entry)
    return formatted


def _format_error_location(location: Any) -> str:
    if not isinstance(location, (list, tuple)):
        return "<root>"

    parts: list[str] = []
    for item in location:
        if item == "body":
            continue
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
            continue
        parts.append(str(item))

    if not parts:
        return "<root>"
    return ".".join(parts)


def _install_model(*, name: str, accept_license: bool) -> dict[str, Any]:
    try:
        return install_from_registry(name, accept_license=accept_license)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _remove_model_or_404(name: str) -> None:
    try:
        removed = remove_model(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not removed:
        raise HTTPException(status_code=404, detail=f"model {name!r} is not installed")


def _find_installed_manifest(name: str) -> dict[str, Any] | None:
    for manifest in list_installed():
        installed_name = manifest.get("name")
        if installed_name == name:
            return manifest
    return None


def _require_installed_manifest(name: str) -> dict[str, Any]:
    manifest = _find_installed_manifest(name)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"model {name!r} is not installed",
        )
    return manifest


def _manifest_family_or_500(manifest: dict[str, Any], model_name: str) -> str:
    family = manifest.get("family")
    if isinstance(family, str) and family:
        return family
    raise HTTPException(
        status_code=500,
        detail=f"model {model_name!r} has invalid family metadata",
    )


def _to_ollama_tag_model(manifest: dict[str, Any]) -> dict[str, Any]:
    name = manifest.get("name")
    model_name = name if isinstance(name, str) else ""

    family = manifest.get("family")
    family_name = family if isinstance(family, str) else ""
    families = [family_name] if family_name else []

    digest_value = _manifest_digest(manifest)
    size_value = _manifest_size_bytes(manifest)

    return {
        "name": model_name,
        "model": model_name,
        "modified_at": _normalize_modified_at(
            manifest.get("pulled_at") or manifest.get("installed_at"),
        ),
        "size": size_value,
        "digest": digest_value,
        "details": {
            "format": "tollama",
            "family": family_name,
            "families": families,
            "parameter_size": "",
            "quantization_level": "",
        },
    }


def _normalize_modified_at(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_MODIFIED_AT

    normalized = value.strip()
    if not normalized:
        return DEFAULT_MODIFIED_AT

    parse_value = normalized[:-1] + "+00:00" if normalized.endswith("Z") else normalized
    try:
        datetime.fromisoformat(parse_value)
    except ValueError:
        return DEFAULT_MODIFIED_AT
    return normalized


def _manifest_digest(manifest: dict[str, Any]) -> str:
    resolved = manifest.get("resolved")
    if isinstance(resolved, dict):
        commit_sha = resolved.get("commit_sha")
        if isinstance(commit_sha, str):
            normalized = commit_sha.strip()
            if normalized:
                return normalized

    digest = manifest.get("digest")
    if isinstance(digest, str):
        return digest
    return ""


def _manifest_snapshot_path(manifest: dict[str, Any]) -> str | None:
    resolved = manifest.get("resolved")
    if not isinstance(resolved, dict):
        return None
    raw_path = resolved.get("snapshot_path")
    if not isinstance(raw_path, str):
        return None
    normalized = raw_path.strip()
    if not normalized:
        return None
    if not Path(normalized).exists():
        return None
    return normalized


def _manifest_source(manifest: dict[str, Any]) -> dict[str, Any] | None:
    source = manifest.get("source")
    if not isinstance(source, dict):
        return None
    return source


def _manifest_metadata(manifest: dict[str, Any]) -> dict[str, Any] | None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata


def _manifest_capabilities(manifest: dict[str, Any]) -> ModelCapabilities | None:
    raw = manifest.get("capabilities")
    if not isinstance(raw, dict):
        return None
    try:
        return ModelCapabilities.model_validate(raw)
    except ValidationError:
        return None


def _resolve_model_capabilities(
    model_name: str,
    manifest: dict[str, Any],
) -> ModelCapabilities | None:
    from_manifest = _manifest_capabilities(manifest)
    if from_manifest is not None:
        return from_manifest

    try:
        spec = get_model_spec(model_name)
    except KeyError:
        return None
    return spec.capabilities


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _metadata_positive_int(metadata: dict[str, Any] | None, keys: tuple[str, ...]) -> int | None:
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value > 0 and value.is_integer():
            return int(value)
    return None


def _optional_model_spec(model_name: str) -> ModelSpec | None:
    try:
        return get_model_spec(model_name)
    except KeyError:
        return None


def _horizon_limit_from_metadata(metadata: dict[str, Any] | None) -> int | None:
    return _metadata_positive_int(metadata, ("max_horizon", "prediction_length"))


def _context_limit_from_metadata(metadata: dict[str, Any] | None) -> int | None:
    return _metadata_positive_int(
        metadata,
        ("max_context", "context_length", "default_context_length"),
    )


def _series_history_length(series_list: list[Any]) -> int:
    if not series_list:
        return 0
    return max(len(series.target) for series in series_list)


def _required_future_covariates(series_list: list[Any]) -> tuple[bool, bool]:
    requires_numeric = False
    requires_categorical = False
    for series in series_list:
        profile = build_covariate_profile(series)
        if profile.known_future_numeric:
            requires_numeric = True
        if profile.known_future_categorical:
            requires_categorical = True
    return requires_numeric, requires_categorical


def _suggest_model_candidate(
    *,
    exclude_model: str,
    min_horizon: int | None = None,
    min_context: int | None = None,
    require_future_numeric: bool = False,
    require_future_categorical: bool = False,
) -> ModelSpec | None:
    installed_names: set[str] = set()
    try:
        for manifest in list_installed():
            if not isinstance(manifest, dict):
                continue
            name = manifest.get("name")
            if isinstance(name, str) and name:
                installed_names.add(name)
    except Exception:  # noqa: BLE001
        pass

    candidates = [spec for spec in list_registry_models() if spec.name != exclude_model]
    candidates.sort(key=lambda spec: (0 if spec.name in installed_names else 1, spec.name))

    for spec in candidates:
        capabilities = spec.capabilities or ModelCapabilities()
        if require_future_numeric and not capabilities.future_covariates_numeric:
            continue
        if require_future_categorical and not capabilities.future_covariates_categorical:
            continue

        metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
        if min_horizon is not None:
            limit = _horizon_limit_from_metadata(metadata)
            if limit is not None and limit < min_horizon:
                continue
        if min_context is not None:
            limit = _context_limit_from_metadata(metadata)
            if limit is not None and limit < min_context:
                continue
        return spec
    return None


def _validation_error_suggestions(
    *,
    error_message: str,
    request: ForecastRequestWithKeepAlive,
    model_capabilities: ModelCapabilities | None,
) -> list[str]:
    suggestions: list[str] = []
    normalized = error_message.lower()

    requires_future_numeric, requires_future_categorical = _required_future_covariates(
        request.series,
    )
    capabilities = model_capabilities or ModelCapabilities()
    missing_future_support = (
        (requires_future_numeric and not capabilities.future_covariates_numeric)
        or (requires_future_categorical and not capabilities.future_covariates_categorical)
    )

    if "does not support" in normalized or missing_future_support:
        if request.parameters.covariates_mode == "strict":
            suggestions.append(
                "Switch to parameters.covariates_mode='best_effort' to drop unsupported covariates "
                "automatically.",
            )
        alternative = _suggest_model_candidate(
            exclude_model=request.model,
            require_future_numeric=requires_future_numeric,
            require_future_categorical=requires_future_categorical,
        )
        if alternative is not None:
            suggestions.append(
                f"Consider switching to {alternative.name!r}, which supports the requested "
                "future covariate shape.",
            )

    if "horizon" in normalized:
        alternative = _suggest_model_candidate(
            exclude_model=request.model,
            min_horizon=request.horizon,
            require_future_numeric=requires_future_numeric,
            require_future_categorical=requires_future_categorical,
        )
        if alternative is not None:
            suggestions.append(
                f"Consider model {alternative.name!r} for horizon {request.horizon}.",
            )

    return _dedupe_preserve_order(suggestions)


def _validation_suggestions(
    *,
    request: ForecastRequestWithKeepAlive,
    model_manifest: dict[str, Any] | None,
    model_capabilities: ModelCapabilities | None,
    normalized_series: list[Any],
    warnings: list[str],
) -> list[str]:
    del warnings
    suggestions: list[str] = []
    suggest_alternative = False
    capabilities = model_capabilities or ModelCapabilities()

    spec = _optional_model_spec(request.model)
    metadata: dict[str, Any] | None = None
    if model_manifest is not None:
        metadata = _manifest_metadata(model_manifest)
    if metadata is None and spec is not None and isinstance(spec.metadata, dict):
        metadata = spec.metadata

    horizon_limit = _horizon_limit_from_metadata(metadata)
    if horizon_limit is not None and request.horizon > horizon_limit:
        suggestions.append(
            f"Requested horizon {request.horizon} exceeds {request.model!r} declared limit "
            f"({horizon_limit}).",
        )
        suggest_alternative = True

    history_length = _series_history_length(normalized_series)
    context_limit = _context_limit_from_metadata(metadata)
    if context_limit is not None and history_length > context_limit:
        suggestions.append(
            f"Series history length {history_length} exceeds {request.model!r} declared context "
            f"limit ({context_limit}).",
        )
        suggest_alternative = True

    requires_future_numeric, requires_future_categorical = _required_future_covariates(
        normalized_series,
    )
    missing_future_support = (
        (requires_future_numeric and not capabilities.future_covariates_numeric)
        or (requires_future_categorical and not capabilities.future_covariates_categorical)
    )
    if missing_future_support:
        suggestions.append(
            f"Model {request.model!r} does not support the provided future_covariates shape.",
        )
        if request.parameters.covariates_mode == "strict":
            suggestions.append(
                "Use parameters.covariates_mode='best_effort' to keep running while dropping "
                "unsupported covariates.",
            )
        suggest_alternative = True

    if suggest_alternative:
        alternative = _suggest_model_candidate(
            exclude_model=request.model,
            min_horizon=(
                request.horizon
                if horizon_limit is not None and request.horizon > horizon_limit
                else None
            ),
            min_context=(
                history_length
                if context_limit is not None and history_length > context_limit
                else None
            ),
            require_future_numeric=requires_future_numeric,
            require_future_categorical=requires_future_categorical,
        )
        if alternative is not None:
            suggestions.append(
                f"Alternative model candidate: {alternative.name!r}.",
            )

    return _dedupe_preserve_order(suggestions)


def _manifest_size_bytes(manifest: dict[str, Any]) -> int:
    size_bytes = manifest.get("size_bytes")
    if isinstance(size_bytes, int) and size_bytes >= 0:
        return size_bytes

    size = manifest.get("size")
    if isinstance(size, int) and size >= 0:
        return size
    return 0


def _manifest_license_accepted(manifest: dict[str, Any] | None) -> bool:
    if not isinstance(manifest, dict):
        return False
    license_info = manifest.get("license")
    if not isinstance(license_info, dict):
        return False
    return bool(license_info.get("accepted"))


def _public_license_view(license_payload: Any) -> dict[str, Any]:
    if not isinstance(license_payload, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in ("type", "needs_acceptance", "accepted"):
        if key in license_payload:
            normalized[key] = license_payload[key]
    if "notice" in license_payload:
        normalized["notice"] = license_payload["notice"]
    return normalized


def _merge_warnings(
    runner_warnings: list[str] | None,
    daemon_warnings: list[str],
) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()
    for warning in (runner_warnings or []):
        if warning in seen:
            continue
        seen.add(warning)
        merged.append(warning)
    for warning in daemon_warnings:
        if warning in seen:
            continue
        seen.add(warning)
        merged.append(warning)
    return merged or None


def _dedupe_messages(messages: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        normalized = message.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _optional_nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _optional_bool_str(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _collect_redacted_config_payload(paths: TollamaPaths) -> dict[str, Any] | None:
    config_path = paths.config_path
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"invalid JSON in {config_path}: {exc.msg}"}
    except OSError as exc:
        return {"error": f"unable to read {config_path}: {exc}"}

    if not isinstance(payload, dict):
        return {"error": f"invalid config in {config_path}: top-level JSON must be an object"}
    return redact_config_dict(payload)


def _load_config_or_default(paths: TollamaPaths) -> TollamaConfig:
    try:
        return load_config(paths)
    except ConfigFileError:
        return TollamaConfig()


def _collect_info_env() -> dict[str, Any]:
    payload = {key: _env_or_none(key) for key in INFO_ENV_KEYS}
    for key in TOKEN_ENV_KEYS:
        payload[f"{key}_present"] = _env_or_none(key) is not None
    return redact_env_dict(payload)


def _collect_pull_defaults(*, config: TollamaConfig) -> dict[str, dict[str, Any]]:
    defaults = resolve_effective_pull_defaults(env=os.environ, config=config)
    redacted: dict[str, dict[str, Any]] = {}
    for key, detail in defaults.items():
        value = detail.get("value")
        if isinstance(value, str) and "proxy" in key:
            value = redact_proxy_url(value)
        redacted[key] = {
            "value": value,
            "source": detail.get("source"),
        }
    return redacted


def _collect_installed_model_entries(*, paths: TollamaPaths) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        manifests = list_installed(paths=paths)
    except Exception:  # noqa: BLE001
        return []

    for manifest in manifests:
        if isinstance(manifest, dict):
            entries.append(_model_from_manifest_for_info(manifest))
    return entries


def _collect_available_model_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in list_registry_models():
        entry: dict[str, Any] = {
            "name": spec.name,
            "family": spec.family,
            "source": spec.source.model_dump(mode="json"),
        }
        if spec.capabilities is not None:
            entry["capabilities"] = spec.capabilities.model_dump(mode="json")
        entries.append(entry)
    return entries


def _collect_loaded_model_entries(app: FastAPI) -> list[dict[str, Any]]:
    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    models: list[dict[str, Any]] = []
    for loaded in tracker.list_models():
        models.append(
            {
                "name": loaded.name,
                "model": loaded.model,
                "family": loaded.family,
                "expires_at": to_utc_iso(loaded.expires_at),
            },
        )
    return models


def _model_from_manifest_for_info(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": _optional_nonempty_str(manifest.get("name")),
        "family": _optional_nonempty_str(manifest.get("family")),
        "digest": _manifest_digest(manifest) or None,
        "size": _manifest_size_bytes(manifest),
        "modified_at": _optional_nonempty_str(
            manifest.get("pulled_at") or manifest.get("installed_at"),
        ),
    }
    capabilities = _manifest_capabilities(manifest)
    if capabilities is None:
        model_name = _optional_nonempty_str(manifest.get("name"))
        if model_name is not None:
            try:
                spec = get_model_spec(model_name)
            except KeyError:
                spec = None
            if spec is not None:
                capabilities = spec.capabilities
    if capabilities is not None:
        payload["capabilities"] = capabilities.model_dump(mode="json")
    return payload


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


def _uptime_seconds(*, started_at: datetime | None, now: datetime) -> int:
    if started_at is None:
        return 0
    seconds = (now - started_at).total_seconds()
    return max(0, int(seconds))


def _docs_public_enabled() -> bool:
    return _env_flag(DOCS_PUBLIC_ENV_NAME, default=False)


def _resolve_cors_origins_from_env() -> tuple[str, ...]:
    raw = _env_or_none(CORS_ORIGINS_ENV_NAME)
    if raw is None:
        return ()

    origins: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        origin = item.strip()
        if not origin or origin in seen:
            continue
        seen.add(origin)
        origins.append(origin)
    return tuple(origins)


def _dashboard_enabled() -> bool:
    return _env_flag(DASHBOARD_ENABLED_ENV_NAME, default=True)


def _dashboard_require_auth_enabled() -> bool:
    return _env_flag(DASHBOARD_REQUIRE_AUTH_ENV_NAME, default=False)


def _resolve_dashboard_static_dir(app: FastAPI) -> Path:
    resource_stack = getattr(app.state, "dashboard_resource_stack", None)
    if not isinstance(resource_stack, ExitStack):
        raise RuntimeError("dashboard resources are unavailable")

    try:
        static_resource = resources.files("tollama.dashboard").joinpath("static")
    except ModuleNotFoundError as exc:
        raise RuntimeError("dashboard package is unavailable") from exc
    static_dir = Path(resource_stack.enter_context(resources.as_file(static_resource)))
    if not static_dir.exists():
        raise RuntimeError("dashboard static assets are unavailable")
    return static_dir


def _dashboard_index_path(app: FastAPI) -> Path:
    index_path = getattr(app.state, "dashboard_index_path", None)
    if isinstance(index_path, Path) and index_path.exists():
        return index_path
    raise HTTPException(status_code=503, detail="dashboard assets are unavailable")


def _resolve_host_binding_from_env() -> str | None:
    effective = _env_or_none("TOLLAMA_EFFECTIVE_HOST_BINDING")
    if effective is not None:
        return effective

    host_binding = _env_or_none("TOLLAMA_HOST")
    if host_binding is not None:
        return host_binding

    port_value = _env_or_none("TOLLAMA_PORT")
    if port_value is not None:
        return f"127.0.0.1:{port_value}"
    return None


def _env_or_none(name: str) -> str | None:
    return _optional_nonempty_str(os.environ.get(name))


def _env_flag(name: str, *, default: bool) -> bool:
    raw = _env_or_none(name)
    if raw is None:
        return default
    resolved = _optional_bool_str(raw)
    if resolved is None:
        return default
    return resolved


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _emit_pull_event(
    progress_cb: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if progress_cb is None:
        return
    progress_cb(payload)


def _friendly_pull_error_message(exc: Exception) -> str:
    if isinstance(exc, OfflineModelUnavailableError):
        return str(exc)
    if isinstance(exc, PullError):
        return str(exc)

    base_message = _compact_exception_message(exc)
    network_error = isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ProxyError,
            httpx.ReadTimeout,
            httpx.TimeoutException,
        ),
    ) or _looks_like_network_failure(base_message)
    if not network_error:
        return base_message

    return (
        f"{base_message}. "
        "If behind a proxy, set HTTP_PROXY/HTTPS_PROXY/NO_PROXY or use "
        "--http-proxy/--https-proxy. "
        "If your network does TLS interception, configure a trusted CA bundle "
        "(preferred) or use --insecure for debugging."
    )


def _compact_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _looks_like_network_failure(message: str) -> bool:
    lowered = message.lower()
    indicators = (
        "connecterror",
        "connection refused",
        "connection reset",
        "connection timed out",
        "name or service not known",
        "proxy",
        "ssl",
        "tls",
        "network is unreachable",
        "temporary failure in name resolution",
        "timed out",
    )
    return any(token in lowered for token in indicators)


def _replace_directory(*, source: Path, destination: Path) -> None:
    _remove_path(destination)
    if source.exists():
        source.replace(destination)
        return
    destination.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink()
    except OSError:
        return


def _unload_expired_models(app: FastAPI) -> None:
    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    now = datetime.now(UTC)
    expired_runners = tracker.expired_runners(now)
    for runner in expired_runners:
        app.state.runner_manager.stop(family=runner)
        tracker.unload_runner(runner)


def _resolve_package_version() -> str:
    try:
        return metadata.version("tollama")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _build_default_runner_manager() -> RunnerManager:
    """Build a RunnerManager using settings from ``config.json``."""
    try:
        paths = TollamaPaths.default()
        config = load_config(paths)
    except Exception:
        return RunnerManager()

    daemon_cfg = config.daemon
    runner_commands = daemon_cfg.runner_commands or None
    return RunnerManager(
        runner_commands=runner_commands,
        auto_bootstrap=daemon_cfg.auto_bootstrap,
        paths=paths,
    )


def _build_prometheus_metrics(app: FastAPI) -> PrometheusMetrics | None:
    return create_prometheus_metrics(
        get_loaded_models=lambda: _count_loaded_models(app),
        get_runner_restarts=lambda: _count_runner_restarts(app),
    )


def _build_usage_meter() -> UsageMeter | None:
    try:
        paths = TollamaPaths.default()
        return create_usage_meter(db_path=paths.base_dir / "usage.db")
    except Exception:  # noqa: BLE001
        return None


def _build_rate_limiter() -> TokenBucketRateLimiter | None:
    return create_rate_limiter_from_env(env=os.environ)


def _optional_prometheus_metrics(app: FastAPI) -> PrometheusMetrics | None:
    metrics = getattr(app.state, "prometheus_metrics", None)
    if isinstance(metrics, PrometheusMetrics):
        return metrics
    return None


def _optional_usage_meter(app: FastAPI) -> UsageMeter | None:
    usage_meter = getattr(app.state, "usage_meter", None)
    if isinstance(usage_meter, UsageMeter):
        return usage_meter
    return None


def _optional_rate_limiter(app: FastAPI) -> TokenBucketRateLimiter | None:
    limiter = getattr(app.state, "rate_limiter", None)
    if isinstance(limiter, TokenBucketRateLimiter):
        return limiter
    return None


def _count_loaded_models(app: FastAPI) -> int:
    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    return len(tracker.list_models())


def _count_runner_restarts(app: FastAPI) -> int:
    statuses = app.state.runner_manager.get_all_statuses()
    total = 0
    for status in statuses:
        restarts = status.get("restarts") if isinstance(status, dict) else None
        if isinstance(restarts, bool):
            continue
        if isinstance(restarts, int) and restarts > 0:
            total += restarts
            continue
        if isinstance(restarts, float) and restarts > 0 and restarts.is_integer():
            total += int(restarts)
    return total


app = create_app()
