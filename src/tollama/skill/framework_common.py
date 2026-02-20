"""Shared tool-spec adapters for non-LangChain agent frameworks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from tollama.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    TollamaClient,
    TollamaClientError,
)
from tollama.core.recommend import recommend_models
from tollama.core.schemas import (
    AnalyzeRequest,
    AutoForecastRequest,
    CompareRequest,
    ForecastRequest,
)

_MODEL_NAME_EXAMPLES = (
    "mock, chronos2, granite-ttm-r2, timesfm-2.5-200m, "
    "moirai-2.0-R-small, sundial-base-128m, toto-open-base-1.0"
)


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    """Framework-agnostic tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., dict[str, Any]]


class _ToolInputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _ModelsInput(_ToolInputBase):
    mode: Literal["installed", "loaded", "available"] = "installed"


class _ForecastInput(_ToolInputBase):
    request: dict[str, Any]


class _AutoForecastInput(_ToolInputBase):
    request: dict[str, Any]


class _AnalyzeInput(_ToolInputBase):
    request: dict[str, Any]


class _CompareInput(_ToolInputBase):
    request: dict[str, Any]


class _RecommendInput(_ToolInputBase):
    horizon: int
    freq: str | None = None
    has_past_covariates: bool = False
    has_future_covariates: bool = False
    has_static_covariates: bool = False
    covariates_type: Literal["numeric", "categorical"] = "numeric"
    allow_restricted_license: bool = False
    top_k: int = 3


def get_agent_tool_specs(
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[AgentToolSpec]:
    """Build a framework-neutral toolset for agent wrappers."""
    client = TollamaClient(base_url=base_url, timeout=timeout)

    return [
        AgentToolSpec(
            name="tollama_health",
            description=(
                "Check tollama daemon status and version before forecasting tasks. "
                "No inputs required."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _health_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_models",
            description=(
                "List models by mode (installed, loaded, available). "
                f"Model examples: {_MODEL_NAME_EXAMPLES}."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["installed", "loaded", "available"],
                        "default": "installed",
                    },
                },
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _models_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_forecast",
            description=(
                "Run a non-streaming forecast request. "
                "Requires request.model, request.horizon, request.series[]."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "request": {"type": "object"},
                },
                "required": ["request"],
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _forecast_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_auto_forecast",
            description=(
                "Run zero-config auto-forecast with model selection metadata. "
                "Requires request.horizon and request.series[]."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "request": {"type": "object"},
                },
                "required": ["request"],
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _auto_forecast_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_analyze",
            description=(
                "Analyze one or more series for frequency, seasonality, trend, "
                "anomalies, stationarity, and data quality. "
                "Requires request.series[]."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "request": {"type": "object"},
                },
                "required": ["request"],
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _analyze_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_compare",
            description=(
                "Run the same forecast request over multiple models and compare outcomes. "
                "Requires request.models, request.horizon, request.series[]."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "request": {"type": "object"},
                },
                "required": ["request"],
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _compare_handler(client=client, **kwargs),
        ),
        AgentToolSpec(
            name="tollama_recommend",
            description=(
                "Recommend models from registry metadata + capability matrix. "
                "Requires horizon. Optional covariate hints and top_k."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "horizon": {"type": "integer", "minimum": 1},
                    "freq": {"type": "string"},
                    "has_past_covariates": {"type": "boolean", "default": False},
                    "has_future_covariates": {"type": "boolean", "default": False},
                    "has_static_covariates": {"type": "boolean", "default": False},
                    "covariates_type": {
                        "type": "string",
                        "enum": ["numeric", "categorical"],
                        "default": "numeric",
                    },
                    "allow_restricted_license": {"type": "boolean", "default": False},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
                },
                "required": ["horizon"],
                "additionalProperties": False,
            },
            handler=lambda **kwargs: _recommend_handler(**kwargs),
        ),
    ]


def _error_payload(*, category: str, exit_code: int, message: str) -> dict[str, Any]:
    return {
        "error": {
            "category": category,
            "exit_code": exit_code,
            "message": message,
        }
    }


def _invalid_request_payload(detail: str) -> dict[str, Any]:
    return _error_payload(category="INVALID_REQUEST", exit_code=2, message=detail)


def _client_error_payload(exc: TollamaClientError) -> dict[str, Any]:
    return _error_payload(category=exc.category, exit_code=exc.exit_code, message=str(exc))


def _health_handler(*, client: TollamaClient) -> dict[str, Any]:
    try:
        payload = client.health()
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return {
        "healthy": True,
        "health": payload.get("health", {}),
        "version": payload.get("version", {}),
    }


def _models_handler(*, client: TollamaClient, mode: str = "installed") -> dict[str, Any]:
    try:
        args = _ModelsInput(mode=mode)
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        items = client.models(mode=args.mode)
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return {
        "mode": args.mode,
        "items": items,
    }


def _forecast_handler(*, client: TollamaClient, request: dict[str, Any]) -> dict[str, Any]:
    try:
        args = _ForecastInput(request=request)
        forecast_request = ForecastRequest.model_validate(args.request)
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        response = client.forecast_response(forecast_request)
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return response.model_dump(mode="json", exclude_none=True)


def _auto_forecast_handler(*, client: TollamaClient, request: dict[str, Any]) -> dict[str, Any]:
    try:
        args = _AutoForecastInput(request=request)
        auto_request = AutoForecastRequest.model_validate(args.request)
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        response = client.auto_forecast(auto_request)
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return response.model_dump(mode="json", exclude_none=True)


def _compare_handler(*, client: TollamaClient, request: dict[str, Any]) -> dict[str, Any]:
    try:
        args = _CompareInput(request=request)
        compare_request = CompareRequest.model_validate(args.request)
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        response = client.compare(compare_request)
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return response.model_dump(mode="json", exclude_none=True)


def _analyze_handler(*, client: TollamaClient, request: dict[str, Any]) -> dict[str, Any]:
    try:
        args = _AnalyzeInput(request=request)
        analyze_request = AnalyzeRequest.model_validate(args.request)
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        response = client.analyze(analyze_request)
    except TollamaClientError as exc:
        return _client_error_payload(exc)
    return response.model_dump(mode="json", exclude_none=True)


def _recommend_handler(
    *,
    horizon: int,
    freq: str | None = None,
    has_past_covariates: bool = False,
    has_future_covariates: bool = False,
    has_static_covariates: bool = False,
    covariates_type: str = "numeric",
    allow_restricted_license: bool = False,
    top_k: int = 3,
) -> dict[str, Any]:
    try:
        args = _RecommendInput(
            horizon=horizon,
            freq=freq,
            has_past_covariates=has_past_covariates,
            has_future_covariates=has_future_covariates,
            has_static_covariates=has_static_covariates,
            covariates_type=covariates_type,
            allow_restricted_license=allow_restricted_license,
            top_k=top_k,
        )
    except ValidationError as exc:
        return _invalid_request_payload(str(exc))

    try:
        return recommend_models(
            horizon=args.horizon,
            freq=args.freq,
            has_past_covariates=args.has_past_covariates,
            has_future_covariates=args.has_future_covariates,
            has_static_covariates=args.has_static_covariates,
            covariates_type=args.covariates_type,
            allow_restricted_license=args.allow_restricted_license,
            top_k=args.top_k,
        )
    except ValueError as exc:
        return _invalid_request_payload(str(exc))
