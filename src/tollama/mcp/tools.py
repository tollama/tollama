"""MCP tool handlers backed by the shared tollama HTTP client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tollama.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    ModelMissingError,
    TollamaClient,
    TollamaClientError,
)
from tollama.core.recommend import recommend_models
from tollama.core.schemas import (
    AnalyzeRequest,
    AutoForecastRequest,
    CompareRequest,
    CounterfactualRequest,
    ForecastRequest,
    GenerateRequest,
    PipelineRequest,
    ReportRequest,
    ScenarioTreeRequest,
    WhatIfRequest,
)

from .schemas import (
    AnalyzeToolInput,
    AutoForecastToolInput,
    CompareToolInput,
    CounterfactualToolInput,
    ForecastToolInput,
    GenerateToolInput,
    HealthToolInput,
    ModelsToolInput,
    PipelineToolInput,
    PullToolInput,
    RecommendToolInput,
    ReportToolInput,
    ScenarioTreeToolInput,
    ShowToolInput,
    WhatIfToolInput,
)


@dataclass(slots=True)
class MCPToolError(RuntimeError):
    """Error surfaced by MCP tool handlers with exit-code-aligned metadata."""

    category: str
    exit_code: int
    message: str
    hint: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


def _make_client(*, base_url: str | None, timeout: float | None) -> TollamaClient:
    return TollamaClient(
        base_url=base_url or DEFAULT_BASE_URL,
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS,
    )


def _raise_from_client_error(exc: TollamaClientError) -> None:
    raise MCPToolError(
        category=exc.category,
        exit_code=exc.exit_code,
        message=str(exc),
        hint=exc.hint,
    ) from exc


def _raise_invalid_request(detail: str) -> None:
    raise MCPToolError(category="INVALID_REQUEST", exit_code=2, message=detail)


def tollama_health(*, base_url: str | None = None, timeout: float | None = None) -> dict[str, Any]:
    """Return daemon health payload for MCP consumers."""
    try:
        args = HealthToolInput(base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        payload = client.health()
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return {
        "healthy": True,
        "health": payload.get("health", {}),
        "version": payload.get("version", {}),
    }


def tollama_models(
    *,
    mode: str = "installed",
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Return model list payload for installed/loaded/available modes."""
    try:
        args = ModelsToolInput(mode=mode, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        items = client.models(mode=args.mode)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return {
        "mode": args.mode,
        "items": items,
    }


def tollama_forecast(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run a non-streaming forecast and return canonical response payload."""
    try:
        args = ForecastToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        forecast_request = ForecastRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.forecast_response(forecast_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_auto_forecast(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run non-streaming auto-forecast and return canonical response payload."""
    try:
        args = AutoForecastToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        auto_request = AutoForecastRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.auto_forecast(auto_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_analyze(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Analyze one or more series and return canonical analysis payload."""
    try:
        args = AnalyzeToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        analyze_request = AnalyzeRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.analyze(analyze_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_generate(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate synthetic series and return canonical response payload."""
    try:
        args = GenerateToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        generate_request = GenerateRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.generate(generate_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_counterfactual(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate intervention counterfactuals and return canonical response payload."""
    try:
        args = CounterfactualToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        counterfactual_request = CounterfactualRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.counterfactual(counterfactual_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_scenario_tree(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Generate probabilistic scenario tree and return canonical response payload."""
    try:
        args = ScenarioTreeToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        scenario_tree_request = ScenarioTreeRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.scenario_tree(scenario_tree_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_report(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run composite report endpoint and return canonical response payload."""
    try:
        args = ReportToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        report_request = ReportRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.report(report_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_what_if(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run what-if scenario analysis and return canonical response payload."""
    try:
        args = WhatIfToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        what_if_request = WhatIfRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.what_if(what_if_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_pipeline(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run autonomous pipeline flow and return analysis + recommendation + forecast output."""
    try:
        args = PipelineToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        pipeline_request = PipelineRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.pipeline(pipeline_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_compare(
    *,
    request: dict[str, Any],
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one request across multiple models and return per-model outcomes."""
    try:
        args = CompareToolInput(request=request, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    try:
        compare_request = CompareRequest.model_validate(args.request)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        response = client.compare(compare_request)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)

    return response.model_dump(mode="json", exclude_none=True)


def tollama_pull(
    *,
    model: str,
    accept_license: bool = False,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Pull a model in non-stream mode."""
    try:
        args = PullToolInput(
            model=model,
            accept_license=accept_license,
            base_url=base_url,
            timeout=timeout,
        )
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        return client.pull(args.model, accept_license=args.accept_license)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)


def tollama_show(
    *,
    model: str,
    base_url: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Fetch model detail payload."""
    try:
        args = ShowToolInput(model=model, base_url=base_url, timeout=timeout)
    except ValidationError as exc:
        _raise_invalid_request(str(exc))

    client = _make_client(base_url=args.base_url, timeout=args.timeout)
    try:
        return client.show(args.model)
    except ModelMissingError as exc:
        _raise_from_client_error(exc)
    except TollamaClientError as exc:
        _raise_from_client_error(exc)


def tollama_recommend(
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
    """Recommend models based on request characteristics."""
    try:
        args = RecommendToolInput(
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
        _raise_invalid_request(str(exc))

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
        _raise_invalid_request(str(exc))
