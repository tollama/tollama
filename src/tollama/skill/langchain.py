"""LangChain tool wrappers backed by the shared tollama HTTP client."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from tollama.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    AsyncTollamaClient,
    TollamaClient,
    TollamaClientError,
)
from tollama.core.recommend import recommend_models
from tollama.core.schemas import (
    AnalyzeRequest,
    AutoForecastRequest,
    CompareRequest,
    ForecastRequest,
    WhatIfRequest,
)

try:
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        'LangChain dependency is not installed. Install with: pip install "tollama[langchain]"',
    ) from exc

_MODEL_NAME_EXAMPLES = (
    "mock, chronos2, granite-ttm-r2, timesfm-2.5-200m, "
    "moirai-2.0-R-small, sundial-base-128m, toto-open-base-1.0"
)


class _ToolInputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HealthToolInput(_ToolInputBase):
    pass


class ModelsToolInput(_ToolInputBase):
    mode: Literal["installed", "loaded", "available"] = "installed"


class ForecastToolInput(_ToolInputBase):
    request: dict[str, Any]


class AutoForecastToolInput(_ToolInputBase):
    request: dict[str, Any]


class AnalyzeToolInput(_ToolInputBase):
    request: dict[str, Any]


class WhatIfToolInput(_ToolInputBase):
    request: dict[str, Any]


class CompareToolInput(_ToolInputBase):
    request: dict[str, Any]


class RecommendToolInput(_ToolInputBase):
    horizon: int
    freq: str | None = None
    has_past_covariates: bool = False
    has_future_covariates: bool = False
    has_static_covariates: bool = False
    covariates_type: Literal["numeric", "categorical"] = "numeric"
    allow_restricted_license: bool = False
    top_k: int = 3


def _make_client(*, base_url: str, timeout: float) -> TollamaClient:
    return TollamaClient(base_url=base_url, timeout=timeout)


def _make_async_client(*, base_url: str, timeout: float) -> AsyncTollamaClient:
    return AsyncTollamaClient(base_url=base_url, timeout=timeout)


def _error_payload(*, category: str, exit_code: int, message: str) -> dict[str, Any]:
    return {
        "error": {
            "category": category,
            "exit_code": exit_code,
            "message": message,
        }
    }


def _client_error_payload(exc: TollamaClientError) -> dict[str, Any]:
    return _error_payload(category=exc.category, exit_code=exc.exit_code, message=str(exc))


def _invalid_request_payload(detail: str) -> dict[str, Any]:
    return _error_payload(category="INVALID_REQUEST", exit_code=2, message=detail)


class _TollamaBaseTool(BaseTool):
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def _client(self) -> TollamaClient:
        return _make_client(base_url=self.base_url, timeout=float(self.timeout))

    def _async_client(self) -> AsyncTollamaClient:
        return _make_async_client(base_url=self.base_url, timeout=float(self.timeout))


class TollamaHealthTool(_TollamaBaseTool):
    """LangChain tool that returns tollama daemon health information."""

    name: str = "tollama_health"
    description: str = (
        "Check tollama daemon status and version. "
        "No required arguments; use optional tool config base_url/timeout. "
        "Use this before forecast calls. "
        "Model choices for follow-up operations include "
        f"{_MODEL_NAME_EXAMPLES}. "
        'Example: tool.invoke({}).'
    )
    args_schema: type[BaseModel] = HealthToolInput

    def _run(self, run_manager: Any | None = None) -> dict[str, Any]:
        del run_manager
        client = self._client()
        try:
            payload = client.health()
        except TollamaClientError as exc:
            return _client_error_payload(exc)
        return {
            "healthy": True,
            "health": payload.get("health", {}),
            "version": payload.get("version", {}),
        }

    async def _arun(self, run_manager: Any | None = None) -> dict[str, Any]:
        del run_manager
        client = self._async_client()
        try:
            payload = await client.health()
        except TollamaClientError as exc:
            return _client_error_payload(exc)
        return {
            "healthy": True,
            "health": payload.get("health", {}),
            "version": payload.get("version", {}),
        }


class TollamaModelsTool(_TollamaBaseTool):
    """LangChain tool that returns model lists for a requested mode."""

    name: str = "tollama_models"
    description: str = (
        "List tollama models by mode. "
        "Input schema: {mode?: 'installed'|'loaded'|'available'}. "
        "Use available mode to discover installable models "
        f"(for example: {_MODEL_NAME_EXAMPLES}). "
        "Use loaded mode for currently active runners. "
        'Example: tool.invoke({"mode":"available"}).'
    )
    args_schema: type[BaseModel] = ModelsToolInput

    def _run(self, mode: str = "installed", run_manager: Any | None = None) -> dict[str, Any]:
        del run_manager
        try:
            args = ModelsToolInput(mode=mode)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            items = client.models(mode=args.mode)
        except TollamaClientError as exc:
            return _client_error_payload(exc)
        return {
            "mode": args.mode,
            "items": items,
        }

    async def _arun(
        self,
        mode: str = "installed",
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = ModelsToolInput(mode=mode)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            items = await client.models(mode=args.mode)
        except TollamaClientError as exc:
            return _client_error_payload(exc)
        return {
            "mode": args.mode,
            "items": items,
        }


class TollamaForecastTool(_TollamaBaseTool):
    """LangChain tool that validates and executes non-streaming forecasts."""

    name: str = "tollama_forecast"
    description: str = (
        "Run a non-streaming forecast against tollama. "
        "Input schema: {request:{model,horizon,series,quantiles?,options?,parameters?}}. "
        "Each series item must include id, timestamps, and target. "
        "Model values include "
        f"{_MODEL_NAME_EXAMPLES}. "
        'Example: tool.invoke({"request":{"model":"chronos2","horizon":3,'
        '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
        '"target":[10,11]}],"options":{}}}).'
    )
    args_schema: type[BaseModel] = ForecastToolInput

    def _run(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = ForecastToolInput(request=request)
            forecast_request = ForecastRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            response = client.forecast_response(forecast_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)

    async def _arun(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = ForecastToolInput(request=request)
            forecast_request = ForecastRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            response = await client.forecast_response(forecast_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)


class TollamaAutoForecastTool(_TollamaBaseTool):
    """LangChain tool that validates and executes non-streaming auto-forecast requests."""

    name: str = "tollama_auto_forecast"
    description: str = (
        "Run zero-config auto-forecast against tollama (model optional). "
        "Input schema: {request:{horizon,series,strategy?,model?,allow_fallback?,"
        "ensemble_top_k?,quantiles?,options?,parameters?}}. "
        "Returns strategy, selection rationale, and canonical forecast payload. "
        'Example: tool.invoke({"request":{"horizon":3,"strategy":"auto",'
        '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
        '"target":[10,11]}],"options":{}}}).'
    )
    args_schema: type[BaseModel] = AutoForecastToolInput

    def _run(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = AutoForecastToolInput(request=request)
            auto_request = AutoForecastRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            response = client.auto_forecast(auto_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)

    async def _arun(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = AutoForecastToolInput(request=request)
            auto_request = AutoForecastRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            response = await client.auto_forecast(auto_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)


class TollamaAnalyzeTool(_TollamaBaseTool):
    """LangChain tool that validates and executes series analysis requests."""

    name: str = "tollama_analyze"
    description: str = (
        "Analyze one or more time series for cadence, seasonality, trend, anomalies, "
        "stationarity, and data quality. "
        "Input schema: {request:{series,parameters?}} where series[] entries include "
        "id, timestamps, and target. "
        'Example: tool.invoke({"request":{"series":[{"id":"s1","freq":"D",'
        '"timestamps":["2025-01-01","2025-01-02","2025-01-03"],"target":[10,11,12]}]}}).'
    )
    args_schema: type[BaseModel] = AnalyzeToolInput

    def _run(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = AnalyzeToolInput(request=request)
            analyze_request = AnalyzeRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            response = client.analyze(analyze_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)

    async def _arun(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = AnalyzeToolInput(request=request)
            analyze_request = AnalyzeRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            response = await client.analyze(analyze_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)


class TollamaWhatIfTool(_TollamaBaseTool):
    """LangChain tool that validates and executes what-if scenario requests."""

    name: str = "tollama_what_if"
    description: str = (
        "Run what-if scenario analysis from a base forecast request plus named transforms. "
        "Input schema: {request:{model,horizon,series,scenarios,quantiles?,options?,parameters?,"
        "continue_on_error?,keep_alive?}}. "
        "Each scenario item includes name and transforms[] where transform has "
        "operation (multiply|add|replace), field (target|past_covariates|future_covariates), "
        "optional key for covariates, optional series_id, and value. "
        'Example: tool.invoke({"request":{"model":"mock","horizon":2,'
        '"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
        '"target":[10,11]}],"scenarios":[{"name":"high_demand","transforms":'
        '[{"operation":"multiply","field":"target","value":1.2}]}],"options":{}}}).'
    )
    args_schema: type[BaseModel] = WhatIfToolInput

    def _run(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = WhatIfToolInput(request=request)
            what_if_request = WhatIfRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            response = client.what_if(what_if_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)

    async def _arun(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = WhatIfToolInput(request=request)
            what_if_request = WhatIfRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            response = await client.what_if(what_if_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)


class TollamaCompareTool(_TollamaBaseTool):
    """LangChain tool that runs a single request across multiple models."""

    name: str = "tollama_compare"
    description: str = (
        "Compare multiple models using the same forecast request payload. "
        "Input schema: {request:{models,horizon,series,quantiles?,options?,timeout?,parameters?}}. "
        "Returns per-model result objects with ok=true/false and response/error payloads. "
        "Model values include "
        f"{_MODEL_NAME_EXAMPLES}. "
        'Example: tool.invoke({"request":{"models":["chronos2","timesfm-2.5-200m"],'
        '"horizon":3,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],'
        '"target":[10,11]}],"options":{}}}).'
    )
    args_schema: type[BaseModel] = CompareToolInput

    def _run(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = CompareToolInput(request=request)
            compare_request = CompareRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._client()
        try:
            response = client.compare(compare_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)

    async def _arun(
        self,
        request: dict[str, Any],
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
        try:
            args = CompareToolInput(request=request)
            compare_request = CompareRequest.model_validate(args.request)
        except ValidationError as exc:
            return _invalid_request_payload(str(exc))

        client = self._async_client()
        try:
            response = await client.compare(compare_request)
        except TollamaClientError as exc:
            return _client_error_payload(exc)

        return response.model_dump(mode="json", exclude_none=True)


class TollamaRecommendTool(_TollamaBaseTool):
    """LangChain tool that recommends models from registry metadata."""

    name: str = "tollama_recommend"
    description: str = (
        "Recommend tollama models for a forecasting task. "
        "Input schema: {horizon,freq?,has_past_covariates?,has_future_covariates?,"
        "has_static_covariates?,covariates_type?,allow_restricted_license?,top_k?}. "
        "Recommendations use registry metadata and capability compatibility. "
        "Candidate model names include "
        f"{_MODEL_NAME_EXAMPLES}. "
        'Example: tool.invoke({"horizon":48,"freq":"D","has_future_covariates":true,'
        '"covariates_type":"numeric","top_k":3}).'
    )
    args_schema: type[BaseModel] = RecommendToolInput

    def _run(
        self,
        horizon: int,
        freq: str | None = None,
        has_past_covariates: bool = False,
        has_future_covariates: bool = False,
        has_static_covariates: bool = False,
        covariates_type: str = "numeric",
        allow_restricted_license: bool = False,
        top_k: int = 3,
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
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

    async def _arun(
        self,
        horizon: int,
        freq: str | None = None,
        has_past_covariates: bool = False,
        has_future_covariates: bool = False,
        has_static_covariates: bool = False,
        covariates_type: str = "numeric",
        allow_restricted_license: bool = False,
        top_k: int = 3,
        run_manager: Any | None = None,
    ) -> dict[str, Any]:
        del run_manager
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


def get_tollama_tools(
    base_url: str = "http://127.0.0.1:11435",
    timeout: float = 10.0,
) -> list[BaseTool]:
    """Build the default tollama LangChain tool set."""
    return [
        TollamaForecastTool(base_url=base_url, timeout=timeout),
        TollamaAutoForecastTool(base_url=base_url, timeout=timeout),
        TollamaAnalyzeTool(base_url=base_url, timeout=timeout),
        TollamaWhatIfTool(base_url=base_url, timeout=timeout),
        TollamaCompareTool(base_url=base_url, timeout=timeout),
        TollamaRecommendTool(base_url=base_url, timeout=timeout),
        TollamaHealthTool(base_url=base_url, timeout=timeout),
        TollamaModelsTool(base_url=base_url, timeout=timeout),
    ]
