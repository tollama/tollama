"""LangChain tool wrappers backed by the shared tollama HTTP client."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from tollama.client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    TollamaClient,
    TollamaClientError,
)
from tollama.core.schemas import ForecastRequest

_LANGCHAIN_IMPORT_HINT = (
    'LangChain dependency is not installed. Install with: pip install "tollama[langchain]"'
)
_MODEL_NAME_EXAMPLES = (
    "mock, chronos2, granite-ttm-r2, timesfm-2.5-200m, "
    "moirai-2.0-R-small, sundial-base-128m, toto-open-base-1.0"
)

try:
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(_LANGCHAIN_IMPORT_HINT) from exc


class _ToolInputBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HealthToolInput(_ToolInputBase):
    pass


class ModelsToolInput(_ToolInputBase):
    mode: Literal["installed", "loaded", "available"] = "installed"


class ForecastToolInput(_ToolInputBase):
    request: dict[str, Any]


def _make_client(*, base_url: str, timeout: float) -> TollamaClient:
    return TollamaClient(base_url=base_url, timeout=timeout)


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
        return self._run(run_manager=run_manager)


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
        return self._run(mode=mode, run_manager=run_manager)


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
        return self._run(request=request, run_manager=run_manager)


def get_tollama_tools(
    base_url: str = "http://127.0.0.1:11435",
    timeout: float = 10.0,
) -> list[BaseTool]:
    """Build the default tollama LangChain tool set."""
    return [
        TollamaForecastTool(base_url=base_url, timeout=timeout),
        TollamaHealthTool(base_url=base_url, timeout=timeout),
        TollamaModelsTool(base_url=base_url, timeout=timeout),
    ]
