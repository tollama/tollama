"""FastAPI application for the public tollama HTTP API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from tollama.core.registry import list_registry_models
from tollama.core.schemas import ForecastRequest, ForecastResponse
from tollama.core.storage import list_installed

from .supervisor import (
    RunnerCallError,
    RunnerProtocolError,
    RunnerSupervisor,
    RunnerUnavailableError,
)

DEFAULT_FORECAST_TIMEOUT_SECONDS = 10.0


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    app.state.supervisor.stop()


def create_app(supervisor: RunnerSupervisor | None = None) -> FastAPI:
    """Create a configured FastAPI daemon app."""
    app = FastAPI(title="tollama daemon", version="0.1.0", lifespan=_lifespan)
    app.state.supervisor = supervisor or RunnerSupervisor()

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, list[dict[str, Any]]]:
        available = [
            spec.model_dump(mode="json", exclude_none=True)
            for spec in list_registry_models()
        ]
        installed = list_installed()
        return {"available": available, "installed": installed}

    @app.post("/v1/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequest) -> ForecastResponse:
        params = payload.model_dump(mode="json", exclude_none=True)

        try:
            raw_result = app.state.supervisor.call(
                method="forecast",
                params=params,
                timeout=DEFAULT_FORECAST_TIMEOUT_SECONDS,
            )
        except RunnerUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (RunnerCallError, RunnerProtocolError) as exc:
            raise HTTPException(status_code=502, detail=_runner_error_detail(exc)) from exc

        try:
            return ForecastResponse.model_validate(raw_result)
        except ValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"runner returned invalid forecast response: {exc}",
            ) from exc

    return app


def _runner_error_detail(exc: RunnerCallError | RunnerProtocolError) -> dict[str, Any] | str:
    if isinstance(exc, RunnerCallError):
        detail: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.data is not None:
            detail["data"] = exc.data
        return detail
    return str(exc)


app = create_app()
