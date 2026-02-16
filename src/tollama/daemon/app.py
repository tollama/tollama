"""FastAPI application for the public tollama HTTP API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, ValidationError

from tollama.core.registry import list_registry_models
from tollama.core.schemas import ForecastRequest, ForecastResponse
from tollama.core.storage import install_from_registry, list_installed, remove_model

from .supervisor import (
    RunnerCallError,
    RunnerProtocolError,
    RunnerSupervisor,
    RunnerUnavailableError,
)

DEFAULT_FORECAST_TIMEOUT_SECONDS = 10.0


class ModelPullRequest(BaseModel):
    """Request body for pulling a model into local storage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: StrictStr = Field(min_length=1)
    accept_license: StrictBool


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
            installed.append(
                {
                    "name": manifest.get("name"),
                    "family": manifest.get("family"),
                    "installed": True,
                    "license": manifest.get("license"),
                },
            )
        return {"available": available, "installed": installed}

    @app.post("/v1/models/pull")
    def pull_model(payload: ModelPullRequest) -> dict[str, Any]:
        try:
            return install_from_registry(payload.name, accept_license=payload.accept_license)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/v1/models/{name}")
    def delete_model(name: str) -> dict[str, Any]:
        try:
            removed = remove_model(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not removed:
            raise HTTPException(status_code=404, detail=f"model {name!r} is not installed")
        return {"removed": True, "name": name}

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
