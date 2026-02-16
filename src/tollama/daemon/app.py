"""FastAPI application for the public tollama HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
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

from tollama.core.registry import list_registry_models
from tollama.core.schemas import ForecastRequest, ForecastResponse
from tollama.core.storage import install_from_registry, list_installed, remove_model

from .loaded_models import LoadedModelTracker, parse_keep_alive, to_utc_iso
from .supervisor import (
    RunnerCallError,
    RunnerProtocolError,
    RunnerSupervisor,
    RunnerUnavailableError,
)

DEFAULT_FORECAST_TIMEOUT_SECONDS = 10.0
DEFAULT_MODIFIED_AT = "1970-01-01T00:00:00Z"
DEFAULT_RUNNER_ID = "default"


class ModelPullRequest(BaseModel):
    """Request body for pulling a model into local storage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: StrictStr = Field(min_length=1)
    accept_license: StrictBool


class ModelShowRequest(BaseModel):
    """Request body for the Ollama-compatible show endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1)


class ModelDeleteRequest(BaseModel):
    """Request body for the Ollama-compatible delete endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1)


class ApiPullRequest(BaseModel):
    """Request body for the Ollama-compatible pull endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: StrictStr = Field(min_length=1)
    stream: StrictBool = True
    accept_license: StrictBool = False


class ForecastRequestWithKeepAlive(ForecastRequest):
    """Forecast request payload with optional Ollama-compatible keep_alive semantics."""

    keep_alive: StrictStr | StrictInt | StrictFloat | None = None


class ApiForecastRequest(ForecastRequestWithKeepAlive):
    """Ollama-compatible forecast request payload."""

    stream: StrictBool = True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    app.state.supervisor.stop()
    app.state.loaded_model_tracker.clear()


def create_app(supervisor: RunnerSupervisor | None = None) -> FastAPI:
    """Create a configured FastAPI daemon app."""
    package_version = _resolve_package_version()
    app = FastAPI(title="tollama daemon", version=package_version, lifespan=_lifespan)
    app.state.supervisor = supervisor or RunnerSupervisor()
    app.state.loaded_model_tracker = LoadedModelTracker()

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/api/version")
    def version() -> dict[str, str]:
        return {"version": package_version}

    @app.get("/api/tags")
    def tags() -> dict[str, list[dict[str, Any]]]:
        manifests = list_installed()
        models = [_to_ollama_tag_model(manifest) for manifest in manifests]
        return {"models": models}

    @app.post("/api/show")
    def show(payload: ModelShowRequest) -> dict[str, Any]:
        manifest = _find_installed_manifest(payload.model)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"model {payload.model!r} is not installed")

        name = manifest.get("name")
        model_name = name if isinstance(name, str) else payload.model
        family = manifest.get("family")
        family_name = family if isinstance(family, str) else ""
        source = manifest.get("source")
        license_data = manifest.get("license")

        return {
            "name": model_name,
            "model": model_name,
            "family": family_name,
            "source": source if isinstance(source, dict) else {},
            "license": license_data if isinstance(license_data, dict) else {},
            "modelfile": "",
            "parameters": "",
        }

    @app.post("/api/pull", response_model=None)
    def pull(payload: ApiPullRequest) -> Any:
        manifest = _install_model(name=payload.model, accept_license=payload.accept_license)
        if not payload.stream:
            return manifest
        return StreamingResponse(
            _pull_stream_lines(name=payload.model, manifest=manifest),
            media_type="application/x-ndjson",
        )

    @app.delete("/api/delete")
    def delete(payload: ModelDeleteRequest) -> dict[str, Any]:
        _remove_model_or_404(payload.model)
        return {"deleted": True, "model": payload.model}

    @app.get("/api/ps")
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
        return _install_model(name=payload.name, accept_license=payload.accept_license)

    @app.delete("/v1/models/{name}")
    def delete_model(name: str) -> dict[str, Any]:
        _remove_model_or_404(name)
        return {"removed": True, "name": name}

    @app.post("/api/forecast", response_model=None)
    def api_forecast(payload: ApiForecastRequest) -> Any:
        response = _execute_forecast(app, payload=payload, extra_exclude={"stream"})
        if not payload.stream:
            return response
        return StreamingResponse(
            _forecast_stream_lines(response),
            media_type="application/x-ndjson",
        )

    @app.post("/v1/forecast", response_model=ForecastResponse)
    def forecast(payload: ForecastRequestWithKeepAlive) -> ForecastResponse:
        return _execute_forecast(app, payload=payload)

    return app


def _execute_forecast(
    app: FastAPI,
    *,
    payload: ForecastRequestWithKeepAlive,
    extra_exclude: set[str] | None = None,
) -> ForecastResponse:
    _unload_expired_models(app)
    request_now = datetime.now(UTC)

    try:
        keep_alive_policy = parse_keep_alive(payload.keep_alive, now=request_now)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    exclude_fields = {"keep_alive"}
    if extra_exclude is not None:
        exclude_fields.update(extra_exclude)
    params = payload.model_dump(mode="json", exclude_none=True, exclude=exclude_fields)

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
        response = ForecastResponse.model_validate(raw_result)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"runner returned invalid forecast response: {exc}",
        ) from exc

    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    if keep_alive_policy.unload_immediately:
        app.state.supervisor.stop()
        tracker.unload_runner(DEFAULT_RUNNER_ID)
        return response

    tracker.upsert(
        name=payload.model,
        model=payload.model,
        family=_resolve_loaded_model_family(payload.model),
        runner=DEFAULT_RUNNER_ID,
        expires_at=keep_alive_policy.expires_at,
        device=None,
    )
    return response


def _forecast_stream_lines(response: ForecastResponse) -> Iterator[str]:
    yield _to_ndjson_line({"status": "loading model"})
    yield _to_ndjson_line({"status": "running forecast"})
    yield _to_ndjson_line(
        {
            "done": True,
            "response": response.model_dump(mode="json", exclude_none=True),
        },
    )


def _to_ndjson_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def _pull_stream_lines(name: str, manifest: dict[str, Any]) -> Iterator[str]:
    yield _to_ndjson_line({"status": "pulling model manifest", "model": name})
    yield _to_ndjson_line({"status": "writing model files", "model": name})
    yield _to_ndjson_line({"done": True, "model": name, "manifest": manifest})


def _runner_error_detail(exc: RunnerCallError | RunnerProtocolError) -> dict[str, Any] | str:
    if isinstance(exc, RunnerCallError):
        detail: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.data is not None:
            detail["data"] = exc.data
        return detail
    return str(exc)


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


def _resolve_loaded_model_family(name: str) -> str:
    manifest = _find_installed_manifest(name)
    if manifest is None:
        return name

    family = manifest.get("family")
    if isinstance(family, str) and family:
        return family
    return name


def _to_ollama_tag_model(manifest: dict[str, Any]) -> dict[str, Any]:
    name = manifest.get("name")
    model_name = name if isinstance(name, str) else ""

    family = manifest.get("family")
    family_name = family if isinstance(family, str) else ""
    families = [family_name] if family_name else []

    digest = manifest.get("digest")
    digest_value = digest if isinstance(digest, str) else ""

    size = manifest.get("size")
    size_value = size if isinstance(size, int) and size >= 0 else 0

    return {
        "name": model_name,
        "model": model_name,
        "modified_at": _normalize_modified_at(manifest.get("installed_at")),
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


def _unload_expired_models(app: FastAPI) -> None:
    tracker: LoadedModelTracker = app.state.loaded_model_tracker
    now = datetime.now(UTC)
    expired_runners = tracker.expired_runners(now)
    for runner in expired_runners:
        app.state.supervisor.stop()
        tracker.unload_runner(runner)


def _resolve_package_version() -> str:
    try:
        return metadata.version("tollama")
    except metadata.PackageNotFoundError:
        return "0.0.0"


app = create_app()
