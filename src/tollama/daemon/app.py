"""FastAPI application for the public tollama HTTP API."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import httpx
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

from tollama.core.config import ConfigFileError, TollamaConfig, load_config
from tollama.core.env_override import set_env_temporarily
from tollama.core.hf_pull import (
    OfflineModelUnavailableError,
    PullError,
    pull_snapshot_to_local_dir,
)
from tollama.core.pull_defaults import resolve_effective_pull_defaults
from tollama.core.redact import redact_config_dict, redact_env_dict, redact_proxy_url
from tollama.core.registry import get_model_spec, list_registry_models
from tollama.core.schemas import ForecastRequest, ForecastResponse
from tollama.core.storage import (
    TollamaPaths,
    install_from_registry,
    list_installed,
    read_manifest,
    remove_model,
    write_manifest,
)

from .loaded_models import LoadedModelTracker, parse_keep_alive, to_utc_iso
from .runner_manager import RunnerManager
from .supervisor import (
    RunnerCallError,
    RunnerError,
    RunnerProtocolError,
    RunnerUnavailableError,
)

DEFAULT_FORECAST_TIMEOUT_SECONDS = 10.0
DEFAULT_MODIFIED_AT = "1970-01-01T00:00:00Z"
INFO_ENV_KEYS = (
    "TOLLAMA_HOME",
    "TOLLAMA_HOST",
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

    model_config = ConfigDict(extra="allow", strict=True)

    model: StrictStr = Field(min_length=1)
    stream: StrictBool = True
    accept_license: StrictBool = False
    insecure: StrictBool | None = None
    offline: StrictBool | None = None
    local_files_only: StrictBool | None = None
    http_proxy: StrictStr | None = None
    https_proxy: StrictStr | None = None
    no_proxy: StrictStr | None = None
    hf_home: StrictStr | None = None
    max_workers: StrictInt | None = Field(default=None, gt=0)
    token: StrictStr | None = None


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

    keep_alive: StrictStr | StrictInt | StrictFloat | None = None


class ApiForecastRequest(ForecastRequestWithKeepAlive):
    """Ollama-compatible forecast request payload."""

    stream: StrictBool = True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.started_at = datetime.now(UTC)
    yield
    app.state.runner_manager.stop()
    app.state.loaded_model_tracker.clear()


def create_app(*, runner_manager: RunnerManager | None = None) -> FastAPI:
    """Create a configured FastAPI daemon app."""
    package_version = _resolve_package_version()
    app = FastAPI(title="tollama daemon", version=package_version, lifespan=_lifespan)
    app.state.runner_manager = runner_manager or RunnerManager()
    app.state.loaded_model_tracker = LoadedModelTracker()
    app.state.config_provider = ConfigProvider()
    app.state.started_at = datetime.now(UTC)
    app.state.host_binding = _resolve_host_binding_from_env()

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.errors()})

    @app.get("/api/version")
    def version() -> dict[str, str]:
        return {"version": package_version}

    @app.get("/api/info")
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
            },
            "runners": app.state.runner_manager.get_all_statuses(),
        }

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

    @app.post("/api/pull", response_model=None)
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
    model_manifest = _require_installed_manifest(payload.model)
    model_family = _manifest_family_or_500(model_manifest, payload.model)
    model_local_dir = _manifest_snapshot_path(model_manifest)

    try:
        keep_alive_policy = parse_keep_alive(payload.keep_alive, now=request_now)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    exclude_fields = {"keep_alive"}
    if extra_exclude is not None:
        exclude_fields.update(extra_exclude)
    params = payload.model_dump(mode="json", exclude_none=True, exclude=exclude_fields)
    params["model_name"] = payload.model
    params["model_family"] = model_family
    if model_local_dir is not None:
        params["model_local_dir"] = model_local_dir

    try:
        raw_result = app.state.runner_manager.call(
            family=model_family,
            method="forecast",
            params=params,
            timeout=DEFAULT_FORECAST_TIMEOUT_SECONDS,
        )
    except RunnerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RunnerCallError as exc:
        status_code = 503 if exc.code == "DEPENDENCY_MISSING" else 502
        raise HTTPException(status_code=status_code, detail=_runner_error_detail(exc)) from exc
    except RunnerProtocolError as exc:
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
        try:
            app.state.runner_manager.unload(
                family=model_family,
                model=payload.model,
                timeout=DEFAULT_FORECAST_TIMEOUT_SECONDS,
            )
        except RunnerError:
            app.state.runner_manager.stop(family=model_family)
        tracker.unload_runner(model_family)
        return response

    tracker.upsert(
        name=payload.model,
        model=payload.model,
        family=model_family,
        runner=model_family,
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
        },
    }
    return spec, manifest, paths


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
    return {
        "name": _optional_nonempty_str(manifest.get("name")),
        "family": _optional_nonempty_str(manifest.get("family")),
        "digest": _manifest_digest(manifest) or None,
        "size": _manifest_size_bytes(manifest),
        "modified_at": _optional_nonempty_str(
            manifest.get("pulled_at") or manifest.get("installed_at"),
        ),
    }


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


def _uptime_seconds(*, started_at: datetime | None, now: datetime) -> int:
    if started_at is None:
        return 0
    seconds = (now - started_at).total_seconds()
    return max(0, int(seconds))


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


app = create_app()
