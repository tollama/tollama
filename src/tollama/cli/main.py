"""Typer-based CLI for serving and querying tollama."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
import uvicorn

from tollama.core.config import ConfigFileError, load_config, update_config
from tollama.core.registry import get_model_spec
from tollama.core.storage import TollamaPaths

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DaemonHTTPError,
    TollamaClient,
)
from .info import collect_info

app = typer.Typer(help="Ollama-style command-line interface for tollama.")
config_app = typer.Typer(help="Manage local tollama defaults in ~/.tollama/config.json.")
app.add_typer(config_app, name="config")

_CONFIG_KEY_PATHS: dict[str, tuple[str, str]] = {
    "pull.offline": ("pull", "offline"),
    "pull.hf_home": ("pull", "hf_home"),
    "pull.http_proxy": ("pull", "http_proxy"),
    "pull.https_proxy": ("pull", "https_proxy"),
    "pull.no_proxy": ("pull", "no_proxy"),
    "pull.local_files_only": ("pull", "local_files_only"),
    "pull.insecure": ("pull", "insecure"),
    "pull.max_workers": ("pull", "max_workers"),
}
_BOOL_CONFIG_KEYS = {
    "pull.offline",
    "pull.local_files_only",
    "pull.insecure",
}
_INT_CONFIG_KEYS = {"pull.max_workers"}


@app.command("serve")
def serve(
    host: str = typer.Option(DEFAULT_DAEMON_HOST, help="Host to bind for the daemon."),
    port: int = typer.Option(DEFAULT_DAEMON_PORT, help="Port to bind for the daemon."),
    log_level: str = typer.Option("info", help="Uvicorn log level."),
) -> None:
    """Run the tollama daemon HTTP server."""
    previous_binding = os.environ.get("TOLLAMA_EFFECTIVE_HOST_BINDING")
    os.environ["TOLLAMA_EFFECTIVE_HOST_BINDING"] = f"{host}:{port}"
    try:
        uvicorn.run("tollama.daemon.app:app", host=host, port=port, log_level=log_level)
    finally:
        if previous_binding is None:
            os.environ.pop("TOLLAMA_EFFECTIVE_HOST_BINDING", None)
        else:
            os.environ["TOLLAMA_EFFECTIVE_HOST_BINDING"] = previous_binding


@app.command("pull")
def pull(
    model: str = typer.Argument(..., help="Model name to pull."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming pull output."),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Accept model license terms when required.",
    ),
    insecure: bool | None = typer.Option(
        None,
        "--insecure/--no-insecure",
        help="Override SSL verification for pull.",
    ),
    offline: bool | None = typer.Option(
        None,
        "--offline/--no-offline",
        help="Override offline pull mode.",
    ),
    local_files_only: bool | None = typer.Option(
        None,
        "--local-files-only/--no-local-files-only",
        help="Use local cache only for pull, without forcing full offline mode.",
    ),
    http_proxy: str | None = typer.Option(None, "--http-proxy", help="HTTP proxy URL."),
    https_proxy: str | None = typer.Option(None, "--https-proxy", help="HTTPS proxy URL."),
    no_proxy: str | None = typer.Option(None, "--no-proxy", help="No proxy host patterns."),
    hf_home: str | None = typer.Option(None, "--hf-home", help="HF_HOME override."),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Hugging Face token override (prefer TOLLAMA_HF_TOKEN).",
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Bypass daemon pull config defaults for this request.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Pull a model from the registry into local storage."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    stream = not no_stream
    resolved_token = token if token is not None else os.environ.get("TOLLAMA_HF_TOKEN")
    include_null_fields: set[str] = set()

    pull_insecure = insecure
    pull_offline = offline
    pull_local_files_only = local_files_only
    pull_http_proxy = http_proxy
    pull_https_proxy = https_proxy
    pull_no_proxy = no_proxy
    pull_hf_home = hf_home
    if no_config:
        if pull_insecure is None:
            pull_insecure = False
        if pull_offline is None:
            pull_offline = False
        if pull_local_files_only is None:
            pull_local_files_only = False

        include_null_fields.update({"http_proxy", "https_proxy", "no_proxy", "hf_home"})

    try:
        result = client.pull_model(
            name=model,
            stream=stream,
            accept_license=accept_license,
            insecure=pull_insecure,
            offline=pull_offline,
            local_files_only=pull_local_files_only,
            http_proxy=pull_http_proxy,
            https_proxy=pull_https_proxy,
            no_proxy=pull_no_proxy,
            hf_home=pull_hf_home,
            token=resolved_token,
            include_null_fields=include_null_fields,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_result(result)


@config_app.command("list")
def config_list(
    json_output: bool = typer.Option(False, "--json", help="Print compact JSON."),
) -> None:
    """Print current local config values."""
    try:
        config = load_config(TollamaPaths.default())
    except ConfigFileError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = config.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Config key path (example: pull.offline)."),
) -> None:
    """Get one config value."""
    key_path = _resolve_config_key_path(key)
    try:
        config = load_config(TollamaPaths.default())
    except ConfigFileError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = config.model_dump(mode="json")
    value = payload[key_path[0]][key_path[1]]
    typer.echo(_format_config_scalar(value))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key path (example: pull.offline)."),
    value: str = typer.Argument(..., help="New value."),
) -> None:
    """Set one config value."""
    key_path = _resolve_config_key_path(key)
    parsed_value = _parse_config_value(key, value)
    updates = {key_path[0]: {key_path[1]: parsed_value}}
    try:
        update_config(TollamaPaths.default(), updates)
    except ConfigFileError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(..., help="Config key path (example: pull.offline)."),
) -> None:
    """Unset one config value by writing null."""
    key_path = _resolve_config_key_path(key)
    updates = {key_path[0]: {key_path[1]: None}}
    try:
        update_config(TollamaPaths.default(), updates)
    except ConfigFileError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("info")
def info(
    json_output: bool = typer.Option(False, "--json", help="Print JSON diagnostics output."),
    timeout: float = typer.Option(2.0, "--timeout", min=0.1, help="Daemon call timeout."),
    verbose: bool = typer.Option(False, "--verbose", help="Include additional model details."),
    local: bool = typer.Option(
        False,
        "--local",
        help="Force local diagnostics collection without calling the daemon.",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Require daemon diagnostics via GET /api/info.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
) -> None:
    """Show local and daemon diagnostics in one view."""
    if local and remote:
        raise typer.BadParameter("--local and --remote cannot be used together")

    mode = "auto"
    if local:
        mode = "local"
    if remote:
        mode = "remote"

    try:
        snapshot = collect_info(
            base_url=base_url,
            paths=TollamaPaths.default(),
            timeout_s=timeout,
            mode=mode,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if json_output:
        typer.echo(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    typer.echo(_render_info_report(snapshot, verbose=verbose))


@app.command("list")
def list_models(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """List installed models via GET /api/tags."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.list_tags()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("ps")
def ps(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Show loaded models via GET /api/ps."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.list_running()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("show")
def show(
    model: str = typer.Argument(..., help="Model name to inspect."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Show model metadata via POST /api/show."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.show_model(model)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("rm")
def rm(
    model: str = typer.Argument(..., help="Installed model name to remove."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Delete a model via DELETE /api/delete."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.remove_model(model)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("run")
def run(
    model: str = typer.Argument(..., help="Model name to run."),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Forecast request JSON file. If omitted, reads stdin or example payload.",
    ),
    horizon: int | None = typer.Option(None, "--horizon", min=1, help="Override request horizon."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming forecast output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Run a forecast through POST /api/forecast, auto-pulling if needed."""
    payload = _load_request_payload(path=input_path, model=model)
    payload["model"] = model
    if horizon is not None:
        payload["horizon"] = horizon

    stream = not no_stream
    client = TollamaClient(base_url=base_url, timeout=timeout)

    try:
        client.show_model(model)
    except DaemonHTTPError as exc:
        if exc.status_code != 404:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        try:
            pull_result = client.pull_model(name=model, stream=stream)
        except RuntimeError as pull_exc:
            typer.echo(f"Error: {pull_exc}", err=True)
            raise typer.Exit(code=1) from pull_exc
        _emit_result(pull_result)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:
        forecast_result = client.forecast(payload, stream=stream)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_result(forecast_result)


def _emit_result(result: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(result, list):
        for item in result:
            _emit_warnings(item)
            typer.echo(json.dumps(item, sort_keys=True))
        return
    _emit_warnings(result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def _emit_warnings(payload: dict[str, Any]) -> None:
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if isinstance(warning, str) and warning:
                typer.echo(f"warning: {warning}", err=True)

    response = payload.get("response")
    if isinstance(response, dict):
        nested = response.get("warnings")
        if isinstance(nested, list):
            for warning in nested:
                if isinstance(warning, str) and warning:
                    typer.echo(f"warning: {warning}", err=True)


def _render_info_report(snapshot: dict[str, Any], *, verbose: bool) -> str:
    client = snapshot.get("client") if isinstance(snapshot, dict) else {}
    paths = snapshot.get("paths") if isinstance(snapshot, dict) else {}
    daemon = snapshot.get("daemon") if isinstance(snapshot, dict) else {}
    models = snapshot.get("models") if isinstance(snapshot, dict) else {}
    config = snapshot.get("config") if isinstance(snapshot, dict) else None
    env_payload = snapshot.get("env") if isinstance(snapshot, dict) else None
    runners = snapshot.get("runners") if isinstance(snapshot, dict) else None
    pull_defaults = snapshot.get("pull_defaults") if isinstance(snapshot, dict) else {}

    base_url = _value_or_unknown(_safe_dict_get(client, "base_url"))
    home_path = _value_or_unknown(_safe_dict_get(paths, "tollama_home"))
    config_path = _value_or_unknown(_safe_dict_get(paths, "config_path"))
    config_exists = bool(_safe_dict_get(paths, "config_exists"))
    source = _value_or_unknown(_safe_dict_get(client, "source"))

    daemon_reachable = bool(_safe_dict_get(daemon, "reachable"))
    daemon_version = _safe_dict_get(daemon, "version")
    daemon_error = _safe_dict_get(daemon, "error")
    daemon_started_at = _safe_dict_get(daemon, "started_at")
    daemon_uptime_seconds = _safe_dict_get(daemon, "uptime_seconds")
    daemon_host_binding = _safe_dict_get(daemon, "host_binding")

    lines: list[str] = ["Tollama"]
    lines.append(f"  Home: {home_path}")
    lines.append(f"  Config: {config_path} ({'exists' if config_exists else 'missing'})")
    lines.append(f"  Info source: {source}")
    if daemon_reachable:
        version_suffix = f" version={daemon_version}" if daemon_version else ""
        lines.append(f"  Daemon: {base_url} (reachable){version_suffix}")
        lines.append(f"    started_at: {_format_info_value(daemon_started_at)}")
        lines.append(f"    uptime_seconds: {_format_info_value(daemon_uptime_seconds)}")
        lines.append(f"    host_binding: {_format_info_value(daemon_host_binding)}")
    else:
        lines.append(f"  Daemon: {base_url} (unreachable)")
        if daemon_error:
            lines.append(f"    error: {daemon_error}")

    if isinstance(config, dict) and "error" in config:
        lines.append("")
        lines.append("Config")
        lines.append(f"  error: {_format_info_value(config.get('error'))}")
    elif isinstance(config, dict):
        lines.append("")
        lines.append("Config")
        lines.extend(_indent_block(json.dumps(config, indent=2, sort_keys=True), spaces=2))

    lines.append("")
    lines.append("Pull defaults (effective)")
    for key in (
        "offline",
        "local_files_only",
        "insecure",
        "hf_home",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "max_workers",
    ):
        detail = pull_defaults.get(key) if isinstance(pull_defaults, dict) else None
        if isinstance(detail, dict):
            value = _format_info_value(detail.get("value"))
            source = _value_or_unknown(detail.get("source"))
        else:
            value = "null"
            source = "unknown"
        lines.append(f"  {key}: {value} (source={source})")

    installed = models.get("installed") if isinstance(models, dict) else None
    installed_entries = installed if isinstance(installed, list) else []
    lines.append("")
    lines.append(f"Models installed: {len(installed_entries)}")
    for item in installed_entries:
        if not isinstance(item, dict):
            continue
        name = _value_or_unknown(item.get("name"))
        family = _value_or_unknown(item.get("family"))
        digest = _value_or_unknown(item.get("digest"))
        size = _format_info_value(item.get("size"))
        modified_at = _format_info_value(item.get("modified_at"))
        covariates = _covariates_summary(item.get("capabilities"))
        lines.append(
            f"  - {name} (family={family}) digest={digest} size={size} modified_at={modified_at}",
        )
        if covariates is not None:
            lines.append(f"    covariates: {covariates}")
        if verbose:
            lines.append(f"    raw: {json.dumps(item, sort_keys=True)}")

    available = models.get("available") if isinstance(models, dict) else None
    available_entries = available if isinstance(available, list) else []
    lines.append("")
    lines.append(f"Models available: {len(available_entries)}")
    for item in available_entries:
        if not isinstance(item, dict):
            continue
        name = _value_or_unknown(item.get("name"))
        family = _value_or_unknown(item.get("family"))
        covariates = _covariates_summary(item.get("capabilities"))
        if covariates is None:
            lines.append(f"  - {name} (family={family})")
        else:
            lines.append(f"  - {name} (family={family}) covariates: {covariates}")
        if verbose:
            lines.append(f"    raw: {json.dumps(item, sort_keys=True)}")

    loaded = models.get("loaded") if isinstance(models, dict) else None
    loaded_entries = loaded if isinstance(loaded, list) else []
    lines.append("")
    lines.append(f"Models loaded: {len(loaded_entries)}")
    for item in loaded_entries:
        if not isinstance(item, dict):
            continue
        name = _value_or_unknown(item.get("name") or item.get("model"))
        expires_at = _format_info_value(item.get("expires_at"))
        family = _value_or_unknown(item.get("family"))
        lines.append(f"  - {name} (family={family}) expires_at={expires_at}")
        if verbose:
            lines.append(f"    raw: {json.dumps(item, sort_keys=True)}")

    runner_entries = runners if isinstance(runners, list) else []
    lines.append("")
    lines.append(f"Runners: {len(runner_entries)}")
    for item in runner_entries:
        if not isinstance(item, dict):
            continue
        family = _value_or_unknown(item.get("family"))
        installed_value = _format_info_value(item.get("installed"))
        running_value = _format_info_value(item.get("running"))
        pid_value = _format_info_value(item.get("pid"))
        restarts_value = _format_info_value(item.get("restarts"))
        lines.append(
            f"  - {family}: installed={installed_value} running={running_value} "
            f"pid={pid_value} restarts={restarts_value}",
        )
        last_error = item.get("last_error")
        if isinstance(last_error, dict):
            lines.append(
                "    last_error: "
                f"{_format_info_value(last_error.get('message'))} "
                f"at={_format_info_value(last_error.get('at'))}",
            )
        if verbose:
            lines.append(f"    raw: {json.dumps(item, sort_keys=True)}")

    if isinstance(env_payload, dict):
        lines.append("")
        lines.append("Environment")
        for key in (
            "TOLLAMA_HOST",
            "TOLLAMA_HOME",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "TOLLAMA_HF_TOKEN_present",
            "HF_TOKEN_present",
            "HUGGINGFACE_HUB_TOKEN_present",
            "HUGGING_FACE_HUB_TOKEN_present",
            "HF_HUB_TOKEN_present",
        ):
            lines.append(f"  {key}: {_format_info_value(env_payload.get(key))}")

    return "\n".join(lines)


def _resolve_config_key_path(key: str) -> tuple[str, str]:
    key_path = _CONFIG_KEY_PATHS.get(key)
    if key_path is None:
        supported = ", ".join(sorted(_CONFIG_KEY_PATHS))
        raise typer.BadParameter(f"unsupported key {key!r}. Supported keys: {supported}")
    return key_path


def _parse_config_value(key: str, value: str) -> bool | int | str | None:
    lowered = value.strip().lower()
    if lowered == "null":
        return None

    if key in _BOOL_CONFIG_KEYS:
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        raise typer.BadParameter(f"invalid boolean value for {key!r}: {value!r}")

    if key in _INT_CONFIG_KEYS:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise typer.BadParameter(f"invalid integer value for {key!r}: {value!r}") from exc
        if parsed <= 0:
            raise typer.BadParameter(f"value for {key!r} must be > 0")
        return parsed

    return value


def _format_config_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_info_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _indent_block(text: str, *, spaces: int) -> list[str]:
    prefix = " " * spaces
    return [f"{prefix}{line}" for line in text.splitlines()]


def _safe_dict_get(payload: Any, key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _value_or_unknown(value: Any) -> str:
    normalized = _format_info_value(value)
    if normalized == "null":
        return "unknown"
    return normalized


def _covariates_summary(capabilities: Any) -> str | None:
    if not isinstance(capabilities, dict):
        return None

    parts: list[str] = []
    if capabilities.get("past_covariates_numeric"):
        parts.append("past-only numeric")
    if capabilities.get("past_covariates_categorical"):
        parts.append("past-only categorical")
    if capabilities.get("future_covariates_numeric"):
        parts.append("known-future numeric")
    if capabilities.get("future_covariates_categorical"):
        parts.append("known-future categorical")
    if capabilities.get("static_covariates"):
        parts.append("static")
    else:
        parts.append("static planned")
    return ", ".join(parts)


def _load_request_payload(path: Path | None, *, model: str) -> dict[str, Any]:
    if path is not None:
        return _load_request_payload_from_path(path)

    stdin_payload = _load_request_payload_from_stdin()
    if stdin_payload is not None:
        return stdin_payload

    default_path = _resolve_default_request_path(model)
    if default_path is not None:
        return _load_request_payload_from_path(default_path)

    raise typer.BadParameter(
        "missing forecast request payload. Provide --input PATH, pipe JSON via stdin, "
        f"or create examples/{model}_request.json",
    )


def _load_request_payload_from_stdin() -> dict[str, Any] | None:
    if sys.stdin.isatty():
        return None

    raw = sys.stdin.read()
    if not raw.strip():
        return None
    return _parse_request_payload(raw, source="stdin")


def _resolve_default_request_path(model: str) -> Path | None:
    candidate_names = [f"{alias}_request.json" for alias in _request_payload_aliases(model)]
    candidate_names.append("request.json")

    roots: list[Path] = [Path.cwd()]
    package_root = _project_root_from_module()
    if package_root is not None and package_root not in roots:
        roots.append(package_root)

    for root in roots:
        example_dir = root / "examples"
        for name in candidate_names:
            candidate = example_dir / name
            if candidate.is_file():
                return candidate

    return None


def _request_payload_aliases(model: str) -> tuple[str, ...]:
    aliases: list[str] = []
    _append_alias(aliases, model)
    _append_alias(aliases, _normalize_request_alias(model))

    implementation = _model_implementation(model)
    if implementation is not None:
        _append_alias(aliases, implementation)
        _append_alias(aliases, _normalize_request_alias(implementation))

        for suffix in ("_torch", "_runner", "_adapter"):
            if implementation.endswith(suffix) and len(implementation) > len(suffix):
                trimmed = implementation[: -len(suffix)]
                _append_alias(aliases, trimmed)
                _append_alias(aliases, _normalize_request_alias(trimmed))

        _append_alias(aliases, implementation.split("_", maxsplit=1)[0])

    return tuple(aliases)


def _append_alias(aliases: list[str], alias: str | None) -> None:
    if alias is None:
        return
    normalized = alias.strip()
    if not normalized:
        return
    if normalized not in aliases:
        aliases.append(normalized)


def _normalize_request_alias(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(".", "_")


def _model_implementation(model: str) -> str | None:
    try:
        spec = get_model_spec(model)
    except (KeyError, OSError, ValueError):
        return None

    metadata = spec.metadata
    if not isinstance(metadata, dict):
        return None

    implementation = metadata.get("implementation")
    if not isinstance(implementation, str):
        return None
    return implementation.strip() or None


def _project_root_from_module() -> Path | None:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) < 4:
        return None
    return resolved.parents[3]


def _load_request_payload_from_path(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(f"unable to read input file: {exc}") from exc

    return _parse_request_payload(raw, source="input file")


def _parse_request_payload(raw: str, *, source: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise typer.BadParameter(f"{source} JSON must be an object")

    return payload


def main() -> None:
    """Console script entrypoint for the Typer app."""
    app()


if __name__ == "__main__":
    main()
