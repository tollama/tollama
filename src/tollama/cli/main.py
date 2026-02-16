"""Typer-based CLI for serving and querying tollama."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import typer
import uvicorn

from tollama.core.config import ConfigFileError, load_config, update_config
from tollama.core.storage import TollamaPaths

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DaemonHTTPError,
    TollamaClient,
)

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
    uvicorn.run("tollama.daemon.app:app", host=host, port=port, log_level=log_level)


@app.command("pull")
def pull(
    model: str = typer.Argument(..., help="Model name to pull."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming pull output."),
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
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    horizon: int | None = typer.Option(None, "--horizon", min=1, help="Override request horizon."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming forecast output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Run a forecast through POST /api/forecast, auto-pulling if needed."""
    payload = _load_request_payload(input)
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
            typer.echo(json.dumps(item, sort_keys=True))
        return
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


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


def _load_request_payload(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(f"unable to read input file: {exc}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"input file is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise typer.BadParameter("input JSON must be an object")

    return payload


def main() -> None:
    """Console script entrypoint for the Typer app."""
    app()


if __name__ == "__main__":
    main()
