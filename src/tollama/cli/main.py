"""Typer-based CLI for serving and querying tollama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import uvicorn

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DaemonHTTPError,
    TollamaClient,
)

app = typer.Typer(help="Ollama-style command-line interface for tollama.")


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
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11434.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Pull a model from the registry into local storage."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    stream = not no_stream
    try:
        result = client.pull_model(name=model, stream=stream)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit_result(result)


@app.command("list")
def list_models(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11434.",
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
        help="Daemon base URL. Defaults to http://localhost:11434.",
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
        help="Daemon base URL. Defaults to http://localhost:11434.",
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
        help="Daemon base URL. Defaults to http://localhost:11434.",
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
        help="Daemon base URL. Defaults to http://localhost:11434.",
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
