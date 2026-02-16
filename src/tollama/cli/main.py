"""Typer-based CLI for serving and querying tollama."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import uvicorn

from .client import DEFAULT_BASE_URL, DEFAULT_DAEMON_HOST, DEFAULT_DAEMON_PORT, TollamaClient

app = typer.Typer(help="Command-line interface for tollama.")


@app.command("serve")
def serve(
    host: str = typer.Option(DEFAULT_DAEMON_HOST, help="Host to bind for the daemon."),
    port: int = typer.Option(DEFAULT_DAEMON_PORT, help="Port to bind for the daemon."),
    log_level: str = typer.Option("info", help="Uvicorn log level."),
) -> None:
    """Run the tollama daemon HTTP server."""
    uvicorn.run("tollama.daemon.app:app", host=host, port=port, log_level=log_level)


@app.command("forecast")
def forecast(
    model: str = typer.Option("mock", help="Model name to set on the outgoing request."),
    input: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://127.0.0.1:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Send a forecast request from a JSON file to the running daemon."""
    payload = _load_request_payload(input)
    payload["model"] = model

    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.forecast(payload)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("models")
def models(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://127.0.0.1:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """List available and installed models reported by the daemon."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.list_models()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("pull")
def pull(
    name: str = typer.Argument(..., help="Model name from model-registry/registry.yaml."),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Accept the model license terms if required.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://127.0.0.1:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Install a model via the running daemon."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        manifest = client.pull_model(name=name, accept_license=accept_license)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))


@app.command("list")
def list_models(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://127.0.0.1:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """List models installed in the daemon-managed store."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        response = client.list_models()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = response.get("installed")
    if not isinstance(payload, list):
        typer.echo("Error: daemon returned unexpected list payload", err=True)
        raise typer.Exit(code=1)

    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("rm")
def rm(
    name: str = typer.Argument(..., help="Installed model name to remove."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://127.0.0.1:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Remove one installed model via the running daemon."""
    client = TollamaClient(base_url=base_url, timeout=timeout)
    try:
        result = client.remove_model(name)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

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
