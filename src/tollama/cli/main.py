"""Typer-based CLI for serving and querying tollama."""

from __future__ import annotations

import json
import os
import shutil
import sys
import webbrowser
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import httpx
import typer
import uvicorn
from tqdm import tqdm

from tollama.core.config import (
    CONFIG_KEY_DESCRIPTIONS,
    ConfigFileError,
    TollamaConfig,
    load_config,
    save_config,
    update_config,
)
from tollama.core.registry import ModelCapabilities, ModelSpec, get_model_spec, list_registry_models
from tollama.core.runtime_bootstrap import (
    FAMILY_EXTRAS,
    FAMILY_PYTHON_CONSTRAINTS,
    BootstrapError,
    ensure_family_runtime,
    list_runtime_statuses,
    remove_family_runtime,
)
from tollama.core.storage import TollamaPaths, list_installed

from .client import (
    DEFAULT_BASE_URL,
    DEFAULT_DAEMON_HOST,
    DEFAULT_DAEMON_PORT,
    DaemonHTTPError,
    TollamaClient,
)
from .dev import dev_app
from .info import collect_info

app = typer.Typer(help="Ollama-style command-line interface for tollama.")
config_app = typer.Typer(help="Manage local tollama defaults in ~/.tollama/config.json.")
runtime_app = typer.Typer(help="Manage per-family isolated runner environments.")
modelfile_app = typer.Typer(help="Manage TSModelfile forecast profiles.")
app.add_typer(config_app, name="config")
app.add_typer(runtime_app, name="runtime")
app.add_typer(modelfile_app, name="modelfile")
app.add_typer(dev_app, name="dev")

_CONFIG_KEY_PATHS: dict[str, tuple[str, str]] = {
    key: cast(tuple[str, str], tuple(key.split(".", maxsplit=1)))
    for key in CONFIG_KEY_DESCRIPTIONS
}
_BOOL_CONFIG_KEYS = {
    "pull.offline",
    "pull.local_files_only",
    "pull.insecure",
}
_INT_CONFIG_KEYS = {"pull.max_workers"}

_UNI2TS_PYTHON_WARNING = "Uni2TS/Moirai dependencies may fail to install on Python 3.12+"
_RUN_TIMEOUT_SECONDS = 300.0
_QUICKSTART_MODEL = "mock"
_QUICKSTART_HORIZON = 3
_DOCTOR_TOKEN_ENV_KEYS = (
    "TOLLAMA_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)
_API_KEY_ENV_NAME = "TOLLAMA_API_KEY"
_TABLE_MAX_COL_WIDTH = 48
_TABLE_DEFAULT_GAP = 2

ProgressMode = Literal["auto", "on", "off"]

_COLOR_SUCCESS = typer.colors.GREEN
_COLOR_WARNING = typer.colors.YELLOW
_COLOR_ERROR = typer.colors.RED
_COLOR_DIM = typer.colors.BRIGHT_BLACK


def _exit_with_message(message: str, *, code: int = 2) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(code=code)


def _style_text(
    text: str,
    *,
    fg: int | None = None,
    bold: bool = False,
    dim: bool = False,
    err: bool = False,
) -> str:
    del err
    return typer.style(text, fg=fg, bold=bold, dim=dim)


def _resolve_progress_enabled(mode: ProgressMode) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _progress_note(enabled: bool, message: str) -> None:
    if not enabled:
        return
    typer.echo(_style_text(message, fg=_COLOR_DIM, err=True), err=True)


def _error_hint(exc: BaseException) -> str | None:
    value = getattr(exc, "hint", None)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _exit_with_runtime_error(exc: RuntimeError, *, code: int = 1) -> NoReturn:
    typer.echo(_style_text(f"Error: {exc}", fg=_COLOR_ERROR, err=True), err=True)
    hint = _error_hint(exc)
    if hint is not None:
        typer.echo(_style_text(f"Hint: {hint}", fg=_COLOR_WARNING, err=True), err=True)
    raise typer.Exit(code=code) from exc


def _extract_completion_incomplete(*args: Any) -> str:
    if not args:
        return ""
    candidate = args[-1]
    if isinstance(candidate, str):
        return candidate
    return ""


def _complete_model_names(*args: Any) -> list[str]:
    incomplete = _extract_completion_incomplete(*args).lower()
    names: set[str] = set()

    try:
        for spec in list_registry_models():
            names.add(spec.name)
    except Exception:  # noqa: BLE001
        pass

    try:
        for manifest in list_installed():
            name = manifest.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    except Exception:  # noqa: BLE001
        pass

    if not incomplete:
        return sorted(names)
    return sorted(name for name in names if name.lower().startswith(incomplete))


def _complete_config_keys(*args: Any) -> list[str]:
    incomplete = _extract_completion_incomplete(*args)
    if not incomplete:
        return sorted(_CONFIG_KEY_PATHS)
    lowered = incomplete.lower()
    return sorted(key for key in _CONFIG_KEY_PATHS if key.lower().startswith(lowered))


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


@app.command("open")
def open_dashboard(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
) -> None:
    """Open the bundled web dashboard in the default browser."""
    dashboard_url = _dashboard_url(base_url)
    opened = webbrowser.open(dashboard_url)
    if not opened:
        _exit_with_message(
            f"Unable to open browser automatically. Open this URL manually: {dashboard_url}",
            code=1,
        )
    typer.echo(dashboard_url)


@app.command("dashboard")
def dashboard(
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Launch the Textual terminal dashboard."""
    try:
        from tollama.tui.app import run_dashboard_app
    except Exception as exc:  # noqa: BLE001
        _exit_with_runtime_error(RuntimeError(str(exc)))

    api_key = os.environ.get(_API_KEY_ENV_NAME)
    try:
        run_dashboard_app(base_url=base_url, timeout=timeout, api_key=api_key)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)


@app.command("pull")
def pull(
    model: str = typer.Argument(
        ...,
        help="Model name to pull.",
        autocompletion=_complete_model_names,
    ),
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
    progress: ProgressMode = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto, on, or off.",
    ),
) -> None:
    """Pull a model from the registry into local storage."""
    client = _make_client(base_url=base_url, timeout=timeout)
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
        _exit_with_runtime_error(exc)
    _emit_result(
        result,
        stream_kind="pull",
        show_progress=_resolve_progress_enabled(progress),
    )


@config_app.command("list")
def config_list(
    json_output: bool = typer.Option(False, "--json", help="Print compact JSON."),
) -> None:
    """Print current local config values."""
    try:
        config = load_config(TollamaPaths.default())
    except ConfigFileError as exc:
        _exit_with_runtime_error(exc)

    payload = config.model_dump(mode="json")
    if json_output:
        typer.echo(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        return
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@config_app.command("get")
def config_get(
    key: str = typer.Argument(
        ...,
        help="Config key path (example: pull.offline).",
        autocompletion=_complete_config_keys,
    ),
) -> None:
    """Get one config value."""
    key_path = _resolve_config_key_path(key)
    try:
        config = load_config(TollamaPaths.default())
    except ConfigFileError as exc:
        _exit_with_runtime_error(exc)

    payload = config.model_dump(mode="json")
    value = payload[key_path[0]][key_path[1]]
    typer.echo(_format_config_scalar(value))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="Config key path (example: pull.offline).",
        autocompletion=_complete_config_keys,
    ),
    value: str = typer.Argument(..., help="New value."),
) -> None:
    """Set one config value."""
    key_path = _resolve_config_key_path(key)
    parsed_value = _parse_config_value(key, value)
    updates = {key_path[0]: {key_path[1]: parsed_value}}
    try:
        update_config(TollamaPaths.default(), updates)
    except ConfigFileError as exc:
        _exit_with_runtime_error(exc)


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(
        ...,
        help="Config key path (example: pull.offline).",
        autocompletion=_complete_config_keys,
    ),
) -> None:
    """Unset one config value by writing null."""
    key_path = _resolve_config_key_path(key)
    updates = {key_path[0]: {key_path[1]: None}}
    try:
        update_config(TollamaPaths.default(), updates)
    except ConfigFileError as exc:
        _exit_with_runtime_error(exc)


@config_app.command("keys")
def config_keys(
    json_output: bool = typer.Option(False, "--json", help="Print compact JSON."),
) -> None:
    """List writable config keys with descriptions and current values."""
    try:
        config = load_config(TollamaPaths.default())
    except ConfigFileError as exc:
        _exit_with_runtime_error(exc)

    entries = _config_key_entries(config)
    if json_output:
        typer.echo(json.dumps(entries, separators=(",", ":"), sort_keys=True))
        return

    rows = [
        (entry["key"], _format_config_scalar(entry["value"]), entry["description"])
        for entry in entries
    ]
    typer.echo(_render_table(("KEY", "VALUE", "DESCRIPTION"), rows))


@config_app.command("init")
def config_init(
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing config.json.",
    ),
) -> None:
    """Write a default config.json template with valid JSON values."""
    paths = TollamaPaths.default()
    config_path = paths.config_path

    if config_path.exists() and not force:
        _exit_with_message(
            f"config already exists at {config_path} (use --force to overwrite)",
            code=1,
        )

    try:
        save_config(paths, TollamaConfig())
    except OSError as exc:
        _exit_with_message(f"Error: unable to write {config_path}: {exc}", code=1)
    typer.echo(f"Wrote default config to {config_path}")


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
            api_key=_env_api_key(),
        )
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    if json_output:
        typer.echo(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    typer.echo(_render_info_report(snapshot, verbose=verbose))


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Print JSON diagnostics output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(2.0, "--timeout", min=0.1, help="Daemon call timeout."),
) -> None:
    """Run local and daemon health checks with pass/warn/fail summaries."""
    checks = _collect_doctor_checks(
        base_url=base_url,
        timeout=timeout,
        paths=TollamaPaths.default(),
    )
    summary = {"pass": 0, "warn": 0, "fail": 0}
    for check in checks:
        status = check["status"]
        if status in summary:
            summary[status] += 1

    if json_output:
        typer.echo(json.dumps({"checks": checks, "summary": summary}, indent=2, sort_keys=True))
    else:
        typer.echo(_style_text("tollama doctor", bold=True))
        for check in checks:
            label = check["status"].upper()
            fg = None
            if check["status"] == "pass":
                fg = _COLOR_SUCCESS
            elif check["status"] == "warn":
                fg = _COLOR_WARNING
            elif check["status"] == "fail":
                fg = _COLOR_ERROR
            typer.echo(f"  {_style_text(f'{label:<5}', fg=fg)} {check['message']}")
        typer.echo(
            "Summary: "
            f"{summary['pass']} passed, {summary['warn']} warning, {summary['fail']} failed",
        )

    if summary["fail"] > 0:
        raise typer.Exit(code=2)
    if summary["warn"] > 0:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command("list")
def list_models(
    json_output: bool = typer.Option(False, "--json", help="Print JSON output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """List installed models via GET /api/tags."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.list_tags()
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    if json_output:
        typer.echo(json.dumps(response, indent=2, sort_keys=True))
        return

    models = response.get("models")
    if not isinstance(models, list):
        models = []
    typer.echo(_render_model_table(models))


@app.command("ps")
def ps(
    json_output: bool = typer.Option(False, "--json", help="Print JSON output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Show loaded models via GET /api/ps."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.list_running()
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    if json_output:
        typer.echo(json.dumps(response, indent=2, sort_keys=True))
        return

    models = response.get("models")
    if not isinstance(models, list):
        models = []
    typer.echo(_render_running_table(models))


@app.command("show")
def show(
    model: str = typer.Argument(
        ...,
        help="Model name to inspect.",
        autocompletion=_complete_model_names,
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Show model metadata via POST /api/show."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.show_model(model)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("explain")
def explain(
    model: str = typer.Argument(
        ...,
        help="Model name to explain.",
        autocompletion=_complete_model_names,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print JSON capability summary.",
    ),
) -> None:
    """Explain model capabilities, limits, license, and practical use cases."""
    try:
        spec = get_model_spec(model)
    except KeyError:
        _exit_with_message(
            f"model {model!r} is not defined in model-registry/registry.yaml",
            code=1,
        )

    manifest = _find_installed_manifest_by_name(model)
    payload = _build_explain_payload(spec=spec, manifest=manifest)
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(_render_explain_payload(payload))


@app.command("rm")
def rm(
    model: str = typer.Argument(
        ...,
        help="Installed model name to remove.",
        autocompletion=_complete_model_names,
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Delete a model via DELETE /api/delete."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.remove_model(model)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@modelfile_app.command("list")
def modelfile_list(
    json_output: bool = typer.Option(False, "--json", help="Print JSON output."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """List stored TSModelfile profiles."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.list_modelfiles()
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    if json_output:
        typer.echo(json.dumps(response, indent=2, sort_keys=True))
        return

    items = response.get("modelfiles")
    if not isinstance(items, list) or not items:
        typer.echo("No modelfiles found.")
        return

    rows: list[tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        profile = item.get("profile")
        model = "-"
        horizon = "-"
        if isinstance(profile, dict):
            model = _string_or_dash(profile.get("model"))
            horizon_value = profile.get("horizon")
            if isinstance(horizon_value, int):
                horizon = str(horizon_value)
        rows.append((_string_or_dash(item.get("name")), model, horizon))

    typer.echo(_render_table(("NAME", "MODEL", "HORIZON"), rows))


@modelfile_app.command("show")
def modelfile_show(
    name: str = typer.Argument(..., help="Modelfile name to inspect."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Show one TSModelfile profile."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.show_modelfile(name)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@modelfile_app.command("create")
def modelfile_create(
    name: str = typer.Argument(..., help="Modelfile name."),
    file: Path | None = typer.Option(
        None,
        "--file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="YAML modelfile content path.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Base model name (used when --file is omitted).",
    ),
    horizon: int | None = typer.Option(
        None,
        "--horizon",
        min=1,
        help="Default horizon (used when --file is omitted).",
    ),
    quantiles: str | None = typer.Option(
        None,
        "--quantiles",
        help="Comma-separated quantiles, e.g. 0.1,0.5,0.9.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Create or update one TSModelfile profile."""
    if file is None and model is None:
        typer.echo("Error: provide --file or --model", err=True)
        raise typer.Exit(code=1)

    payload: dict[str, Any] = {"name": name}
    if file is not None:
        payload["content"] = file.read_text(encoding="utf-8")
    else:
        profile: dict[str, Any] = {"model": model}
        if horizon is not None:
            profile["horizon"] = horizon
        if quantiles is not None:
            profile["quantiles"] = _parse_quantiles_option(quantiles)
        payload["profile"] = profile

    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.create_modelfile(
            name=name,
            profile=payload.get("profile") if isinstance(payload.get("profile"), dict) else None,
            content=payload.get("content") if isinstance(payload.get("content"), str) else None,
        )
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@modelfile_app.command("rm")
def modelfile_rm(
    name: str = typer.Argument(..., help="Modelfile name to remove."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(10.0, min=0.1, help="HTTP timeout in seconds."),
) -> None:
    """Delete one TSModelfile profile."""
    client = _make_client(base_url=base_url, timeout=timeout)
    try:
        response = client.remove_modelfile(name)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    typer.echo(json.dumps(response, indent=2, sort_keys=True))


@app.command("run")
def run(
    model: str | None = typer.Argument(
        None,
        help="Model name to run. Omit to select from installed models in an interactive terminal.",
        autocompletion=_complete_model_names,
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Forecast request JSON file. If omitted, reads stdin or example payload.",
    ),
    horizon: int | None = typer.Option(None, "--horizon", min=1, help="Override request horizon."),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable or disable streaming forecast output.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate request only without running inference.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Prompt to choose an example request file when --input is omitted.",
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Accept model license terms when auto-pulling a missing model.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(
        _RUN_TIMEOUT_SECONDS,
        min=0.1,
        help="HTTP timeout in seconds. Increase for first-run model load/inference.",
    ),
    progress: ProgressMode = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto, on, or off.",
    ),
) -> None:
    """Run a forecast through POST /api/forecast, auto-pulling if needed."""
    show_progress = _resolve_progress_enabled(progress)
    client = _make_client(base_url=base_url, timeout=timeout)
    resolved_model = _resolve_run_model_name(model, client=client)
    _emit_uni2ts_python_runtime_warning(resolved_model)
    payload = _load_request_payload(path=input_path, model=resolved_model, interactive=interactive)
    payload["model"] = resolved_model
    if horizon is not None:
        payload["horizon"] = horizon
    if timeout is not None:
        payload["timeout"] = timeout

    if dry_run:
        _progress_note(show_progress, "Validating request payload ...")
        try:
            validation_result = client.validate_request(payload)
        except RuntimeError as exc:
            _exit_with_runtime_error(exc)

        typer.echo(json.dumps(validation_result, indent=2, sort_keys=True))
        is_valid = bool(validation_result.get("valid"))
        raise typer.Exit(code=0 if is_valid else 2)

    _progress_note(show_progress, f"Checking model availability for {resolved_model!r} ...")
    try:
        client.show_model(resolved_model)
    except DaemonHTTPError as exc:
        if exc.status_code != 404:
            _exit_with_runtime_error(exc)
        _progress_note(show_progress, f"Model {resolved_model!r} is missing, pulling ...")
        try:
            pull_result = client.pull_model(
                name=resolved_model,
                stream=stream,
                accept_license=accept_license,
            )
        except RuntimeError as pull_exc:
            _exit_with_runtime_error(pull_exc)
        _emit_result(
            pull_result,
            stream_kind="pull",
            show_progress=show_progress,
        )
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    _progress_note(show_progress, f"Running forecast with {resolved_model!r} ...")
    try:
        forecast_result = client.forecast(payload, stream=stream)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)
    _emit_result(
        forecast_result,
        stream_kind="forecast",
        show_progress=show_progress,
    )
    _progress_note(show_progress, "Forecast complete.")


def _resolve_run_model_name(model: str | None, *, client: TollamaClient) -> str:
    if model is not None:
        return model
    if not sys.stdin.isatty():
        _exit_with_message(
            "missing model name. Provide MODEL argument or run in an interactive terminal.",
            code=2,
        )

    try:
        response = client.list_tags()
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    names: list[str] = []
    models = response.get("models")
    if isinstance(models, list):
        for item in models:
            if not isinstance(item, dict):
                continue
            candidate = item.get("name") or item.get("model")
            if isinstance(candidate, str) and candidate and candidate not in names:
                names.append(candidate)

    if not names:
        _exit_with_message(
            "no installed models found. Pull one with `tollama pull <model>`.",
            code=1,
        )

    typer.echo(_style_text("Select a model:", bold=True))
    for index, name in enumerate(names, start=1):
        typer.echo(f"  {index}. {name}")

    response_text = typer.prompt("Enter a number", default="1").strip()
    try:
        selected_index = int(response_text)
    except ValueError:
        _exit_with_message(f"invalid selection {response_text!r}; expected a number", code=1)

    if selected_index < 1 or selected_index > len(names):
        _exit_with_message(f"selection must be between 1 and {len(names)}", code=1)
    return names[selected_index - 1]


@app.command("quickstart")
def quickstart(
    model: str = typer.Option(
        _QUICKSTART_MODEL,
        "--model",
        autocompletion=_complete_model_names,
        help="Model name to use for quickstart.",
    ),
    horizon: int = typer.Option(
        _QUICKSTART_HORIZON,
        "--horizon",
        min=1,
        help="Forecast horizon for quickstart demo payload.",
    ),
    accept_license: bool = typer.Option(
        False,
        "--accept-license",
        help="Accept model license terms when required by pull.",
    ),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL,
        help="Daemon base URL. Defaults to http://localhost:11435.",
    ),
    timeout: float = typer.Option(30.0, min=0.1, help="HTTP timeout in seconds."),
    progress: ProgressMode = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto, on, or off.",
    ),
) -> None:
    """Pull a model, run demo forecast, and print next-step commands."""
    show_progress = _resolve_progress_enabled(progress)
    client = _make_client(base_url=base_url, timeout=timeout)

    _progress_note(show_progress, "Checking daemon health ...")
    try:
        client.health()
    except RuntimeError as exc:
        typer.echo(
            _style_text(
                f"Error: unable to reach tollama daemon at {base_url}: {exc}",
                fg=_COLOR_ERROR,
                err=True,
            ),
            err=True,
        )
        hint = _error_hint(exc)
        if hint is not None:
            typer.echo(_style_text(f"Hint: {hint}", fg=_COLOR_WARNING, err=True), err=True)
        else:
            typer.echo(
                _style_text("Hint: Start it with `tollama serve`.", fg=_COLOR_WARNING, err=True),
                err=True,
            )
        raise typer.Exit(code=1) from exc

    _progress_note(show_progress, f"Pulling {model!r} ...")
    try:
        pull_result = client.pull_model(
            name=model,
            stream=False,
            accept_license=accept_license,
        )
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    request_payload = _quickstart_request_payload(model=model, horizon=horizon)
    _progress_note(show_progress, f"Forecasting with {model!r} ...")
    try:
        forecast_result = client.forecast(request_payload, stream=False)
    except RuntimeError as exc:
        _exit_with_runtime_error(exc)

    typer.echo(_style_text("tollama quickstart complete", fg=_COLOR_SUCCESS, bold=True))
    typer.echo("")
    typer.echo(_style_text("Pull result:", bold=True))
    _emit_result(pull_result, stream_kind="pull", show_progress=show_progress)
    typer.echo("")
    typer.echo(_style_text("Forecast result:", bold=True))
    _emit_result(forecast_result, stream_kind="forecast", show_progress=show_progress)
    typer.echo("")
    typer.echo(_style_text("Next steps:", bold=True))
    typer.echo("  1. tollama list")
    typer.echo("  2. tollama run mock --input examples/request.json --no-stream")
    typer.echo(
        "  3. python -c \"from tollama import Tollama; "
        "print(Tollama().models('available'))\"",
    )


def _quickstart_request_payload(*, model: str, horizon: int) -> dict[str, Any]:
    timestamps = [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
        "2025-01-04",
        "2025-01-05",
    ]
    target = [10.0, 11.0, 12.0, 13.0, 14.0]
    return {
        "model": model,
        "horizon": horizon,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "demo_series",
                "freq": "D",
                "timestamps": timestamps,
                "target": target,
            }
        ],
        "options": {},
    }


def _emit_uni2ts_python_runtime_warning(model: str) -> None:
    if not _is_python_312_or_newer():
        return

    model_name = model.lower()
    if "moirai" in model_name or "uni2ts" in model_name:
        typer.echo(
            _style_text(f"warning: {_UNI2TS_PYTHON_WARNING}", fg=_COLOR_WARNING, err=True),
            err=True,
        )
        return

    try:
        spec = get_model_spec(model)
    except KeyError:
        return
    if spec.family == "uni2ts":
        typer.echo(
            _style_text(f"warning: {_UNI2TS_PYTHON_WARNING}", fg=_COLOR_WARNING, err=True),
            err=True,
        )


def _is_python_312_or_newer() -> bool:
    return (sys.version_info.major, sys.version_info.minor) >= (3, 12)


def _emit_result(
    result: dict[str, Any] | list[dict[str, Any]],
    *,
    stream_kind: str = "generic",
    show_progress: bool = False,
) -> None:
    if isinstance(result, list):
        pull_progress: tqdm[Any] | None = None
        for item in result:
            _emit_warnings(item)
            if stream_kind == "pull" and _is_pull_progress_event(item):
                if show_progress:
                    pull_progress = _update_pull_progress_bar(pull_progress, item)
                else:
                    status = item.get("status")
                    if isinstance(status, str) and status:
                        typer.echo(f" {status}", err=True)
                continue
            typer.echo(json.dumps(item, sort_keys=True))
        if pull_progress is not None:
            pull_progress.close()
        return
    _emit_warnings(result)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def _emit_warnings(payload: dict[str, Any]) -> None:
    warnings = payload.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            if isinstance(warning, str) and warning:
                typer.echo(
                    _style_text(f"warning: {warning}", fg=_COLOR_WARNING, err=True),
                    err=True,
                )

    response = payload.get("response")
    if isinstance(response, dict):
        nested = response.get("warnings")
        if isinstance(nested, list):
            for warning in nested:
                if isinstance(warning, str) and warning:
                    typer.echo(
                        _style_text(f"warning: {warning}", fg=_COLOR_WARNING, err=True),
                        err=True,
                    )


def _is_pull_progress_event(item: dict[str, Any]) -> bool:
    status = item.get("status")
    if not isinstance(status, str):
        return False
    return status not in {"success"}


def _update_pull_progress_bar(
    progress_bar: tqdm[Any] | None,
    event: dict[str, Any],
) -> tqdm[Any] | None:
    status = event.get("status")
    completed = event.get("completed_bytes")
    total = event.get("total_bytes")
    has_bytes = isinstance(completed, int) and not isinstance(completed, bool)
    has_total = isinstance(total, int) and not isinstance(total, bool) and total > 0

    if not has_bytes and not has_total:
        if isinstance(status, str) and status:
            typer.echo(_style_text(f" {status}", fg=_COLOR_DIM, err=True), err=True)
        return progress_bar

    if progress_bar is None:
        initial_total = total if has_total else None
        progress_bar = tqdm(
            total=initial_total,
            unit="B",
            unit_scale=True,
            desc="pull",
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
        )

    if has_total and progress_bar.total != total:
        progress_bar.total = total

    if has_bytes:
        if completed >= progress_bar.n:
            progress_bar.update(completed - progress_bar.n)
        else:
            progress_bar.n = completed
            progress_bar.refresh()

    if isinstance(status, str) and status:
        progress_bar.set_description_str(status)
    return progress_bar


def _render_model_table(models: list[dict[str, Any]]) -> str:
    rows: list[tuple[str, str, str, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = _string_or_dash(model.get("name") or model.get("model"))
        family = "-"
        details = model.get("details")
        if isinstance(details, dict):
            family = _string_or_dash(details.get("family"))
        if family == "-":
            family = _string_or_dash(model.get("family"))

        size_value = model.get("size")
        if isinstance(size_value, int) and not isinstance(size_value, bool):
            size = _format_bytes(size_value)
        else:
            size = "-"

        modified = _string_or_dash(model.get("modified_at"))
        rows.append((name, family, size, modified))

    if not rows:
        return "No models installed."
    return _render_table(("NAME", "FAMILY", "SIZE", "MODIFIED"), rows, right_align={2})


def _render_running_table(models: list[dict[str, Any]]) -> str:
    rows: list[tuple[str, str, str]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = _string_or_dash(model.get("name") or model.get("model"))
        family = "-"
        details = model.get("details")
        if isinstance(details, dict):
            family = _string_or_dash(details.get("family"))
        if family == "-":
            family = _string_or_dash(model.get("family"))
        expires = _string_or_dash(model.get("expires_at"))
        rows.append((name, family, expires))

    if not rows:
        return "No models loaded."
    return _render_table(("NAME", "FAMILY", "EXPIRES"), rows)


def _find_installed_manifest_by_name(model: str) -> dict[str, Any] | None:
    try:
        installed = list_installed()
    except Exception:  # noqa: BLE001
        return None
    for item in installed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name == model:
            return item
    return None


def _build_explain_payload(
    *,
    spec: ModelSpec,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    capabilities = spec.capabilities or ModelCapabilities()
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    manifest_license = manifest.get("license") if isinstance(manifest, dict) else {}
    accepted = None
    if isinstance(manifest_license, dict) and "accepted" in manifest_license:
        accepted = bool(manifest_license.get("accepted"))

    max_horizon = _resolve_numeric_metadata(metadata, ("max_horizon", "prediction_length"))
    max_context = _resolve_numeric_metadata(
        metadata,
        ("max_context", "context_length", "default_context_length"),
    )

    return {
        "model": spec.name,
        "family": spec.family,
        "installed": manifest is not None,
        "source": spec.source.model_dump(mode="json"),
        "license": {
            "type": spec.license.type,
            "needs_acceptance": spec.license.needs_acceptance,
            "accepted": accepted,
            "notice": spec.license.notice,
        },
        "limits": {
            "max_horizon": max_horizon,
            "max_context": max_context,
        },
        "capabilities": {
            "past_covariates_numeric": capabilities.past_covariates_numeric,
            "past_covariates_categorical": capabilities.past_covariates_categorical,
            "future_covariates_numeric": capabilities.future_covariates_numeric,
            "future_covariates_categorical": capabilities.future_covariates_categorical,
            "static_covariates": capabilities.static_covariates,
        },
        "recommended_use_cases": _recommended_use_cases(spec=spec, capabilities=capabilities),
    }


def _resolve_numeric_metadata(metadata: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value > 0 and value.is_integer():
            return int(value)
    return None


def _recommended_use_cases(*, spec: ModelSpec, capabilities: ModelCapabilities) -> list[str]:
    use_cases: list[str] = []
    family = spec.family
    if family == "mock":
        use_cases.append("Fast local smoke tests and integration checks.")
    if family in {"torch", "timesfm", "uni2ts"}:
        use_cases.append("General purpose forecasting with covariate-aware workflows.")
    if family == "timesfm":
        use_cases.append("High-throughput batch forecasting for many related series.")
    if family == "sundial":
        use_cases.append("Long-horizon forecasting where covariates are unavailable.")
    if family == "toto":
        use_cases.append("Large-context production workloads with numeric history features.")

    if capabilities.future_covariates_numeric or capabilities.future_covariates_categorical:
        use_cases.append("Known-future covariate forecasting scenarios.")
    elif capabilities.past_covariates_numeric or capabilities.past_covariates_categorical:
        use_cases.append("Historical covariate enrichment without future covariates.")
    else:
        use_cases.append("Target-only forecasting from clean univariate history.")

    # Stable order, deduplicated.
    deduped: list[str] = []
    for item in use_cases:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _render_explain_payload(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(_style_text(str(payload.get("model") or "-"), bold=True))
    lines.append(f"family: {_string_or_dash(payload.get('family'))}")
    lines.append(f"installed: {'yes' if payload.get('installed') else 'no'}")

    source = payload.get("source")
    if isinstance(source, dict):
        repo_id = _string_or_dash(source.get("repo_id"))
        revision = _string_or_dash(source.get("revision"))
        lines.append(f"source: {repo_id} @ {revision}")

    license_payload = payload.get("license")
    if isinstance(license_payload, dict):
        lines.append("")
        lines.append(_style_text("license", bold=True))
        lines.append(f"  type: {_string_or_dash(license_payload.get('type'))}")
        needs = bool(license_payload.get("needs_acceptance"))
        lines.append(f"  acceptance required: {'yes' if needs else 'no'}")
        accepted = license_payload.get("accepted")
        if isinstance(accepted, bool):
            lines.append(f"  accepted locally: {'yes' if accepted else 'no'}")
        notice = license_payload.get("notice")
        if isinstance(notice, str) and notice.strip():
            lines.append(f"  notice: {notice.strip()}")

    limits = payload.get("limits")
    if isinstance(limits, dict):
        lines.append("")
        lines.append(_style_text("limits", bold=True))
        lines.append(f"  max_horizon: {_string_or_dash(limits.get('max_horizon'))}")
        lines.append(f"  max_context: {_string_or_dash(limits.get('max_context'))}")

    capabilities = payload.get("capabilities")
    if isinstance(capabilities, dict):
        lines.append("")
        lines.append(_style_text("capabilities", bold=True))
        for key in (
            "past_covariates_numeric",
            "past_covariates_categorical",
            "future_covariates_numeric",
            "future_covariates_categorical",
            "static_covariates",
        ):
            value = capabilities.get(key)
            if isinstance(value, bool):
                rendered = "yes" if value else "no"
            else:
                rendered = _string_or_dash(value)
            lines.append(f"  {key}: {rendered}")

    use_cases = payload.get("recommended_use_cases")
    if isinstance(use_cases, list) and use_cases:
        lines.append("")
        lines.append(_style_text("recommended use cases", bold=True))
        for item in use_cases:
            if isinstance(item, str) and item:
                lines.append(f"  - {item}")

    return "\n".join(lines)


def _render_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    right_align: set[int] | None = None,
) -> str:
    return _render_table_with_layout(headers, rows, right_align=right_align)


def _render_table_with_layout(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    *,
    right_align: set[int] | None = None,
) -> str:
    normalized_rows = [
        tuple(_truncate_cell(str(value), max_width=_TABLE_MAX_COL_WIDTH) for value in row)
        for row in rows
    ]
    widths = [len(_truncate_cell(header, max_width=_TABLE_MAX_COL_WIDTH)) for header in headers]
    for row in normalized_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    aligned_headers = []
    for idx, header in enumerate(headers):
        truncated = _truncate_cell(header, max_width=_TABLE_MAX_COL_WIDTH)
        aligned_headers.append(_align_cell(truncated, widths[idx], idx, right_align))
    header_line = (" " * _TABLE_DEFAULT_GAP).join(aligned_headers)
    separator_parts = ["-" * widths[idx] for idx in range(len(headers))]
    separator_line = (" " * _TABLE_DEFAULT_GAP).join(separator_parts)
    row_lines = [
        (" " * _TABLE_DEFAULT_GAP).join(
            _align_cell(value, widths[idx], idx, right_align)
            for idx, value in enumerate(row)
        )
        for row in normalized_rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _truncate_cell(value: str, *, max_width: int) -> str:
    if max_width < 4:
        return value[:max_width]
    if len(value) <= max_width:
        return value
    return f"{value[: max_width - 3]}..."


def _align_cell(value: str, width: int, index: int, right_align: set[int] | None) -> str:
    if right_align is not None and index in right_align:
        return value.rjust(width)
    return value.ljust(width)


def _format_bytes(size: int) -> str:
    if size < 0:
        return "-"

    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1

    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    if value >= 10:
        return f"{value:.1f} {units[unit_index]}"
    return f"{value:.2f} {units[unit_index]}"


def _string_or_dash(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "-"


def _parse_quantiles_option(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        normalized = chunk.strip()
        if not normalized:
            continue
        try:
            values.append(float(normalized))
        except ValueError as exc:
            raise typer.BadParameter(f"invalid quantile value: {normalized!r}") from exc
    if not values:
        raise typer.BadParameter("at least one quantile is required")
    return values


def _env_api_key() -> str | None:
    value = os.environ.get(_API_KEY_ENV_NAME)
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _make_client(*, base_url: str, timeout: float) -> TollamaClient:
    api_key = _env_api_key()
    if api_key is not None:
        try:
            return TollamaClient(base_url=base_url, timeout=timeout, api_key=api_key)
        except TypeError:
            # Some tests monkeypatch ``TollamaClient`` with a legacy constructor.
            return TollamaClient(base_url=base_url, timeout=timeout)  # type: ignore[call-arg]
    return TollamaClient(base_url=base_url, timeout=timeout)


def _auth_header(api_key: str | None) -> dict[str, str] | None:
    if api_key is None:
        return None
    return {"Authorization": f"Bearer {api_key}"}


def _collect_doctor_checks(
    *,
    base_url: str,
    timeout: float,
    paths: TollamaPaths,
) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []

    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    for family in FAMILY_EXTRAS:
        name = f"python_constraint_{family}"
        constraint = FAMILY_PYTHON_CONSTRAINTS.get(family)
        if constraint is None:
            checks.append(
                {
                    "name": name,
                    "status": "pass",
                    "message": f"Python {python_version} has no special constraint for {family}",
                },
            )
            continue

        if _python_version_satisfies(constraint):
            checks.append(
                {
                    "name": name,
                    "status": "pass",
                    "message": f"Python {python_version} satisfies {constraint} for {family}",
                },
            )
        else:
            checks.append(
                {
                    "name": name,
                    "status": "fail",
                    "message": f"Python {python_version} violates {constraint} for {family}",
                },
            )

    for status in list_runtime_statuses(paths=paths):
        family = str(status.get("family") or "unknown")
        installed = bool(status.get("installed"))
        checks.append(
            {
                "name": f"runtime_{family}",
                "status": "pass" if installed else "warn",
                "message": f"runtime_{family}: {'installed' if installed else 'not installed'}",
            },
        )

    free_bytes = shutil.disk_usage(str(paths.base_dir.parent)).free
    free_gib = free_bytes / (1024**3)
    if free_gib < 1.0:
        disk_status = "fail"
    elif free_gib <= 5.0:
        disk_status = "warn"
    else:
        disk_status = "pass"
    checks.append(
        {
            "name": "disk_space",
            "status": disk_status,
            "message": f"disk_space: {free_gib:.1f} GiB free",
        },
    )

    has_hf_token = any(os.environ.get(key) for key in _DOCTOR_TOKEN_ENV_KEYS)
    checks.append(
        {
            "name": "hf_token",
            "status": "pass" if has_hf_token else "warn",
            "message": (
                "hf_token: token environment variable detected"
                if has_hf_token
                else "hf_token: no token environment variable detected"
            ),
        },
    )

    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
            auth_header = _auth_header(_env_api_key())
            if auth_header is None:
                response = client.get("/v1/health")
            else:
                response = client.get("/v1/health", headers=auth_header)
    except httpx.HTTPError as exc:
        checks.append(
            {
                "name": "daemon",
                "status": "fail",
                "message": f"daemon: unreachable at {base_url} ({exc})",
            },
        )
    else:
        if response.is_success:
            checks.append(
                {
                    "name": "daemon",
                    "status": "pass",
                    "message": f"daemon: reachable at {base_url}",
                },
            )
        else:
            checks.append(
                {
                    "name": "daemon",
                    "status": "fail",
                    "message": f"daemon: {base_url} returned HTTP {response.status_code}",
                },
            )

    config_path = paths.config_path
    if not config_path.exists():
        checks.append(
            {
                "name": "config",
                "status": "pass",
                "message": f"config: missing at {config_path} (using defaults)",
            },
        )
    else:
        try:
            load_config(paths)
        except ConfigFileError as exc:
            checks.append(
                {
                    "name": "config",
                    "status": "fail",
                    "message": f"config: invalid at {config_path} ({exc})",
                },
            )
        else:
            checks.append(
                {
                    "name": "config",
                    "status": "pass",
                    "message": f"config: valid at {config_path}",
                },
            )

    return checks


def _python_version_satisfies(constraint: str) -> bool:
    current = (sys.version_info.major, sys.version_info.minor)
    for operator in ("<=", ">=", "==", "<", ">"):
        if not constraint.startswith(operator):
            continue

        raw_target = constraint[len(operator) :].strip()
        target = _parse_version_tuple(raw_target)
        if target is None:
            return False
        normalized_current = _normalize_version_tuple(current, len(target))
        normalized_target = _normalize_version_tuple(target, len(target))

        if operator == "<":
            return normalized_current < normalized_target
        if operator == "<=":
            return normalized_current <= normalized_target
        if operator == ">":
            return normalized_current > normalized_target
        if operator == ">=":
            return normalized_current >= normalized_target
        return normalized_current == normalized_target
    return False


def _parse_version_tuple(value: str) -> tuple[int, ...] | None:
    parts = value.split(".")
    if not parts:
        return None
    parsed: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        parsed.append(int(part))
    return tuple(parsed)


def _normalize_version_tuple(value: tuple[int, ...], width: int) -> tuple[int, ...]:
    if len(value) >= width:
        return value[:width]
    return (*value, *([0] * (width - len(value))))


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
        suggestion = get_close_matches(key, sorted(_CONFIG_KEY_PATHS), n=1, cutoff=0.6)
        if suggestion:
            _exit_with_message(f"unknown key {key!r}. Did you mean {suggestion[0]!r}?")
        supported = ", ".join(sorted(_CONFIG_KEY_PATHS))
        _exit_with_message(f"unknown key {key!r}. Supported keys: {supported}")
    return key_path


def _config_key_entries(config: TollamaConfig) -> list[dict[str, Any]]:
    payload = config.model_dump(mode="json")
    entries: list[dict[str, Any]] = []
    for key in sorted(_CONFIG_KEY_PATHS):
        section, field = _CONFIG_KEY_PATHS[key]
        section_payload = payload.get(section)
        value: Any = None
        if isinstance(section_payload, dict):
            value = section_payload.get(field)
        entries.append(
            {
                "key": key,
                "value": value,
                "description": CONFIG_KEY_DESCRIPTIONS[key],
            }
        )
    return entries


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


def _load_request_payload(
    path: Path | None,
    *,
    model: str,
    interactive: bool = False,
) -> dict[str, Any]:
    if path is not None:
        return _load_request_payload_from_path(path)

    stdin_payload = _load_request_payload_from_stdin()
    if stdin_payload is not None:
        return stdin_payload

    if interactive:
        selected = _prompt_example_request_path(model)
        if selected is not None:
            return _load_request_payload_from_path(selected)

    default_path = _resolve_default_request_path(model)
    if default_path is not None:
        return _load_request_payload_from_path(default_path)

    _exit_with_message(
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
    candidates = _candidate_request_paths(model)
    if not candidates:
        return None
    return candidates[0]


def _candidate_request_paths(model: str) -> list[Path]:
    candidate_names = [f"{alias}_request.json" for alias in _request_payload_aliases(model)]
    candidate_names.append("request.json")
    candidates: list[Path] = []

    roots: list[Path] = [Path.cwd()]
    package_root = _project_root_from_module()
    if package_root is not None and package_root not in roots:
        roots.append(package_root)

    for root in roots:
        example_dir = root / "examples"
        for name in candidate_names:
            candidate = example_dir / name
            if candidate.is_file():
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def _prompt_example_request_path(model: str) -> Path | None:
    if not sys.stdin.isatty():
        return None

    candidates = _candidate_request_paths(model)
    if not candidates:
        return None

    typer.echo(_style_text("Select an example request payload:", bold=True))
    for index, candidate in enumerate(candidates, start=1):
        typer.echo(f"  {index}. {candidate}")

    response = typer.prompt(
        "Enter a number (blank to skip)",
        default="",
        show_default=False,
    ).strip()
    if not response:
        return None

    try:
        selected_index = int(response)
    except ValueError:
        _exit_with_message(f"invalid selection {response!r}; expected a number", code=1)

    if selected_index < 1 or selected_index > len(candidates):
        _exit_with_message(
            f"selection must be between 1 and {len(candidates)}",
            code=1,
        )

    return candidates[selected_index - 1]


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
        _exit_with_message(f"{source} is not valid JSON: {exc}")

    if not isinstance(payload, dict):
        _exit_with_message(f"{source} JSON must be an object")

    return payload


# ---------------------------------------------------------------------------
# ``tollama runtime`` subcommands
# ---------------------------------------------------------------------------

_ALL_BOOTSTRAPPABLE = list(FAMILY_EXTRAS.keys())


@runtime_app.command("list")
def runtime_list(
    json_output: bool = typer.Option(False, "--json", help="Print JSON output."),
) -> None:
    """Show status of per-family isolated runner environments."""
    paths = TollamaPaths.default()
    statuses = list_runtime_statuses(paths=paths)

    if json_output:
        typer.echo(json.dumps(statuses, indent=2))
        return

    for entry in statuses:
        mark = "✓" if entry["installed"] else "✗"
        version = entry.get("tollama_version") or "—"
        py_ver = entry.get("python_version") or "—"
        constraint = entry.get("python_constraint")
        constraint_tag = f"  (requires python{constraint})" if constraint else ""
        mark_fg = _COLOR_SUCCESS if entry["installed"] else _COLOR_ERROR
        typer.echo(
            f"  {_style_text(mark, fg=mark_fg)}  {entry['family']:<12}  tollama={version}"
            f"  python={py_ver}{constraint_tag}"
        )


@runtime_app.command("install")
def runtime_install(
    family: str = typer.Argument(
        None,
        help="Runner family to install (torch, timesfm, uni2ts, sundial, toto).",
    ),
    all_families: bool = typer.Option(False, "--all", help="Install all runner families."),
    reinstall: bool = typer.Option(
        False, "--reinstall", help="Force reinstall even if up-to-date.",
    ),
    progress: ProgressMode = typer.Option(
        "auto",
        "--progress",
        help="Progress display mode: auto, on, or off.",
    ),
) -> None:
    """Create or update an isolated venv for a runner family."""
    families = _resolve_runtime_families(family, all_families=all_families)
    paths = TollamaPaths.default()
    show_progress = _resolve_progress_enabled(progress)

    for fam in families:
        typer.echo(f"Installing runtime for {fam!r} ...")
        _progress_note(
            show_progress,
            f"Bootstrapping virtualenv and installing dependencies for {fam!r} ...",
        )
        try:
            python_path = ensure_family_runtime(fam, paths=paths, reinstall=reinstall)
            typer.echo(_style_text(f"  ✓ {fam}: {python_path}", fg=_COLOR_SUCCESS))
        except BootstrapError as exc:
            typer.echo(_style_text(f"  ✗ {fam}: {exc}", fg=_COLOR_ERROR, err=True), err=True)
            raise typer.Exit(code=1) from exc


@runtime_app.command("remove")
def runtime_remove(
    family: str = typer.Argument(
        None,
        help="Runner family to remove (torch, timesfm, uni2ts, sundial, toto).",
    ),
    all_families: bool = typer.Option(False, "--all", help="Remove all runner runtimes."),
) -> None:
    """Remove an isolated runner virtualenv."""
    families = _resolve_runtime_families(family, all_families=all_families)
    paths = TollamaPaths.default()

    for fam in families:
        removed = remove_family_runtime(fam, paths=paths)
        if removed:
            typer.echo(f"  ✓ removed {fam}")
        else:
            typer.echo(f"  — {fam} not installed")


@runtime_app.command("update")
def runtime_update(
    family: str = typer.Argument(
        None,
        help="Runner family to update.",
    ),
    all_families: bool = typer.Option(False, "--all", help="Update all installed runtimes."),
) -> None:
    """Reinstall runner runtime(s) to pick up tollama version changes."""
    paths = TollamaPaths.default()

    if all_families:
        statuses = list_runtime_statuses(paths=paths)
        families = [s["family"] for s in statuses if s["installed"]]
        if not families:
            typer.echo("No runtimes installed.")
            return
    elif family:
        families = [family]
    else:
        typer.echo("Error: provide a family name or --all", err=True)
        raise typer.Exit(code=1)

    for fam in families:
        typer.echo(f"Updating runtime for {fam!r} …")
        try:
            python_path = ensure_family_runtime(fam, paths=paths, reinstall=True)
            typer.echo(f"  ✓ {fam}: {python_path}")
        except BootstrapError as exc:
            typer.echo(f"  ✗ {fam}: {exc}", err=True)
            raise typer.Exit(code=1) from exc


def _resolve_runtime_families(family: str | None, *, all_families: bool) -> list[str]:
    """Resolve family argument + --all flag to a list of family names."""
    if all_families:
        return list(_ALL_BOOTSTRAPPABLE)
    if family is None:
        typer.echo("Error: provide a family name or --all", err=True)
        raise typer.Exit(code=1)
    if family not in FAMILY_EXTRAS:
        typer.echo(
            f"Error: unknown family {family!r}; choose from {', '.join(FAMILY_EXTRAS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    return [family]


def _dashboard_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized:
        normalized = DEFAULT_BASE_URL
    return f"{normalized}/dashboard"


def main() -> None:
    """Console script entrypoint for the Typer app."""
    app()


if __name__ == "__main__":
    main()
