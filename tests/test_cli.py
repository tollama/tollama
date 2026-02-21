"""Tests for the Typer-based tollama CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from tollama.cli.client import DaemonHTTPError
from tollama.cli.main import (
    _RUN_TIMEOUT_SECONDS,
    _complete_model_names,
    _dashboard_url,
    _prompt_example_request_path,
    _render_table,
    _resolve_default_request_path,
    _resolve_progress_enabled,
    _resolve_run_model_name,
    _truncate_cell,
    app,
)
from tollama.core.registry import ModelSpec


def _sample_request_payload() -> dict[str, object]:
    return {
        "model": "original",
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [3.0, 4.0],
            }
        ],
        "options": {},
    }


def _new_runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:
        # Click<8 does not support `mix_stderr`.
        return CliRunner()


def _result_stdout(result: object) -> str:
    stdout = getattr(result, "stdout", None)
    if isinstance(stdout, str):
        return stdout
    output = getattr(result, "output", None)
    if isinstance(output, str):
        return output
    return ""


def _result_stderr(result: object) -> str:
    stderr = getattr(result, "stderr", None)
    if isinstance(stderr, str):
        return stderr
    # Click<8 may mix stderr into output.
    return _result_stdout(result)


def test_serve_runs_uvicorn_with_expected_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int, log_level: str) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("tollama.cli.main.uvicorn.run", _fake_run)

    runner = _new_runner()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert captured == {
        "target": "tollama.daemon.app:app",
        "host": "127.0.0.1",
        "port": 11435,
        "log_level": "info",
    }


def test_open_command_launches_browser(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_open(url: str) -> bool:
        captured["url"] = url
        return True

    monkeypatch.setattr("tollama.cli.main.webbrowser.open", _fake_open)
    runner = _new_runner()
    result = runner.invoke(app, ["open", "--base-url", "http://127.0.0.1:11435"])

    assert result.exit_code == 0
    assert captured["url"] == "http://127.0.0.1:11435/dashboard"
    assert "http://127.0.0.1:11435/dashboard" in _result_stdout(result)


def test_open_command_fails_when_browser_cannot_open(monkeypatch) -> None:
    monkeypatch.setattr("tollama.cli.main.webbrowser.open", lambda *_args, **_kwargs: False)
    runner = _new_runner()

    result = runner.invoke(app, ["open"])

    assert result.exit_code == 1
    assert "Unable to open browser automatically." in _result_stdout(result)


def test_dashboard_command_invokes_tui_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_dashboard_app(*, base_url: str, timeout: float, api_key: str | None) -> None:
        captured["base_url"] = base_url
        captured["timeout"] = timeout
        captured["api_key"] = api_key

    monkeypatch.setattr("tollama.tui.app.run_dashboard_app", _fake_run_dashboard_app)
    monkeypatch.setenv("TOLLAMA_API_KEY", "secret-key")
    runner = _new_runner()

    result = runner.invoke(
        app,
        ["dashboard", "--base-url", "http://localhost:11999", "--timeout", "3.5"],
    )

    assert result.exit_code == 0
    assert captured == {
        "base_url": "http://localhost:11999",
        "timeout": 3.5,
        "api_key": "secret-key",
    }


def test_dashboard_url_helper_normalizes_trailing_slash() -> None:
    assert _dashboard_url("http://localhost:11435/") == "http://localhost:11435/dashboard"
    assert _dashboard_url("") == "http://localhost:11435/dashboard"


def test_pull_supports_streaming_and_non_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "env-token")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            accept_license: bool = False,
            insecure: bool | None,
            offline: bool | None,
            local_files_only: bool | None,
            http_proxy: str | None,
            https_proxy: str | None,
            no_proxy: str | None,
            hf_home: str | None,
            max_workers: int | None = None,
            token: str | None = None,
            include_null_fields: set[str] | None = None,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["pull"] = {
                "name": name,
                "stream": stream,
                "accept_license": accept_license,
                "insecure": insecure,
                "offline": offline,
                "local_files_only": local_files_only,
                "http_proxy": http_proxy,
                "https_proxy": https_proxy,
                "no_proxy": no_proxy,
                "hf_home": hf_home,
                "max_workers": max_workers,
                "token": token,
                "include_null_fields": include_null_fields,
            }
            if stream:
                return [{"status": "pulling manifest"}, {"status": "success", "model": name}]
            return {"status": "success", "model": name}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()

    streamed = runner.invoke(app, ["pull", "mock"])
    assert streamed.exit_code == 0
    lines = [line for line in _result_stdout(streamed).splitlines() if line.strip()]
    assert lines == ['{"model": "mock", "status": "success"}']
    assert "pulling manifest" in _result_stderr(streamed)
    assert captured["base_url"] == "http://localhost:11435"
    assert captured["pull"] == {
        "name": "mock",
        "stream": True,
        "accept_license": False,
        "insecure": None,
        "offline": None,
        "local_files_only": None,
        "http_proxy": None,
        "https_proxy": None,
        "no_proxy": None,
        "hf_home": None,
        "max_workers": None,
        "token": "env-token",
        "include_null_fields": set(),
    }

    non_stream = runner.invoke(app, ["pull", "mock", "--no-stream"])
    assert non_stream.exit_code == 0
    assert json.loads(_result_stdout(non_stream)) == {"status": "success", "model": "mock"}
    assert captured["pull"]["stream"] is False

    with_accept = runner.invoke(app, ["pull", "mock", "--accept-license"])
    assert with_accept.exit_code == 0
    assert captured["pull"]["accept_license"] is True

    with_flags = runner.invoke(
        app,
        [
            "pull",
            "mock",
            "--offline",
            "--local-files-only",
            "--insecure",
            "--http-proxy",
            "http://proxy:8080",
            "--https-proxy",
            "http://proxy:8443",
            "--no-proxy",
            "localhost,127.0.0.1",
            "--hf-home",
            "/tmp/hf",
            "--token",
            "flag-token",
        ],
    )
    assert with_flags.exit_code == 0
    assert captured["pull"] == {
        "name": "mock",
        "stream": True,
        "accept_license": False,
        "insecure": True,
        "offline": True,
        "local_files_only": True,
        "http_proxy": "http://proxy:8080",
        "https_proxy": "http://proxy:8443",
        "no_proxy": "localhost,127.0.0.1",
        "hf_home": "/tmp/hf",
        "max_workers": None,
        "token": "flag-token",
        "include_null_fields": set(),
    }


def test_pull_prints_hint_when_client_error_exposes_hint(monkeypatch) -> None:
    class _HintedError(RuntimeError):
        def __init__(self, message: str, hint: str) -> None:
            super().__init__(message)
            self.hint = hint

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            accept_license: bool = False,
            insecure: bool | None = None,
            offline: bool | None = None,
            local_files_only: bool | None = None,
            http_proxy: str | None = None,
            https_proxy: str | None = None,
            no_proxy: str | None = None,
            hf_home: str | None = None,
            max_workers: int | None = None,
            token: str | None = None,
            include_null_fields: set[str] | None = None,
        ) -> dict[str, object]:
            del (
                name,
                stream,
                accept_license,
                insecure,
                offline,
                local_files_only,
                http_proxy,
                https_proxy,
                no_proxy,
                hf_home,
                max_workers,
                token,
                include_null_fields,
            )
            raise _HintedError("model missing", "Run `tollama pull <model>`.")

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()

    result = runner.invoke(app, ["pull", "missing", "--no-stream"])

    assert result.exit_code == 1
    stderr = _result_stderr(result)
    assert "Error: model missing" in stderr
    assert "Hint: Run `tollama pull <model>`." in stderr


def test_complete_model_names_uses_registry_and_installed_entries(monkeypatch) -> None:
    monkeypatch.setattr(
        "tollama.cli.main.list_registry_models",
        lambda: [SimpleNamespace(name="chronos2"), SimpleNamespace(name="timesfm-2.5-200m")],
    )
    monkeypatch.setattr(
        "tollama.cli.main.list_installed",
        lambda: [{"name": "local-model"}, {"name": 123}],
    )

    assert _complete_model_names("ch") == ["chronos2"]
    all_names = _complete_model_names("")
    assert all_names == ["chronos2", "local-model", "timesfm-2.5-200m"]


def test_resolve_progress_enabled_modes(monkeypatch) -> None:
    monkeypatch.setattr("tollama.cli.main.sys.stderr", SimpleNamespace(isatty=lambda: True))
    assert _resolve_progress_enabled("auto") is True
    assert _resolve_progress_enabled("on") is True
    assert _resolve_progress_enabled("off") is False

    monkeypatch.setattr("tollama.cli.main.sys.stderr", SimpleNamespace(isatty=lambda: False))
    assert _resolve_progress_enabled("auto") is False


def test_table_truncation_adds_ellipsis() -> None:
    text = _truncate_cell("x" * 60, max_width=20)
    assert text.endswith("...")
    rendered = _render_table(("VALUE",), [("x" * 80,)])
    assert "..." in rendered


def test_prompt_example_request_path_returns_selected_candidate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    candidate_one = tmp_path / "examples" / "request1.json"
    candidate_two = tmp_path / "examples" / "request2.json"
    candidate_one.parent.mkdir(parents=True, exist_ok=True)
    candidate_one.write_text("{}", encoding="utf-8")
    candidate_two.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "tollama.cli.main._candidate_request_paths",
        lambda _model: [candidate_one, candidate_two],
    )
    monkeypatch.setattr("tollama.cli.main.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("tollama.cli.main.typer.prompt", lambda *args, **kwargs: "2")

    selected = _prompt_example_request_path("mock")

    assert selected == candidate_two


def test_list_ps_show_and_rm_commands_call_api_client(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            self._base_url = base_url
            self._timeout = timeout

        def list_tags(self) -> dict[str, object]:
            return {"models": [{"name": "mock"}]}

        def list_running(self) -> dict[str, object]:
            return {"models": [{"name": "mock", "expires_at": None}]}

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name, "model": name}

        def remove_model(self, name: str) -> dict[str, object]:
            return {"deleted": True, "model": name}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert "NAME" in _result_stdout(listed)
    assert "mock" in _result_stdout(listed)

    listed_json = runner.invoke(app, ["list", "--json"])
    assert listed_json.exit_code == 0
    assert json.loads(_result_stdout(listed_json))["models"][0]["name"] == "mock"

    running = runner.invoke(app, ["ps"])
    assert running.exit_code == 0
    assert "NAME" in _result_stdout(running)
    assert "mock" in _result_stdout(running)

    running_json = runner.invoke(app, ["ps", "--json"])
    assert running_json.exit_code == 0
    assert json.loads(_result_stdout(running_json))["models"][0]["name"] == "mock"

    shown = runner.invoke(app, ["show", "mock"])
    assert shown.exit_code == 0
    assert json.loads(_result_stdout(shown))["name"] == "mock"

    removed = runner.invoke(app, ["rm", "mock"])
    assert removed.exit_code == 0
    assert json.loads(_result_stdout(removed)) == {"deleted": True, "model": "mock"}


def test_explain_command_renders_human_summary(monkeypatch) -> None:
    spec = ModelSpec.model_validate(
        {
            "name": "chronos2",
            "family": "torch",
            "source": {"type": "huggingface", "repo_id": "amazon/chronos-2", "revision": "main"},
            "license": {"type": "apache-2.0", "needs_acceptance": False},
            "metadata": {"max_horizon": 128, "max_context": 512},
            "capabilities": {
                "past_covariates_numeric": True,
                "past_covariates_categorical": True,
                "future_covariates_numeric": True,
                "future_covariates_categorical": True,
                "static_covariates": False,
            },
        },
    )
    monkeypatch.setattr("tollama.cli.main.get_model_spec", lambda _model: spec)
    monkeypatch.setattr(
        "tollama.cli.main.list_installed",
        lambda: [{"name": "chronos2", "license": {"accepted": True}}],
    )

    runner = _new_runner()
    result = runner.invoke(app, ["explain", "chronos2"])

    assert result.exit_code == 0
    output = _result_stdout(result)
    assert "chronos2" in output
    assert "family: torch" in output
    assert "max_horizon: 128" in output
    assert "accepted locally: yes" in output


def test_explain_command_json_output(monkeypatch) -> None:
    spec = ModelSpec.model_validate(
        {
            "name": "sundial-base-128m",
            "family": "sundial",
            "source": {
                "type": "huggingface",
                "repo_id": "thuml/sundial-base-128m",
                "revision": "main",
            },
            "license": {"type": "apache-2.0", "needs_acceptance": False},
            "metadata": {"max_horizon": 720, "max_context": 2880},
            "capabilities": {
                "past_covariates_numeric": False,
                "past_covariates_categorical": False,
                "future_covariates_numeric": False,
                "future_covariates_categorical": False,
                "static_covariates": False,
            },
        },
    )
    monkeypatch.setattr("tollama.cli.main.get_model_spec", lambda _model: spec)
    monkeypatch.setattr("tollama.cli.main.list_installed", lambda: [])

    runner = _new_runner()
    result = runner.invoke(app, ["explain", "sundial-base-128m", "--json"])

    assert result.exit_code == 0
    payload = json.loads(_result_stdout(result))
    assert payload["model"] == "sundial-base-128m"
    assert payload["limits"]["max_context"] == 2880
    assert payload["installed"] is False


def test_modelfile_commands_call_api_client(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def list_modelfiles(self) -> dict[str, object]:
            return {
                "modelfiles": [
                    {
                        "name": "baseline",
                        "path": "/tmp/baseline.yaml",
                        "profile": {"model": "mock", "horizon": 3},
                    }
                ]
            }

        def show_modelfile(self, name: str) -> dict[str, object]:
            return {"name": name, "profile": {"model": "mock", "horizon": 3}}

        def create_modelfile(
            self,
            name: str,
            *,
            profile: dict[str, object] | None = None,
            content: str | None = None,
        ) -> dict[str, object]:
            captured["create"] = {"name": name, "profile": profile, "content": content}
            return {"name": name, "path": "/tmp/baseline.yaml", "profile": {"model": "mock"}}

        def remove_modelfile(self, name: str) -> dict[str, object]:
            return {"deleted": True, "name": name}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()

    listed = runner.invoke(app, ["modelfile", "list"])
    assert listed.exit_code == 0
    assert "baseline" in _result_stdout(listed)

    shown = runner.invoke(app, ["modelfile", "show", "baseline"])
    assert shown.exit_code == 0
    assert json.loads(_result_stdout(shown))["name"] == "baseline"

    yaml_path = tmp_path / "baseline.yaml"
    yaml_path.write_text("model: mock\nhorizon: 4\n", encoding="utf-8")
    created = runner.invoke(app, ["modelfile", "create", "baseline", "--file", str(yaml_path)])
    assert created.exit_code == 0
    assert "baseline" in _result_stdout(created)
    assert captured["create"] == {
        "name": "baseline",
        "profile": None,
        "content": "model: mock\nhorizon: 4\n",
    }

    removed = runner.invoke(app, ["modelfile", "rm", "baseline"])
    assert removed.exit_code == 0
    assert json.loads(_result_stdout(removed)) == {"deleted": True, "name": "baseline"}


def test_quickstart_pulls_model_and_runs_demo_forecast(monkeypatch) -> None:
    captured: dict[str, object] = {"calls": []}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def health(self) -> dict[str, object]:
            cast_calls = captured["calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append("health")
            return {"health": {"status": "ok"}, "version": {"version": "0.1.0"}}

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            accept_license: bool = False,
        ) -> dict[str, object] | list[dict[str, object]]:
            cast_calls = captured["calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append("pull")
            captured["pull"] = {
                "name": name,
                "stream": stream,
                "accept_license": accept_license,
            }
            return {"status": "success", "model": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            cast_calls = captured["calls"]
            assert isinstance(cast_calls, list)
            cast_calls.append("forecast")
            captured["forecast"] = {"payload": payload, "stream": stream}
            return {
                "model": payload["model"],
                "forecasts": [
                    {
                        "id": "demo_series",
                        "freq": "D",
                        "start_timestamp": "2025-01-06",
                        "mean": [15.0, 16.0, 17.0],
                    }
                ],
            }

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()
    result = runner.invoke(app, ["quickstart"])

    assert result.exit_code == 0
    assert captured["base_url"] == "http://localhost:11435"
    assert captured["timeout"] == 30.0
    assert captured["calls"] == ["health", "pull", "forecast"]
    assert captured["pull"] == {"name": "mock", "stream": False, "accept_license": False}
    assert captured["forecast"]["stream"] is False
    assert captured["forecast"]["payload"]["model"] == "mock"
    assert captured["forecast"]["payload"]["horizon"] == 3

    output = _result_stdout(result)
    assert "tollama quickstart complete" in output
    assert "Next steps:" in output
    assert "tollama run mock --input examples/request.json --no-stream" in output


def test_quickstart_prints_daemon_guidance_when_unreachable(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def health(self) -> dict[str, object]:
            raise RuntimeError("connection refused")

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()
    result = runner.invoke(app, ["quickstart", "--base-url", "http://daemon.test"])

    assert result.exit_code == 1
    stderr = _result_stderr(result)
    assert "unable to reach tollama daemon at http://daemon.test" in stderr
    assert "tollama serve" in stderr


def test_run_auto_pulls_when_model_not_installed(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def show_model(self, name: str) -> dict[str, object]:
            raise DaemonHTTPError(action="show model", status_code=404, detail="missing")

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            accept_license: bool = False,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["pull"] = {
                "name": name,
                "stream": stream,
                "accept_license": accept_license,
            }
            return {"name": name, "family": "mock"}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["forecast"] = {"payload": payload, "stream": stream}
            return {"model": payload["model"], "forecasts": []}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()
    result = runner.invoke(
        app,
        ["run", "mock", "--input", str(request_path), "--horizon", "7", "--no-stream"],
    )

    assert result.exit_code == 0
    assert captured["timeout"] == _RUN_TIMEOUT_SECONDS
    assert captured["pull"] == {"name": "mock", "stream": False, "accept_license": False}
    assert captured["forecast"]["stream"] is False
    assert captured["forecast"]["payload"]["model"] == "mock"
    assert captured["forecast"]["payload"]["horizon"] == 7


def test_run_streaming_outputs_ndjson_lines(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            pass

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            assert stream is True
            return [
                {"status": "running forecast"},
                {"done": True, "response": {"model": payload["model"]}},
            ]

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path)])

    assert result.exit_code == 0
    lines = [line for line in _result_stdout(result).splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"status": "running forecast"}
    assert json.loads(lines[1]) == {"done": True, "response": {"model": "mock"}}


def test_run_auto_pull_accept_license_flag(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def show_model(self, name: str) -> dict[str, object]:
            raise DaemonHTTPError(action="show model", status_code=404, detail="missing")

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
            accept_license: bool = False,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["pull"] = {
                "name": name,
                "stream": stream,
                "accept_license": accept_license,
            }
            return {"name": name, "family": "uni2ts"}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["forecast"] = {"payload": payload, "stream": stream}
            return {"model": payload["model"], "forecasts": []}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "moirai-2.0-R-small",
            "--input",
            str(request_path),
            "--accept-license",
            "--no-stream",
        ],
    )

    assert result.exit_code == 0
    assert captured["pull"] == {
        "name": "moirai-2.0-R-small",
        "stream": False,
        "accept_license": True,
    }


def test_run_accepts_stdin_payload_when_input_omitted(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["forecast"] = {"payload": payload, "stream": stream}
            return {"model": payload["model"], "forecasts": []}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "mock", "--no-stream"],
        input=json.dumps(_sample_request_payload()),
    )

    assert result.exit_code == 0
    assert captured["forecast"]["stream"] is False
    assert captured["forecast"]["payload"]["model"] == "mock"
    assert captured["forecast"]["payload"]["horizon"] == 2


def test_run_uses_default_example_payload_when_input_omitted(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "chronos2_request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["forecast"] = {"payload": payload, "stream": stream}
            return {"model": payload["model"], "forecasts": []}

    monkeypatch.setattr("tollama.cli.main._load_request_payload_from_stdin", lambda: None)
    monkeypatch.setattr(
        "tollama.cli.main._resolve_default_request_path",
        lambda model: request_path,
    )
    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "chronos2", "--no-stream"])

    assert result.exit_code == 0
    assert captured["forecast"]["stream"] is False
    assert captured["forecast"]["payload"]["model"] == "chronos2"


def test_run_errors_when_payload_sources_are_missing(monkeypatch) -> None:
    monkeypatch.setattr("tollama.cli.main._load_request_payload_from_stdin", lambda: None)
    monkeypatch.setattr("tollama.cli.main._resolve_default_request_path", lambda model: None)

    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--no-stream"])

    assert result.exit_code != 0
    assert "missing forecast request payload" in _result_stdout(result)


def test_resolve_run_model_name_prompts_when_not_provided(monkeypatch) -> None:
    class _FakeClient:
        def list_tags(self) -> dict[str, object]:
            return {"models": [{"name": "mock"}, {"name": "chronos2"}]}

    monkeypatch.setattr("tollama.cli.main.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("tollama.cli.main.typer.prompt", lambda *args, **kwargs: "2")

    selected = _resolve_run_model_name(None, client=_FakeClient())  # type: ignore[arg-type]
    assert selected == "chronos2"


def test_run_errors_when_model_is_missing_in_non_interactive_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")
    monkeypatch.setattr(
        "tollama.cli.main.sys.stdin",
        SimpleNamespace(isatty=lambda: False, read=lambda: ""),
    )

    runner = _new_runner()
    result = runner.invoke(app, ["run", "--input", str(request_path), "--no-stream"])

    assert result.exit_code == 2
    assert "missing model name" in _result_stdout(result)


def test_run_rejects_non_json_input(tmp_path: Path) -> None:
    request_path = tmp_path / "broken.json"
    request_path.write_text("{not-json}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path)])
    assert result.exit_code != 0
    assert "input file is not valid JSON" in _result_stdout(result)


def test_run_help_mentions_input_and_stream_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Rich may inject ANSI style escapes into option tokens when color is forced,
    # so normalize help text before checking for stable flag names.
    normalized_stdout = re.sub(r"\x1b\[[0-9;]*m", "", _result_stdout(result))
    assert "--input" in normalized_stdout
    assert "--stream" in normalized_stdout
    assert "--no-stream" in normalized_stdout
    assert "stdin" in normalized_stdout


def test_run_warns_for_uni2ts_models_on_python_312_plus(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            del stream
            return {"model": payload["model"], "forecasts": []}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    monkeypatch.setattr("tollama.cli.main._is_python_312_or_newer", lambda: True)
    monkeypatch.setattr(
        "tollama.cli.main.get_model_spec",
        lambda _: SimpleNamespace(family="uni2ts"),
    )

    runner = _new_runner()
    result = runner.invoke(
        app,
        ["run", "moirai-2.0-R-small", "--input", str(request_path), "--no-stream"],
    )

    assert result.exit_code == 0
    assert (
        "warning: Uni2TS/Moirai dependencies may fail to install on Python 3.12+"
        in _result_stderr(result)
    )


@pytest.mark.parametrize(
    ("model", "implementation", "filename"),
    [
        ("granite-ttm-r2", "granite_ttm", "granite_ttm_request.json"),
        ("timesfm-2.5-200m", "timesfm_2p5_torch", "timesfm_2p5_request.json"),
        ("moirai-2.0-R-small", "moirai_2p0", "moirai_request.json"),
    ],
)
def test_resolve_default_request_path_uses_implementation_aliases(
    monkeypatch,
    tmp_path: Path,
    model: str,
    implementation: str,
    filename: str,
) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    expected = examples_dir / filename
    expected.write_text("{}", encoding="utf-8")
    (examples_dir / "request.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tollama.cli.main._project_root_from_module", lambda: None)
    monkeypatch.setattr(
        "tollama.cli.main.get_model_spec",
        lambda _: SimpleNamespace(metadata={"implementation": implementation}),
    )

    resolved = _resolve_default_request_path(model)

    assert resolved == expected


def test_resolve_default_request_path_prefers_specific_moirai_alias(
    monkeypatch,
    tmp_path: Path,
) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    preferred = examples_dir / "moirai_2p0_request.json"
    fallback = examples_dir / "moirai_request.json"
    preferred.write_text("{}", encoding="utf-8")
    fallback.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tollama.cli.main._project_root_from_module", lambda: None)
    monkeypatch.setattr(
        "tollama.cli.main.get_model_spec",
        lambda _: SimpleNamespace(metadata={"implementation": "moirai_2p0"}),
    )

    resolved = _resolve_default_request_path("moirai-2.0-R-small")
    assert resolved == preferred


def test_run_dry_run_exits_zero_when_validation_succeeds(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def validate_request(self, payload: dict[str, object]) -> dict[str, object]:
            captured["payload"] = payload
            return {"valid": True, "errors": [], "warnings": []}

        def show_model(self, name: str) -> dict[str, object]:
            raise AssertionError("show_model should not be called in --dry-run mode")

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path), "--dry-run"])

    assert result.exit_code == 0
    body = json.loads(_result_stdout(result))
    assert body["valid"] is True
    assert captured["payload"]["model"] == "mock"


def test_run_dry_run_exits_two_when_validation_fails(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def validate_request(self, payload: dict[str, object]) -> dict[str, object]:
            del payload
            return {"valid": False, "errors": ["field 'horizon': required"], "warnings": []}

        def show_model(self, name: str) -> dict[str, object]:
            raise AssertionError("show_model should not be called in --dry-run mode")

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path), "--dry-run"])

    assert result.exit_code == 2
    body = json.loads(_result_stdout(result))
    assert body["valid"] is False
    assert body["errors"]


def test_doctor_json_output_and_exit_code_zero(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "token")
    monkeypatch.setattr("tollama.cli.main.FAMILY_PYTHON_CONSTRAINTS", {})
    monkeypatch.setattr(
        "tollama.cli.main.list_runtime_statuses",
        lambda paths: [{"family": "torch", "installed": True}],
    )
    monkeypatch.setattr(
        "tollama.cli.main.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10, used=1, free=int(10 * 1024**3)),
    )

    class _FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def __enter__(self) -> _FakeHTTPClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            del exc_type, exc, tb
            return False

        def get(self, path: str) -> SimpleNamespace:
            assert path == "/v1/health"
            return SimpleNamespace(is_success=True, status_code=200)

    monkeypatch.setattr("tollama.cli.main.httpx.Client", _FakeHTTPClient)

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    body = json.loads(_result_stdout(result))
    assert {"checks", "summary"} <= set(body)
    assert body["summary"]["warn"] == 0
    assert body["summary"]["fail"] == 0


def test_doctor_warn_exit_code_one(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    monkeypatch.delenv("TOLLAMA_HF_TOKEN", raising=False)
    monkeypatch.setattr("tollama.cli.main.FAMILY_PYTHON_CONSTRAINTS", {})
    monkeypatch.setattr(
        "tollama.cli.main.list_runtime_statuses",
        lambda paths: [{"family": "torch", "installed": False}],
    )
    monkeypatch.setattr(
        "tollama.cli.main.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10, used=8, free=int(2 * 1024**3)),
    )

    class _FakeHTTPClient:
        def __init__(self, *, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def __enter__(self) -> _FakeHTTPClient:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            del exc_type, exc, tb
            return False

        def get(self, path: str) -> SimpleNamespace:
            assert path == "/v1/health"
            return SimpleNamespace(is_success=True, status_code=200)

    monkeypatch.setattr("tollama.cli.main.httpx.Client", _FakeHTTPClient)

    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Summary:" in _result_stdout(result)
    assert "warning" in _result_stdout(result)


def test_doctor_fail_exit_code_two(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "token")
    monkeypatch.setattr("tollama.cli.main.FAMILY_PYTHON_CONSTRAINTS", {})
    monkeypatch.setattr(
        "tollama.cli.main.list_runtime_statuses",
        lambda paths: [{"family": "torch", "installed": True}],
    )
    monkeypatch.setattr(
        "tollama.cli.main.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10, used=1, free=int(10 * 1024**3)),
    )

    def _raise_http_client(*, base_url: str, timeout: float) -> object:
        del base_url, timeout
        raise httpx.HTTPError("unreachable")

    monkeypatch.setattr("tollama.cli.main.httpx.Client", _raise_http_client)

    runner = _new_runner()
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 2
    assert "daemon: unreachable" in _result_stdout(result)


def test_run_warnings_are_emitted_to_stderr_with_color(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_sample_request_payload()), encoding="utf-8")

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            del base_url, timeout

        def show_model(self, name: str) -> dict[str, object]:
            return {"name": name}

        def forecast(
            self,
            payload: dict[str, object],
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            del stream
            return {
                "model": payload["model"],
                "forecasts": [],
                "warnings": ["watch out"],
            }

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = _new_runner()
    result = runner.invoke(
        app,
        ["run", "mock", "--input", str(request_path), "--no-stream"],
        color=True,
    )

    assert result.exit_code == 0
    assert "warning: watch out" in _result_stderr(result)
    assert "\x1b[" in _result_stderr(result)
