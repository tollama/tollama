"""Tests for the Typer-based tollama CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tollama.cli.client import DaemonHTTPError
from tollama.cli.main import app


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


def test_serve_runs_uvicorn_with_expected_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int, log_level: str) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("tollama.cli.main.uvicorn.run", _fake_run)

    runner = CliRunner()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    assert captured == {
        "target": "tollama.daemon.app:app",
        "host": "127.0.0.1",
        "port": 11434,
        "log_level": "info",
    }


def test_pull_supports_streaming_and_non_stream(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def pull_model(
            self,
            name: str,
            *,
            stream: bool,
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["pull"] = {"name": name, "stream": stream}
            if stream:
                return [{"status": "pulling model manifest"}, {"done": True, "model": name}]
            return {"name": name, "family": "mock"}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()

    streamed = runner.invoke(app, ["pull", "mock"])
    assert streamed.exit_code == 0
    lines = [line for line in streamed.stdout.splitlines() if line.strip()]
    assert json.loads(lines[0]) == {"status": "pulling model manifest"}
    assert json.loads(lines[-1]) == {"done": True, "model": "mock"}
    assert captured["base_url"] == "http://localhost:11434"
    assert captured["pull"] == {"name": "mock", "stream": True}

    non_stream = runner.invoke(app, ["pull", "mock", "--no-stream"])
    assert non_stream.exit_code == 0
    assert json.loads(non_stream.stdout)["name"] == "mock"
    assert captured["pull"] == {"name": "mock", "stream": False}


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
    runner = CliRunner()

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    assert json.loads(listed.stdout)["models"][0]["name"] == "mock"

    running = runner.invoke(app, ["ps"])
    assert running.exit_code == 0
    assert json.loads(running.stdout)["models"][0]["name"] == "mock"

    shown = runner.invoke(app, ["show", "mock"])
    assert shown.exit_code == 0
    assert json.loads(shown.stdout)["name"] == "mock"

    removed = runner.invoke(app, ["rm", "mock"])
    assert removed.exit_code == 0
    assert json.loads(removed.stdout) == {"deleted": True, "model": "mock"}


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
        ) -> dict[str, object] | list[dict[str, object]]:
            captured["pull"] = {"name": name, "stream": stream}
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
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "mock", "--input", str(request_path), "--horizon", "7", "--no-stream"],
    )

    assert result.exit_code == 0
    assert captured["pull"] == {"name": "mock", "stream": False}
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
    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path)])

    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"status": "running forecast"}
    assert json.loads(lines[1]) == {"done": True, "response": {"model": "mock"}}


def test_run_rejects_non_json_input(tmp_path: Path) -> None:
    request_path = tmp_path / "broken.json"
    request_path.write_text("{not-json}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["run", "mock", "--input", str(request_path)])
    assert result.exit_code != 0
    assert "input file is not valid JSON" in result.stdout


def test_run_help_mentions_required_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.stdout
    assert "--no-stream" in result.stdout
