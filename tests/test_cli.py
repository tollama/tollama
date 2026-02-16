"""Tests for the Typer-based tollama CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tollama.cli.main import app


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
        "port": 11435,
        "log_level": "info",
    }


def test_forecast_reads_file_overrides_model_and_prints_json(monkeypatch, tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "model": "original",
                "horizon": 2,
                "quantiles": [],
                "series": [
                    {
                        "id": "s1",
                        "freq": "D",
                        "timestamps": ["2025-01-01"],
                        "target": [3.0],
                    }
                ],
                "options": {},
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def forecast(self, payload: dict[str, object]) -> dict[str, object]:
            captured["payload"] = payload
            return {"model": payload["model"], "ok": True}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model",
            "mock",
            "--input",
            str(request_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["payload"]["model"] == "mock"
    output_json = json.loads(result.stdout)
    assert output_json == {"model": "mock", "ok": True}


def test_forecast_rejects_non_json_input(tmp_path: Path) -> None:
    request_path = tmp_path / "broken.json"
    request_path.write_text("{not-json}", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "forecast",
            "--model",
            "mock",
            "--input",
            str(request_path),
        ],
    )

    assert result.exit_code != 0
    assert "input file is not valid JSON" in result.stdout


def test_pull_list_and_rm_model_via_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            captured["base_url"] = base_url
            captured["timeout"] = timeout

        def pull_model(self, name: str, accept_license: bool) -> dict[str, object]:
            captured["pulled"] = {"name": name, "accept_license": accept_license}
            return {"name": name, "family": "mock"}

        def list_models(self) -> dict[str, object]:
            return {
                "installed": [{"name": "mock", "family": "mock", "installed": True}],
                "available": [],
            }

        def remove_model(self, name: str) -> dict[str, object]:
            captured["removed"] = name
            return {"removed": True, "name": name}

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)

    runner = CliRunner()

    pulled = runner.invoke(app, ["pull", "mock"])
    assert pulled.exit_code == 0
    pulled_payload = json.loads(pulled.stdout)
    assert pulled_payload["name"] == "mock"
    assert captured["pulled"] == {"name": "mock", "accept_license": False}

    listed = runner.invoke(app, ["list"])
    assert listed.exit_code == 0
    listed_payload = json.loads(listed.stdout)
    assert [item["name"] for item in listed_payload] == ["mock"]

    removed = runner.invoke(app, ["rm", "mock"])
    assert removed.exit_code == 0
    removed_payload = json.loads(removed.stdout)
    assert removed_payload == {"removed": True, "name": "mock"}
    assert captured["removed"] == "mock"


def test_pull_surfaces_daemon_error(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, base_url: str, timeout: float) -> None:
            pass

        def pull_model(self, name: str, accept_license: bool) -> dict[str, object]:
            raise RuntimeError("pull model 'chronos2' failed with HTTP 409: license required")

    monkeypatch.setattr("tollama.cli.main.TollamaClient", _FakeClient)
    runner = CliRunner()

    denied = runner.invoke(app, ["pull", "chronos2"])
    assert denied.exit_code != 0
    assert "HTTP 409" in denied.stdout
