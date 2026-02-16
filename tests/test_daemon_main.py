"""Tests for tollamad entrypoint configuration parsing."""

from __future__ import annotations

import tollama.daemon.main as daemon_main


def test_main_parses_tollama_host_host_port(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int, log_level: str) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("tollama.daemon.main.uvicorn.run", _fake_run)
    monkeypatch.setenv("TOLLAMA_HOST", "0.0.0.0:11435")
    monkeypatch.delenv("TOLLAMA_PORT", raising=False)

    assert daemon_main.main() == 0
    assert captured == {
        "target": "tollama.daemon.app:app",
        "host": "0.0.0.0",
        "port": 11435,
        "log_level": "info",
    }


def test_main_uses_default_bind_when_host_and_port_unset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int, log_level: str) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("tollama.daemon.main.uvicorn.run", _fake_run)
    monkeypatch.delenv("TOLLAMA_HOST", raising=False)
    monkeypatch.delenv("TOLLAMA_PORT", raising=False)

    assert daemon_main.main() == 0
    assert captured == {
        "target": "tollama.daemon.app:app",
        "host": "127.0.0.1",
        "port": 11435,
        "log_level": "info",
    }


def test_main_prefers_tollama_host_over_legacy_port(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(target: str, *, host: str, port: int, log_level: str) -> None:
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("tollama.daemon.main.uvicorn.run", _fake_run)
    monkeypatch.setenv("TOLLAMA_HOST", "0.0.0.0:11435")
    monkeypatch.setenv("TOLLAMA_PORT", "12500")

    assert daemon_main.main() == 0
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 11435


def test_main_raises_for_invalid_tollama_host(monkeypatch) -> None:
    monkeypatch.setenv("TOLLAMA_HOST", "127.0.0.1")
    monkeypatch.delenv("TOLLAMA_PORT", raising=False)

    try:
        daemon_main.main()
    except ValueError as exc:
        assert "invalid TOLLAMA_HOST" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid TOLLAMA_HOST")
