"""Tests for compare schemas, daemon endpoint, and tool wrappers."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tollama.client import InvalidRequestError, TollamaClient
from tollama.core.schemas import CompareRequest, CompareResponse
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.runner_manager import RunnerManager
from tollama.mcp.tools import MCPToolError, tollama_compare

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
if _HAS_LANGCHAIN:
    _HAS_LANGCHAIN = importlib.util.find_spec("langchain_core.tools") is not None


@pytest.fixture
def langchain_tools():
    if not _HAS_LANGCHAIN:
        pytest.skip("langchain_core.tools is not installed")
    return importlib.import_module("tollama.skill.langchain")


def _series_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02"],
            "target": [1.0, 2.0],
        }
    ]


def _compare_payload(*, models: list[str]) -> dict[str, Any]:
    return {
        "models": models,
        "horizon": 2,
        "series": _series_payload(),
        "options": {},
    }


def test_compare_request_requires_unique_models() -> None:
    with pytest.raises(ValueError, match="unique"):
        CompareRequest.model_validate(_compare_payload(models=["mock", "mock"]))


def test_daemon_compare_endpoint_returns_partial_success(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)
    install_from_registry("chronos2", accept_license=True, paths=paths)

    manager = RunnerManager(
        runner_commands={
            "mock": ("tollama-runner-mock",),
            "torch": ("tollama-runner-does-not-exist",),
        },
    )
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/api/compare", json=_compare_payload(models=["mock", "chronos2"]))

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {"requested_models": 2, "succeeded": 1, "failed": 1}

    results = {item["model"]: item for item in body["results"]}
    assert results["mock"]["ok"] is True
    assert results["mock"]["response"]["model"] == "mock"
    assert results["chronos2"]["ok"] is False
    assert results["chronos2"]["error"]["category"] == "RUNNER_UNAVAILABLE"
    assert results["chronos2"]["error"]["status_code"] == 503


def test_daemon_compare_endpoint_records_model_missing_error(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    manager = RunnerManager(runner_commands={"mock": ("tollama-runner-mock",)})
    with TestClient(create_app(runner_manager=manager)) as client:
        response = client.post("/api/compare", json=_compare_payload(models=["mock", "missing"]))

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {"requested_models": 2, "succeeded": 1, "failed": 1}
    results = {item["model"]: item for item in body["results"]}
    assert results["missing"]["ok"] is False
    assert results["missing"]["error"]["category"] == "MODEL_MISSING"
    assert results["missing"]["error"]["status_code"] == 404


def test_client_compare_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/compare"
        payload = request.read().decode("utf-8")
        assert "chronos2" in payload
        return httpx.Response(
            200,
            json={
                "models": ["chronos2", "timesfm-2.5-200m"],
                "horizon": 2,
                "results": [
                    {
                        "model": "chronos2",
                        "ok": True,
                        "response": {
                            "model": "chronos2",
                            "forecasts": [
                                {
                                    "id": "s1",
                                    "freq": "D",
                                    "start_timestamp": "2025-01-03",
                                    "mean": [2.0, 2.0],
                                }
                            ],
                        },
                    },
                    {
                        "model": "timesfm-2.5-200m",
                        "ok": False,
                        "error": {
                            "category": "RUNNER_UNAVAILABLE",
                            "status_code": 503,
                            "message": "runner unavailable",
                        },
                    },
                ],
                "summary": {"requested_models": 2, "succeeded": 1, "failed": 1},
            },
        )

    client = TollamaClient(
        base_url="http://daemon.test",
        timeout=3.0,
        transport=httpx.MockTransport(handler),
    )
    response = client.compare(_compare_payload(models=["chronos2", "timesfm-2.5-200m"]))

    assert isinstance(response, CompareResponse)
    assert response.summary.succeeded == 1
    assert response.results[0].ok is True
    assert response.results[1].ok is False


def test_client_compare_400_maps_to_invalid_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/compare"
        return httpx.Response(400, json={"detail": "invalid compare request"})

    client = TollamaClient(
        base_url="http://daemon.test",
        timeout=3.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        client.compare(_compare_payload(models=["mock", "chronos2"]))

    assert exc_info.value.exit_code == 2


def test_mcp_tollama_compare_invalid_request_maps_to_invalid_request() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_compare(request=_compare_payload(models=["mock"]))

    assert exc_info.value.category == "INVALID_REQUEST"
    assert exc_info.value.exit_code == 2


def test_mcp_tollama_compare_success(monkeypatch) -> None:
    expected = CompareResponse.model_validate(
        {
            "models": ["mock", "chronos2"],
            "horizon": 2,
            "results": [
                {
                    "model": "mock",
                    "ok": True,
                    "response": {
                        "model": "mock",
                        "forecasts": [
                            {
                                "id": "s1",
                                "freq": "D",
                                "start_timestamp": "2025-01-03",
                                "mean": [2.0, 2.0],
                            }
                        ],
                    },
                },
                {
                    "model": "chronos2",
                    "ok": False,
                    "error": {
                        "category": "RUNNER_UNAVAILABLE",
                        "status_code": 503,
                        "message": "runner unavailable",
                    },
                },
            ],
            "summary": {"requested_models": 2, "succeeded": 1, "failed": 1},
        },
    )

    class _FakeClient:
        def compare(self, _request: CompareRequest) -> CompareResponse:
            return expected

    monkeypatch.setattr("tollama.mcp.tools._make_client", lambda **_: _FakeClient())

    payload = tollama_compare(request=_compare_payload(models=["mock", "chronos2"]))

    assert payload["summary"] == {"requested_models": 2, "succeeded": 1, "failed": 1}
    assert payload["results"][0]["ok"] is True


def test_langchain_compare_tool_success(monkeypatch, langchain_tools) -> None:
    tool = langchain_tools.TollamaCompareTool()

    class _FakeClient:
        def compare(self, _request: CompareRequest) -> CompareResponse:
            return CompareResponse.model_validate(
                {
                    "models": ["mock", "chronos2"],
                    "horizon": 2,
                    "results": [
                        {
                            "model": "mock",
                            "ok": True,
                            "response": {
                                "model": "mock",
                                "forecasts": [
                                    {
                                        "id": "s1",
                                        "freq": "D",
                                        "start_timestamp": "2025-01-03",
                                        "mean": [2.0, 2.0],
                                    }
                                ],
                            },
                        },
                        {
                            "model": "chronos2",
                            "ok": False,
                            "error": {
                                "category": "RUNNER_UNAVAILABLE",
                                "status_code": 503,
                                "message": "runner unavailable",
                            },
                        },
                    ],
                    "summary": {"requested_models": 2, "succeeded": 1, "failed": 1},
                },
            )

    monkeypatch.setattr("tollama.skill.langchain._make_client", lambda **_: _FakeClient())
    payload = tool._run(request=_compare_payload(models=["mock", "chronos2"]))

    assert payload["summary"]["succeeded"] == 1
    assert payload["results"][1]["ok"] is False


def test_langchain_compare_tool_invalid_request(langchain_tools) -> None:
    tool = langchain_tools.TollamaCompareTool()
    payload = tool._run(request=_compare_payload(models=["mock"]))

    assert payload["error"]["category"] == "INVALID_REQUEST"
    assert payload["error"]["exit_code"] == 2
