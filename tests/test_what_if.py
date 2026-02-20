"""Tests for what-if scenario transforms and daemon endpoint behavior."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tollama.core.scenarios import apply_scenario
from tollama.core.schemas import SeriesInput, WhatIfScenario
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


def _series_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": "s1",
            "freq": "D",
            "timestamps": ["2025-01-01", "2025-01-02"],
            "target": [1.0, 3.0],
            "past_covariates": {"temperature": [15.0, 16.0]},
            "future_covariates": {"temperature": [17.0, 18.0]},
        }
    ]


def _what_if_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "series": _series_payload(),
        "options": {},
        "scenarios": [
            {
                "name": "high_demand",
                "transforms": [
                    {"operation": "multiply", "field": "target", "value": 1.2},
                ],
            },
            {
                "name": "cold_weather",
                "transforms": [
                    {
                        "operation": "add",
                        "field": "future_covariates",
                        "key": "temperature",
                        "value": -5.0,
                    },
                ],
            },
        ],
    }


def test_apply_scenario_transforms_target_and_covariates() -> None:
    series = [SeriesInput.model_validate(_series_payload()[0])]
    scenario = WhatIfScenario.model_validate(
        {
            "name": "scenario_a",
            "transforms": [
                {"operation": "multiply", "field": "target", "value": 2.0},
                {
                    "operation": "add",
                    "field": "future_covariates",
                    "key": "temperature",
                    "value": -3.0,
                },
            ],
        },
    )

    updated = apply_scenario(series=series, scenario=scenario)

    assert updated[0].target == [2.0, 6.0]
    assert updated[0].future_covariates == {"temperature": [14.0, 15.0]}
    assert series[0].target == [1.0, 3.0]


def test_daemon_what_if_endpoint_returns_baseline_and_scenarios(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    with TestClient(create_app()) as client:
        response = client.post("/api/what-if", json=_what_if_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert body["horizon"] == 2
    assert body["baseline"]["model"] == "mock"
    assert body["summary"] == {"requested_scenarios": 2, "succeeded": 2, "failed": 0}
    assert body["results"][0]["scenario"] == "high_demand"
    assert body["results"][0]["ok"] is True
    assert body["results"][0]["response"]["forecasts"][0]["mean"] == pytest.approx([3.6, 3.6])


def test_daemon_what_if_endpoint_collects_invalid_scenario_errors(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=True, paths=paths)

    payload = _what_if_payload()
    payload["scenarios"].append(
        {
            "name": "invalid_covariate",
            "transforms": [
                {
                    "operation": "add",
                    "field": "future_covariates",
                    "key": "missing",
                    "value": 1.0,
                },
            ],
        },
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/what-if", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {"requested_scenarios": 3, "succeeded": 2, "failed": 1}
    failed = next(item for item in body["results"] if item["scenario"] == "invalid_covariate")
    assert failed["ok"] is False
    assert failed["error"]["category"] == "INVALID_SCENARIO"
    assert failed["error"]["status_code"] == 400
