"""Optional integration coverage for Granite TTM forecasting."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths
from tollama.daemon.app import create_app


@pytest.mark.integration
def test_granite_ttm_forecast_integration(monkeypatch, tmp_path) -> None:
    if os.environ.get("TOLLAMA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("set TOLLAMA_RUN_INTEGRATION_TESTS=1 to run integration tests")

    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    pytest.importorskip("tsfm_public")

    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    start = datetime(2025, 1, 1, tzinfo=UTC)
    payload = {
        "model": "granite-ttm-r2",
        "horizon": 5,
        "quantiles": [],
        "series": [
            {
                "id": "integration-series",
                "freq": "D",
                "timestamps": [
                    (start + timedelta(days=index)).date().isoformat()
                    for index in range(120)
                ],
                "target": [100.0 + float(index) for index in range(120)],
            }
        ],
        "options": {"device": "cpu"},
    }

    with TestClient(create_app()) as client:
        pull_response = client.post(
            "/api/pull",
            json={"model": "granite-ttm-r2", "stream": False},
        )
        assert pull_response.status_code == 200
        assert pull_response.json()["status"] == "success"

        forecast_response = client.post("/v1/forecast", json=payload)

    assert forecast_response.status_code == 200
    body = forecast_response.json()
    assert body["model"] == "granite-ttm-r2"
    assert len(body["forecasts"]) == 1
    assert len(body["forecasts"][0]["mean"]) == 5
