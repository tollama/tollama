"""Optional integration coverage for real TimesFM forecasting."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths
from tollama.daemon.app import create_app


@pytest.mark.integration
def test_timesfm_2p5_forecast_integration(monkeypatch, tmp_path) -> None:
    if os.environ.get("TOLLAMA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("set TOLLAMA_RUN_INTEGRATION_TESTS=1 to run integration tests")

    pytest.importorskip("timesfm")
    pytest.importorskip("numpy")
    pytest.importorskip("pandas")

    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    start = datetime(2025, 1, 1, tzinfo=UTC)
    payload = {
        "model": "timesfm-2.5-200m",
        "horizon": 8,
        "quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "integration-series-1",
                "freq": "D",
                "timestamps": [
                    (start + timedelta(days=index)).date().isoformat()
                    for index in range(200)
                ],
                "target": [100.0 + float(index) for index in range(200)],
            }
        ],
        "options": {"device": "cpu"},
    }

    with TestClient(create_app()) as client:
        pull_response = client.post(
            "/api/pull",
            json={"model": "timesfm-2.5-200m", "stream": False},
        )
        assert pull_response.status_code == 200
        assert pull_response.json()["status"] == "success"

        forecast_response = client.post("/v1/forecast", json=payload)

    assert forecast_response.status_code == 200
    body = forecast_response.json()
    assert body["model"] == "timesfm-2.5-200m"
    assert len(body["forecasts"]) == 1
    assert len(body["forecasts"][0]["mean"]) == 8
