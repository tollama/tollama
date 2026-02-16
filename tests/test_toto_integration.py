"""Optional integration coverage for real Toto forecasting."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths
from tollama.daemon.app import create_app


@pytest.mark.integration
def test_toto_forecast_integration(monkeypatch, tmp_path) -> None:
    if os.environ.get("TOLLAMA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("set TOLLAMA_RUN_INTEGRATION_TESTS=1 to run integration tests")

    torch = pytest.importorskip("torch")
    pytest.importorskip("toto")
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")

    if not torch.cuda.is_available() and os.environ.get("TOLLAMA_TOTO_INTEGRATION_CPU") != "1":
        pytest.skip("set TOLLAMA_TOTO_INTEGRATION_CPU=1 to run Toto integration on CPU")

    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    start = datetime(2024, 1, 1, tzinfo=UTC)
    payload = {
        "model": "toto-open-base-1.0",
        "horizon": 32,
        "quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "integration-series-1",
                "freq": "D",
                "timestamps": [
                    (start + timedelta(days=index)).date().isoformat()
                    for index in range(512)
                ],
                "target": [100.0 + float(index) for index in range(512)],
            }
        ],
        "options": {"num_samples": 32, "samples_per_batch": 32},
    }

    with TestClient(create_app()) as client:
        pull_response = client.post(
            "/api/pull",
            json={"model": "toto-open-base-1.0", "stream": False},
        )
        assert pull_response.status_code == 200
        assert pull_response.json()["status"] == "success"

        forecast_response = client.post("/v1/forecast", json=payload)

    assert forecast_response.status_code == 200
    body = forecast_response.json()
    assert body["model"] == "toto-open-base-1.0"
    assert len(body["forecasts"]) == 1
    forecast = body["forecasts"][0]
    assert len(forecast["mean"]) == 32
    q10 = forecast["quantiles"]["0.1"]
    q50 = forecast["quantiles"]["0.5"]
    q90 = forecast["quantiles"]["0.9"]
    assert len(q10) == 32
    assert len(q50) == 32
    assert len(q90) == 32
    for p10, p50, p90 in zip(q10, q50, q90, strict=True):
        assert float(p10) <= float(p50) <= float(p90)
