"""Optional integration coverage for real Chronos forecasting."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app


@pytest.mark.integration
def test_chronos2_forecast_integration(monkeypatch, tmp_path) -> None:
    if os.environ.get("TOLLAMA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("set TOLLAMA_RUN_INTEGRATION_TESTS=1 to run integration tests")

    pytest.importorskip("chronos")
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")

    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("chronos2", accept_license=True, paths=paths)

    payload = {
        "model": "chronos2",
        "horizon": 2,
        "quantiles": [0.1, 0.5, 0.9],
        "series": [
            {
                "id": "integration-series",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                    "2025-01-08",
                    "2025-01-09",
                    "2025-01-10",
                ],
                "target": [10.0, 11.0, 12.0, 12.5, 13.0, 13.5, 14.0, 14.2, 14.5, 15.0],
            }
        ],
        "options": {},
    }

    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "chronos2"
    assert len(body["forecasts"]) == 1
    assert len(body["forecasts"][0]["mean"]) == 2
