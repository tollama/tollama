"""Tests for synthetic series generation helpers and endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tollama.core.schemas import GenerateRequest
from tollama.core.synthetic import generate_synthetic_series
from tollama.daemon.app import create_app


def _generate_payload() -> dict[str, object]:
    return {
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                    "2025-01-07",
                ],
                "target": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0],
            }
        ],
        "count": 2,
        "length": 7,
        "seed": 42,
        "method": "statistical",
    }


def test_generate_synthetic_series_is_deterministic_with_seed() -> None:
    request = GenerateRequest.model_validate(_generate_payload())

    first = generate_synthetic_series(request)
    second = generate_synthetic_series(request)

    assert first == second
    assert first.method == "statistical"
    assert len(first.generated) == 2
    assert first.generated[0].source_id == "s1"
    assert len(first.generated[0].target) == 7
    assert len(first.generated[0].timestamps) == 7


def test_generate_endpoint_returns_synthetic_payload() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/generate", json=_generate_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "statistical"
    assert len(body["generated"]) == 2
    assert body["generated"][0]["source_id"] == "s1"
    assert len(body["generated"][0]["target"]) == 7


def test_generate_endpoint_rejects_too_short_source_series() -> None:
    payload = _generate_payload()
    payload["series"][0]["timestamps"] = ["2025-01-01", "2025-01-02"]
    payload["series"][0]["target"] = [1.0, 2.0]

    with TestClient(create_app()) as client:
        response = client.post("/api/generate", json=payload)

    assert response.status_code == 400
    assert "at least 3 points" in str(response.json()["detail"])
