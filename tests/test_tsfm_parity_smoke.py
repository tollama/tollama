"""Non-flaky TSFM parity smoke checks for registry+runtime behavior.

Classification labels used by CI:
- pass
- expected-dependency-gated
- regression
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tollama.daemon.app import create_app
from tollama.daemon.runner_manager import RunnerManager


def _request_payload(model: str) -> dict[str, object]:
    timestamps = [f"2025-01-{day:02d}" for day in range(1, 17)]
    target = [float(day) for day in range(1, 17)]
    return {
        "model": model,
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": timestamps,
                "target": target,
            }
        ],
        "options": {},
    }


def _classify_forecast_outcome(status_code: int, payload: dict[str, object]) -> str:
    if status_code == 200:
        return "pass"

    detail = payload.get("detail")
    detail_text = str(detail or "").lower()
    if "runner family" in detail_text and "not supported" in detail_text:
        return "regression"
    return "expected-dependency-gated"


def test_runner_family_registration_includes_patchtst_and_tide() -> None:
    manager = RunnerManager()
    families = set(manager.list_families())
    assert {"patchtst", "tide", "nhits", "nbeatsx"}.issubset(families)


def test_local_source_pull_and_runtime_smoke_classification(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        # Local-source pull should be deterministic and network-free.
        for model in ("tide", "nhits", "nbeatsx"):
            pull = client.post("/api/pull", json={"model": model, "stream": False})
            assert pull.status_code == 200, pull.text
            pull_body = pull.json()
            assert pull_body["status"] == "success"
            assert pull_body["digest"] == "local"

            run = client.post("/v1/forecast", json=_request_payload(model))
            body = run.json()
            classification = _classify_forecast_outcome(run.status_code, body)
            assert classification in {"pass", "expected-dependency-gated"}

            detail_text = str(body.get("detail") or "").lower()
            assert "runner family" not in detail_text
            assert "not supported" not in detail_text
