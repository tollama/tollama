"""Contract tests for daemon/runner protocol and daemon HTTP health payloads."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tollama.core.protocol import ProtocolRequest, decode_request_line, encode_line
from tollama.daemon.app import HealthResponse, create_app
from tollama.daemon.openapi_artifact import canonicalize_openapi_schema


def test_protocol_request_round_trip_contract() -> None:
    request = ProtocolRequest(
        id="req-contract",
        method="forecast",
        params={"model": "mock", "horizon": 2},
    )

    decoded = decode_request_line(encode_line(request))

    assert decoded == request


def test_openapi_artifact_matches_application_schema() -> None:
    generated = canonicalize_openapi_schema(create_app().openapi())
    artifact_path = Path(__file__).resolve().parents[2] / "docs" / "openapi.json"
    checked_in = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert checked_in == generated


def test_health_endpoint_contract_uses_documented_schema() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    payload = HealthResponse.model_validate(response.json())
    assert payload.status == "ok"
