"""OpenAPI documentation coverage tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tollama.daemon.app import create_app

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
_IGNORE_SCHEMA_NAMES = {"HTTPValidationError", "ValidationError"}


def _openapi_schema(monkeypatch, tmp_path: Path) -> dict[str, object]:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    with TestClient(create_app()) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_covers_all_documented_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    documented_operations: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue
        for method in route.methods:
            normalized = method.lower()
            if normalized in _HTTP_METHODS:
                documented_operations.add((route.path, normalized))

    for path, method in sorted(documented_operations):
        assert path in schema["paths"], f"missing OpenAPI path: {path}"
        assert method in schema["paths"][path], (
            f"missing OpenAPI operation: {method.upper()} {path}"
        )


def test_openapi_operations_have_summary_description_and_tags(monkeypatch, tmp_path: Path) -> None:
    schema = _openapi_schema(monkeypatch, tmp_path)

    missing_summary: list[str] = []
    missing_description: list[str] = []
    missing_tags: list[str] = []
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if method not in _HTTP_METHODS:
                continue
            label = f"{method.upper()} {path}"
            if not operation.get("summary"):
                missing_summary.append(label)
            if not operation.get("description"):
                missing_description.append(label)
            if not operation.get("tags"):
                missing_tags.append(label)

    assert not missing_summary, f"missing summaries: {missing_summary}"
    assert not missing_description, f"missing descriptions: {missing_description}"
    assert not missing_tags, f"missing tags: {missing_tags}"


def test_openapi_component_properties_have_descriptions(monkeypatch, tmp_path: Path) -> None:
    schema = _openapi_schema(monkeypatch, tmp_path)

    missing: list[str] = []
    components = schema.get("components", {}).get("schemas", {})
    for schema_name, payload in components.items():
        if schema_name in _IGNORE_SCHEMA_NAMES:
            continue
        properties = payload.get("properties")
        if not isinstance(properties, dict):
            continue
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            if "description" not in prop_schema:
                missing.append(f"{schema_name}.{prop_name}")

    assert not missing, f"missing property descriptions: {missing}"
