"""Tests for stable OpenAPI artifact canonicalization."""

from __future__ import annotations

from tollama.daemon.openapi_artifact import canonicalize_openapi_schema


def test_canonicalize_openapi_schema_collapses_equivalent_io_pairs() -> None:
    schema = {
        "components": {
            "schemas": {
                "ForecastNarrative-Input": {
                    "properties": {
                        "series": {
                            "items": {"type": "string"},
                            "type": "array",
                        },
                    },
                    "type": "object",
                },
                "ForecastNarrative-Output": {
                    "properties": {
                        "series": {
                            "items": {"type": "string"},
                            "type": "array",
                        },
                    },
                    "type": "object",
                },
                "ForecastResponse-Input": {
                    "properties": {
                        "narrative": {
                            "$ref": "#/components/schemas/ForecastNarrative-Input",
                        },
                    },
                    "type": "object",
                },
                "ForecastResponse-Output": {
                    "properties": {
                        "narrative": {
                            "$ref": "#/components/schemas/ForecastNarrative-Output",
                        },
                    },
                    "type": "object",
                },
            },
        },
        "paths": {
            "/forecast": {
                "get": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ForecastResponse-Output",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    canonical = canonicalize_openapi_schema(schema)
    schemas = canonical["components"]["schemas"]

    assert "ForecastNarrative" in schemas
    assert "ForecastNarrative-Input" not in schemas
    assert "ForecastNarrative-Output" not in schemas
    assert "ForecastResponse" in schemas
    assert "ForecastResponse-Input" not in schemas
    assert "ForecastResponse-Output" not in schemas
    assert (
        canonical["paths"]["/forecast"]["get"]["responses"]["200"]["content"]["application/json"][
            "schema"
        ]["$ref"]
        == "#/components/schemas/ForecastResponse"
    )
    assert (
        schemas["ForecastResponse"]["properties"]["narrative"]["$ref"]
        == "#/components/schemas/ForecastNarrative"
    )


def test_canonicalize_openapi_schema_preserves_distinct_io_pairs() -> None:
    schema = {
        "components": {
            "schemas": {
                "Decision-Input": {
                    "properties": {
                        "kind": {"type": "string"},
                    },
                    "type": "object",
                },
                "Decision-Output": {
                    "properties": {
                        "kind": {"enum": ["ok", "skip"], "type": "string"},
                    },
                    "type": "object",
                },
            },
        },
    }

    canonical = canonicalize_openapi_schema(schema)
    schemas = canonical["components"]["schemas"]

    assert "Decision-Input" in schemas
    assert "Decision-Output" in schemas
    assert "Decision" not in schemas
