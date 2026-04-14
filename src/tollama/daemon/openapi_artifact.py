"""Helpers for generating stable OpenAPI artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

_SCHEMA_REF_PREFIX = "#/components/schemas/"


def canonicalize_openapi_schema(value: Any) -> Any:
    """Return a deterministic representation for artifact export and diffing."""
    value = _collapse_equivalent_io_schemas(value)
    if isinstance(value, dict):
        return {key: canonicalize_openapi_schema(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        canonical_items = [canonicalize_openapi_schema(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return value


def _collapse_equivalent_io_schemas(value: Any) -> Any:
    """Collapse equivalent ``Foo-Input``/``Foo-Output`` schemas into ``Foo``."""
    if not isinstance(value, dict):
        return value

    components = value.get("components")
    if not isinstance(components, dict):
        return value

    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return value

    collapsed = deepcopy(value)
    alias_map: dict[str, str] = {}

    while True:
        schema_map = collapsed["components"]["schemas"]
        pair = _find_equivalent_io_pair(schema_map=schema_map, alias_map=alias_map)
        if pair is None:
            break

        input_name, output_name, base_name, canonical_schema = pair
        if base_name in schema_map:
            schema_map[base_name] = _rewrite_schema_refs(schema_map[base_name], alias_map)
        else:
            schema_map[base_name] = canonical_schema

        del schema_map[input_name]
        del schema_map[output_name]
        alias_map[f"{_SCHEMA_REF_PREFIX}{input_name}"] = f"{_SCHEMA_REF_PREFIX}{base_name}"
        alias_map[f"{_SCHEMA_REF_PREFIX}{output_name}"] = f"{_SCHEMA_REF_PREFIX}{base_name}"

    return _rewrite_schema_refs(collapsed, alias_map)


def _find_equivalent_io_pair(
    *,
    schema_map: dict[str, Any],
    alias_map: dict[str, str],
) -> tuple[str, str, str, Any] | None:
    for input_name in sorted(schema_map):
        if not input_name.endswith("-Input"):
            continue

        base_name = input_name.removesuffix("-Input")
        output_name = f"{base_name}-Output"
        if output_name not in schema_map:
            continue

        input_schema = _rewrite_schema_refs(schema_map[input_name], alias_map)
        output_schema = _rewrite_schema_refs(schema_map[output_name], alias_map)
        if canonicalize_openapi_schema(input_schema) != canonicalize_openapi_schema(output_schema):
            continue

        if base_name in schema_map:
            base_schema = _rewrite_schema_refs(schema_map[base_name], alias_map)
            if canonicalize_openapi_schema(base_schema) != canonicalize_openapi_schema(
                output_schema,
            ):
                continue

        return input_name, output_name, base_name, output_schema

    return None


def _rewrite_schema_refs(value: Any, alias_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, child in value.items():
            if key == "$ref" and isinstance(child, str):
                rewritten[key] = _resolve_ref_alias(child, alias_map)
            else:
                rewritten[key] = _rewrite_schema_refs(child, alias_map)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_schema_refs(item, alias_map) for item in value]
    return value


def _resolve_ref_alias(value: str, alias_map: dict[str, str]) -> str:
    while value in alias_map:
        value = alias_map[value]
    return value
