"""Model registry loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictStr

NonEmptyStr = StrictStr

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "model-registry" / "registry.yaml"


class ModelSource(BaseModel):
    """Where a model can be resolved from."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: NonEmptyStr = Field(min_length=1)
    repo_id: NonEmptyStr = Field(min_length=1)
    revision: NonEmptyStr = Field(min_length=1)


class ModelLicense(BaseModel):
    """License metadata and acceptance requirements."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: NonEmptyStr = Field(
        min_length=1,
        validation_alias=AliasChoices("type", "id"),
        serialization_alias="type",
    )
    needs_acceptance: StrictBool
    notice: StrictStr | None = None


class ModelSpec(BaseModel):
    """Canonical model definition loaded from the registry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonEmptyStr = Field(min_length=1)
    family: NonEmptyStr = Field(min_length=1)
    source: ModelSource
    license: ModelLicense
    defaults: dict[StrictStr, JsonValue] | None = None
    metadata: dict[StrictStr, JsonValue] | None = None


class RegistryFile(BaseModel):
    """Top-level registry file schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    models: list[ModelSpec] = Field(min_length=1)


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Load and validate registry specs keyed by model name."""
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    raw = registry_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("registry file must contain a top-level object")

    registry = RegistryFile.model_validate(payload)
    by_name: dict[str, ModelSpec] = {}
    for spec in registry.models:
        if spec.name in by_name:
            raise ValueError(f"duplicate model name in registry: {spec.name!r}")
        by_name[spec.name] = spec

    return by_name


def list_registry_models(path: str | Path | None = None) -> list[ModelSpec]:
    """List all model specs sorted by name."""
    specs = load_registry(path)
    return sorted(specs.values(), key=lambda item: item.name)


def get_model_spec(name: str, path: str | Path | None = None) -> ModelSpec:
    """Fetch one model spec by name."""
    specs = load_registry(path)
    try:
        return specs[name]
    except KeyError as exc:
        raise KeyError(f"model {name!r} is not defined in the registry") from exc
