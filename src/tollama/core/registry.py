"""Model registry loading and validation."""

from __future__ import annotations

import json
import os
from importlib.metadata import distribution
from pathlib import Path

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictStr,
)

NonEmptyStr = StrictStr

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "model-registry" / "registry.yaml"
PACKAGED_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_registry" / "registry.yaml"
MODEL_REGISTRY_ENV_VAR = "TOLLAMA_MODEL_REGISTRY_PATH"


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


class ModelCapabilities(BaseModel):
    """Model-family covariate compatibility metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    past_covariates_numeric: StrictBool = False
    past_covariates_categorical: StrictBool = False
    future_covariates_numeric: StrictBool = False
    future_covariates_categorical: StrictBool = False
    static_covariates: StrictBool = False


class ModelSpec(BaseModel):
    """Canonical model definition loaded from the registry."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonEmptyStr = Field(min_length=1)
    family: NonEmptyStr = Field(min_length=1)
    source: ModelSource
    license: ModelLicense
    defaults: dict[StrictStr, JsonValue] | None = None
    metadata: dict[StrictStr, JsonValue] | None = None
    capabilities: ModelCapabilities | None = None


class RegistryFile(BaseModel):
    """Top-level registry file schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    models: list[ModelSpec] = Field(min_length=1)


def resolve_default_registry_path() -> Path:
    """Resolve the default registry file for source and installed package layouts."""
    override = os.environ.get(MODEL_REGISTRY_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser()

    # Prefer the repository-local registry during source checkout so local edits are
    # reflected immediately. Installed wheels fall back to the packaged copy.
    for candidate in (DEFAULT_REGISTRY_PATH, PACKAGED_REGISTRY_PATH):
        if candidate.is_file():
            return candidate

    editable_registry = _editable_registry_path()
    if editable_registry is not None:
        return editable_registry

    return DEFAULT_REGISTRY_PATH


def _editable_registry_path() -> Path | None:
    """Resolve registry.yaml from editable-install metadata when package data moved."""
    try:
        direct_url_text = distribution("tollama").read_text("direct_url.json")
    except Exception:
        return None
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return None

    url = direct_url.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None

    candidate = Path(url[len("file://") :]) / "model-registry" / "registry.yaml"
    if candidate.is_file():
        return candidate
    return None


def load_registry(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Load and validate registry specs keyed by model name."""
    registry_path = Path(path) if path is not None else resolve_default_registry_path()
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
