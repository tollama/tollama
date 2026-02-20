"""TSModelfile schema and local storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from .schemas import ForecastParameters
from .storage import TollamaPaths

NonEmptyStr = StrictStr
PositiveInt = StrictInt
Quantile = StrictFloat


class TSModelfile(BaseModel):
    """Declarative forecast profile used as request defaults."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: NonEmptyStr = Field(min_length=1)
    horizon: PositiveInt | None = Field(default=None, gt=0)
    quantiles: list[Quantile] = Field(default_factory=list)
    options: dict[NonEmptyStr, Any] = Field(default_factory=dict)
    parameters: ForecastParameters = Field(default_factory=ForecastParameters)
    covariate_mappings: dict[NonEmptyStr, NonEmptyStr] | None = None
    preprocessing: dict[NonEmptyStr, Any] | None = None
    enabled: StrictBool = True

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: list[Quantile]) -> list[Quantile]:
        if any(item <= 0.0 or item >= 1.0 for item in value):
            raise ValueError("quantiles values must be between 0 and 1")
        if value != sorted(value):
            raise ValueError("quantiles must be sorted in ascending order")
        if len(value) != len(set(value)):
            raise ValueError("quantiles must be unique")
        return value

    @classmethod
    def from_yaml(cls, content: str) -> TSModelfile:
        """Parse and validate YAML modelfile content."""
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            raise ValueError("modelfile YAML must contain a top-level object")
        return cls.model_validate(payload)

    def to_yaml(self) -> str:
        """Render normalized YAML content."""
        payload = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)


class StoredModelfile(BaseModel):
    """Stored modelfile metadata and resolved profile content."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonEmptyStr = Field(min_length=1)
    path: NonEmptyStr = Field(min_length=1)
    profile: TSModelfile


class ModelfileUpsertRequest(BaseModel):
    """Request shape for creating/updating one modelfile."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: NonEmptyStr = Field(min_length=1)
    profile: TSModelfile | None = None
    content: NonEmptyStr | None = None

    def resolved_profile(self) -> TSModelfile:
        """Return parsed profile from either typed profile or raw YAML content."""
        if self.profile is not None and self.content is not None:
            raise ValueError("provide either profile or content, not both")
        if self.profile is not None:
            return self.profile
        if self.content is not None:
            return TSModelfile.from_yaml(self.content)
        raise ValueError("profile or content is required")


class ModelfileListResponse(BaseModel):
    """List response payload for modelfile APIs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    modelfiles: list[StoredModelfile] = Field(default_factory=list)


def list_modelfiles(*, paths: TollamaPaths | None = None) -> list[StoredModelfile]:
    """List stored modelfiles sorted by name."""
    resolved_paths = paths or TollamaPaths.default()
    directory = resolved_paths.modelfiles_dir
    if not directory.exists():
        return []

    entries: list[StoredModelfile] = []
    for file_path in sorted(directory.glob("*.yaml")):
        name = file_path.stem
        profile = _load_profile_from_path(file_path)
        entries.append(
            StoredModelfile(
                name=name,
                path=str(file_path),
                profile=profile,
            )
        )

    entries.sort(key=lambda item: item.name)
    return entries


def load_modelfile(name: str, *, paths: TollamaPaths | None = None) -> StoredModelfile:
    """Load one stored modelfile by name."""
    resolved_paths = paths or TollamaPaths.default()
    file_path = resolved_paths.modelfile_path(name)
    if not file_path.exists():
        raise FileNotFoundError(f"modelfile {name!r} not found")

    profile = _load_profile_from_path(file_path)
    return StoredModelfile(name=name, path=str(file_path), profile=profile)


def write_modelfile(
    name: str,
    profile: TSModelfile,
    *,
    paths: TollamaPaths | None = None,
) -> StoredModelfile:
    """Atomically write one modelfile profile."""
    resolved_paths = paths or TollamaPaths.default()
    file_path = resolved_paths.modelfile_path(name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_content = profile.to_yaml()
    temporary = file_path.with_suffix(".tmp")
    temporary.write_text(normalized_content, encoding="utf-8")
    temporary.replace(file_path)

    return StoredModelfile(name=name, path=str(file_path), profile=profile)


def remove_modelfile(name: str, *, paths: TollamaPaths | None = None) -> bool:
    """Delete one modelfile by name."""
    resolved_paths = paths or TollamaPaths.default()
    file_path = resolved_paths.modelfile_path(name)
    if not file_path.exists():
        return False
    file_path.unlink()
    return True


def merge_modelfile_defaults(
    *,
    request_payload: dict[str, Any],
    profile: TSModelfile,
    request_fields_set: set[str],
) -> dict[str, Any]:
    """Apply deterministic precedence: request > modelfile > schema defaults."""
    merged = dict(request_payload)

    if "model" not in request_fields_set and not merged.get("model"):
        merged["model"] = profile.model

    if "horizon" not in request_fields_set and merged.get("horizon") is None:
        if profile.horizon is not None:
            merged["horizon"] = profile.horizon

    if "quantiles" not in request_fields_set and profile.quantiles:
        merged["quantiles"] = list(profile.quantiles)

    profile_options = profile.options or {}
    request_options = merged.get("options") or {}
    merged["options"] = _shallow_merge_maps(profile_options, request_options)

    merged["parameters"] = _merge_parameters(
        base=profile.parameters,
        override_payload=merged.get("parameters"),
        override_provided="parameters" in request_fields_set,
    )

    return merged


def _merge_parameters(
    *,
    base: ForecastParameters,
    override_payload: Any,
    override_provided: bool,
) -> ForecastParameters:
    if not override_provided:
        return base

    base_map = base.model_dump(mode="python", exclude_none=True)
    if isinstance(override_payload, ForecastParameters):
        override_map = override_payload.model_dump(
            mode="python",
            exclude_none=True,
            exclude_unset=True,
        )
    elif isinstance(override_payload, dict):
        override_map = dict(override_payload)
    else:
        override_map = {}

    merged = _deep_merge_maps(base_map, override_map)
    return ForecastParameters.model_validate(merged)


def _deep_merge_maps(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_maps(merged[key], value)
            continue
        merged[key] = value
    return merged


def _shallow_merge_maps(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged


def _load_profile_from_path(path: Path) -> TSModelfile:
    raw = path.read_text(encoding="utf-8")
    return TSModelfile.from_yaml(raw)
