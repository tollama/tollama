"""Helpers for benchmark-backed routing defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError

from .config import RoutingDefaults, TollamaConfig
from .storage import TollamaPaths

ROUTING_MANIFEST_ENV_NAME = "TOLLAMA_ROUTING_MANIFEST"
_REPO_DEFAULT_ROUTING_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "reports"
    / "cross_model_refresh_latest"
    / "routing.json"
)
_REPO_DEFAULT_BENCHMARK_RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "reports"
    / "cross_model_refresh_latest"
    / "result.json"
)


class RoutingManifest(BaseModel):
    """Persisted benchmark-backed routing policy."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: StrictInt = 1
    generated_at: StrictStr | None = None
    run_id: StrictStr | None = None
    eval_ref: StrictStr | None = None
    forecast_id: StrictStr | None = None
    source: StrictStr | None = None
    routing: RoutingDefaults = Field(default_factory=RoutingDefaults)
    policy: StrictStr | None = None
    caveats: list[StrictStr] = Field(default_factory=list)
    preprocessing_metadata: dict[str, Any] = Field(default_factory=dict)
    routing_rationale: dict[str, dict[str, Any]] = Field(default_factory=dict)


def get_routing_manifest_path(paths: TollamaPaths) -> Path:
    """Resolve routing manifest path with env override support."""
    override = os.environ.get(ROUTING_MANIFEST_ENV_NAME)
    if override:
        return Path(override).expanduser()
    return paths.routing_path


def load_routing_manifest(*, paths: TollamaPaths | None = None) -> RoutingManifest | None:
    """Load routing manifest from env/default paths when available."""
    resolved_paths = paths or TollamaPaths.default()
    candidates = [get_routing_manifest_path(resolved_paths)]
    if not os.environ.get(ROUTING_MANIFEST_ENV_NAME):
        candidates.append(_REPO_DEFAULT_ROUTING_PATH)
        candidates.append(_REPO_DEFAULT_BENCHMARK_RESULT_PATH)

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return load_routing_manifest_from_path(candidate)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    return None


def load_routing_manifest_from_path(path: str | Path) -> RoutingManifest:
    """Load and normalize a routing manifest or benchmark result from a specific path."""
    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise ValueError(f"routing manifest not found: {candidate}")

    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"unable to read routing manifest {candidate}: {exc}") from exc

    try:
        coerced = _coerce_manifest_payload(payload=payload, source_path=candidate)
        return RoutingManifest.model_validate(coerced)
    except ValidationError as exc:
        raise ValueError(f"invalid routing manifest in {candidate}: {exc}") from exc


def save_routing_manifest(
    manifest: RoutingManifest,
    *,
    paths: TollamaPaths | None = None,
    path: str | Path | None = None,
) -> Path:
    """Persist routing manifest atomically."""
    resolved_paths = paths or TollamaPaths.default()
    target = (
        Path(path).expanduser() if path is not None else get_routing_manifest_path(resolved_paths)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(target)
    return target


def resolve_effective_routing_defaults(
    *,
    config: TollamaConfig,
    paths: TollamaPaths | None = None,
) -> RoutingDefaults:
    """Resolve routing defaults with config taking precedence over benchmark policy."""
    manifest = load_routing_manifest(paths=paths)
    if manifest is None:
        return config.routing

    payload = config.routing.model_dump(mode="python")
    benchmark_payload = manifest.routing.model_dump(mode="python")
    for key, value in benchmark_payload.items():
        if payload.get(key) is None:
            payload[key] = value
    return RoutingDefaults.model_validate(payload)


def _coerce_manifest_payload(*, payload: object, source_path: Path) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("routing manifest must contain a top-level object")

    if "routing" in payload:
        return payload

    if "routing_recommendation" in payload:
        recommendation = payload.get("routing_recommendation")
        if not isinstance(recommendation, dict):
            raise ValueError("routing_recommendation must be an object")
        return {
            "version": 1,
            "generated_at": payload.get("generated_at"),
            "run_id": payload.get("run_id"),
            "eval_ref": payload.get("eval_ref") or payload.get("run_id"),
            "forecast_id": payload.get("forecast_id"),
            "source": f"benchmark_result:{source_path}",
            "routing": {
                "default": recommendation.get("default"),
                "fast_path": recommendation.get("fast_path"),
                "high_accuracy": recommendation.get("high_accuracy"),
            },
            "policy": recommendation.get("policy"),
            "caveats": recommendation.get("caveats", []),
            "preprocessing_metadata": payload.get("preprocessing_metadata", {}),
            "routing_rationale": (
                payload.get("routing_rationale") or recommendation.get("rationale", {})
            ),
        }

    if _looks_like_routing_defaults(payload):
        return {
            "version": 1,
            "source": f"legacy:{source_path}",
            "routing": payload,
        }

    raise ValueError("unsupported routing manifest format")


def _looks_like_routing_defaults(payload: dict[str, object]) -> bool:
    return any(key in payload for key in ("default", "fast_path", "high_accuracy"))
