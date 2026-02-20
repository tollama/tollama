"""Local model installation state for tollama."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .registry import ModelSpec, get_model_spec

_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


@dataclass(frozen=True)
class TollamaPaths:
    """Filesystem layout for local tollama state."""

    base_dir: Path

    @classmethod
    def default(cls) -> TollamaPaths:
        override = os.environ.get("TOLLAMA_HOME")
        if override:
            return cls(base_dir=Path(override).expanduser())
        return cls(base_dir=Path.home() / ".tollama")

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def runtimes_dir(self) -> Path:
        return self.base_dir / "runtimes"

    @property
    def modelfiles_dir(self) -> Path:
        return self.base_dir / "modelfiles"

    def model_dir(self, name: str) -> Path:
        return self.models_dir / name

    def manifest_path(self, name: str) -> Path:
        return self.model_dir(name) / "manifest.json"

    def modelfile_path(self, name: str) -> Path:
        validated_name = _validate_name(name)
        return self.modelfiles_dir / f"{validated_name}.yaml"

    @property
    def config_path(self) -> Path:
        return self.base_dir / "config.json"


def is_installed(name: str, *, paths: TollamaPaths | None = None) -> bool:
    """Check whether a model manifest exists locally."""
    validated_name = _validate_name(name)
    resolved_paths = paths or TollamaPaths.default()
    return resolved_paths.manifest_path(validated_name).is_file()


def install_from_registry(
    name: str,
    accept_license: bool,
    *,
    paths: TollamaPaths | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Install a model by writing a local manifest from registry metadata."""
    validated_name = _validate_name(name)
    spec = get_model_spec(validated_name, path=registry_path)

    if spec.license.needs_acceptance and not accept_license:
        raise PermissionError(
            f"model {validated_name!r} requires license acceptance; pass accept_license=True",
        )

    manifest = _build_manifest(spec, accepted=accept_license)
    return write_manifest(validated_name, manifest, paths=paths)


def list_installed(*, paths: TollamaPaths | None = None) -> list[dict[str, Any]]:
    """List installed model manifests sorted by model name."""
    resolved_paths = paths or TollamaPaths.default()
    models_dir = resolved_paths.models_dir
    if not models_dir.exists():
        return []

    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(models_dir.glob("*/manifest.json")):
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid manifest content: {manifest_path}")
        manifests.append(data)

    manifests.sort(key=lambda item: str(item.get("name", "")))
    return manifests


def read_manifest(name: str, *, paths: TollamaPaths | None = None) -> dict[str, Any] | None:
    """Read one installed manifest by model name."""
    validated_name = _validate_name(name)
    resolved_paths = paths or TollamaPaths.default()
    manifest_path = resolved_paths.manifest_path(validated_name)
    if not manifest_path.exists():
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid manifest content: {manifest_path}")
    return payload


def write_manifest(
    name: str,
    manifest: dict[str, Any],
    *,
    paths: TollamaPaths | None = None,
) -> dict[str, Any]:
    """Atomically write one model manifest."""
    validated_name = _validate_name(name)
    resolved_paths = paths or TollamaPaths.default()
    manifest_path = resolved_paths.manifest_path(validated_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)
    return manifest


def remove_model(name: str, *, paths: TollamaPaths | None = None) -> bool:
    """Remove a locally installed model directory."""
    validated_name = _validate_name(name)
    resolved_paths = paths or TollamaPaths.default()
    model_dir = resolved_paths.model_dir(validated_name)
    if not model_dir.exists():
        return False

    shutil.rmtree(model_dir)
    return True


def _build_manifest(spec: ModelSpec, *, accepted: bool) -> dict[str, Any]:
    license_accepted = accepted or (not spec.license.needs_acceptance)
    accepted_at = datetime.now(UTC).isoformat().replace("+00:00", "Z") if license_accepted else None
    manifest: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "source": spec.source.model_dump(mode="json"),
        "resolved": {
            "commit_sha": None,
            "snapshot_path": None,
        },
        "size_bytes": 0,
        "pulled_at": None,
        "installed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "license": {
            "type": spec.license.type,
            "needs_acceptance": spec.license.needs_acceptance,
            "accepted": license_accepted,
            "accepted_at": accepted_at,
        },
    }
    if spec.license.notice is not None:
        manifest["license"]["notice"] = spec.license.notice
    if spec.metadata is not None:
        manifest["metadata"] = spec.metadata
    if spec.capabilities is not None:
        manifest["capabilities"] = spec.capabilities.model_dump(mode="json")
    return manifest


def _validate_name(name: str) -> str:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid model name: {name!r}")
    return name
