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

    def model_dir(self, name: str) -> Path:
        return self.models_dir / name

    def manifest_path(self, name: str) -> Path:
        return self.model_dir(name) / "manifest.json"


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

    resolved_paths = paths or TollamaPaths.default()
    manifest_path = resolved_paths.manifest_path(validated_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(spec, accepted=accept_license)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


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
    return {
        "name": spec.name,
        "family": spec.family,
        "source": spec.source.model_dump(mode="json"),
        "installed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "license": {
            "type": spec.license.type,
            "needs_acceptance": spec.license.needs_acceptance,
            "accepted": license_accepted,
        },
    }


def _validate_name(name: str) -> str:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"invalid model name: {name!r}")
    return name
