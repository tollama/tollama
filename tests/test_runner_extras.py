from __future__ import annotations

import tomllib
from pathlib import Path


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return pyproject["project"]["optional-dependencies"]


def test_runner_tide_extra_includes_lightning_import_dependency() -> None:
    runner_tide = _optional_dependencies()["runner_tide"]

    assert any(dep.startswith("pytorch-lightning") for dep in runner_tide)
