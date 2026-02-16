"""Tests for model registry loading and local model storage."""

from __future__ import annotations

import json

import pytest

from tollama.core.registry import load_registry
from tollama.core.storage import (
    TollamaPaths,
    install_from_registry,
    is_installed,
    list_installed,
    remove_model,
)


def test_registry_loads_required_model_specs() -> None:
    registry = load_registry()
    assert {"mock", "chronos2", "timesfm2p5", "moirai1p1-base", "granite-ttm-r2"} <= set(registry)

    mock = registry["mock"]
    assert mock.family == "mock"
    assert mock.source.repo_id == "tollama/mock-runner"
    assert mock.license.needs_acceptance is False


def test_install_list_and_remove_model_manifest_in_temp_store(tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")

    assert not is_installed("mock", paths=paths)

    manifest = install_from_registry("mock", accept_license=False, paths=paths)
    manifest_path = paths.manifest_path("mock")
    assert manifest_path.is_file()
    assert is_installed("mock", paths=paths)

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["name"] == "mock"
    assert manifest["family"] == "mock"
    assert manifest["source"]["repo_id"] == "tollama/mock-runner"
    assert manifest["license"]["accepted"] is True
    assert isinstance(manifest["installed_at"], str)

    installed = list_installed(paths=paths)
    assert [entry["name"] for entry in installed] == ["mock"]

    assert remove_model("mock", paths=paths) is True
    assert not is_installed("mock", paths=paths)
    assert remove_model("mock", paths=paths) is False


def test_install_requires_license_acceptance_when_flagged(tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")

    with pytest.raises(PermissionError):
        install_from_registry("timesfm2p5", accept_license=False, paths=paths)

    manifest = install_from_registry("timesfm2p5", accept_license=True, paths=paths)
    assert manifest["name"] == "timesfm2p5"
    assert manifest["license"]["needs_acceptance"] is True
    assert manifest["license"]["accepted"] is True
