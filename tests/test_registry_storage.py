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
    assert {
        "mock",
        "chronos2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
        "granite-ttm-r2",
    } <= set(registry)

    mock = registry["mock"]
    assert mock.family == "mock"
    assert mock.source.repo_id == "tollama/mock-runner"
    assert mock.license.needs_acceptance is False
    assert mock.capabilities is not None
    assert mock.capabilities.past_covariates_numeric is False

    granite = registry["granite-ttm-r2"]
    assert granite.family == "torch"
    assert granite.source.repo_id == "ibm-granite/granite-timeseries-ttm-r2"
    assert granite.source.revision == "90-30-ft-l1-r2.1"
    assert granite.capabilities is not None
    assert granite.capabilities.past_covariates_numeric is True
    assert granite.capabilities.past_covariates_categorical is False
    assert granite.metadata == {
        "implementation": "granite_ttm",
        "context_length": 90,
        "prediction_length": 30,
        "license": "apache-2.0",
    }

    timesfm = registry["timesfm-2.5-200m"]
    assert timesfm.family == "timesfm"
    assert timesfm.source.repo_id == "google/timesfm-2.5-200m-pytorch"
    assert timesfm.source.revision == "main"
    assert timesfm.license.type == "apache-2.0"
    assert timesfm.license.needs_acceptance is False
    assert timesfm.capabilities is not None
    assert timesfm.capabilities.future_covariates_numeric is True
    assert timesfm.capabilities.future_covariates_categorical is False
    assert timesfm.metadata == {
        "implementation": "timesfm_2p5_torch",
        "max_context": 1024,
        "max_horizon": 256,
        "use_quantiles_by_default": True,
    }

    moirai = registry["moirai-2.0-R-small"]
    assert moirai.family == "uni2ts"
    assert moirai.source.repo_id == "Salesforce/moirai-2.0-R-small"
    assert moirai.source.revision == "main"
    assert moirai.license.type == "cc-by-nc-4.0"
    assert moirai.license.needs_acceptance is True
    assert moirai.license.notice is not None
    assert moirai.metadata == {
        "implementation": "moirai_2p0",
        "default_context_length": 1680,
    }

    sundial = registry["sundial-base-128m"]
    assert sundial.family == "sundial"
    assert sundial.source.repo_id == "thuml/sundial-base-128m"
    assert sundial.source.revision == "main"
    assert sundial.license.type == "apache-2.0"
    assert sundial.license.needs_acceptance is False
    assert sundial.capabilities is not None
    assert sundial.capabilities.past_covariates_numeric is False
    assert sundial.metadata == {
        "implementation": "sundial_base",
        "max_context": 2880,
        "max_horizon": 720,
        "default_num_samples": 20,
    }

    toto = registry["toto-open-base-1.0"]
    assert toto.family == "toto"
    assert toto.source.repo_id == "Datadog/Toto-Open-Base-1.0"
    assert toto.source.revision == "main"
    assert toto.license.type == "apache-2.0"
    assert toto.license.needs_acceptance is False
    assert toto.capabilities is not None
    assert toto.capabilities.past_covariates_numeric is True
    assert toto.capabilities.future_covariates_numeric is False
    assert toto.metadata == {
        "implementation": "toto_open_base",
        "max_context": 4096,
        "max_horizon": 720,
        "default_num_samples": 256,
        "default_samples_per_batch": 256,
        "default_use_kv_cache": True,
    }


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
    assert isinstance(manifest["license"]["accepted_at"], str)
    assert manifest["resolved"] == {"commit_sha": None, "snapshot_path": None}
    assert manifest["size_bytes"] == 0
    assert manifest["pulled_at"] is None
    assert isinstance(manifest["installed_at"], str)
    assert manifest["capabilities"]["past_covariates_numeric"] is False

    installed = list_installed(paths=paths)
    assert [entry["name"] for entry in installed] == ["mock"]

    assert remove_model("mock", paths=paths) is True
    assert not is_installed("mock", paths=paths)
    assert remove_model("mock", paths=paths) is False


def test_install_requires_license_acceptance_when_flagged(tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")

    with pytest.raises(PermissionError):
        install_from_registry("moirai-2.0-R-small", accept_license=False, paths=paths)

    moirai_manifest = install_from_registry("moirai-2.0-R-small", accept_license=True, paths=paths)
    assert moirai_manifest["name"] == "moirai-2.0-R-small"
    assert moirai_manifest["license"]["needs_acceptance"] is True
    assert moirai_manifest["license"]["accepted"] is True
    assert isinstance(moirai_manifest["license"]["accepted_at"], str)
