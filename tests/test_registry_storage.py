"""Tests for model registry loading and local model storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tollama.core import registry as registry_module
from tollama.core.registry import list_registry_models, load_registry
from tollama.core.storage import (
    TollamaPaths,
    install_from_registry,
    is_installed,
    list_installed,
    remove_model,
)


def test_load_registry_falls_back_to_packaged_copy_when_repo_registry_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    packaged_registry = tmp_path / "package" / "model_registry" / "registry.yaml"
    packaged_registry.parent.mkdir(parents=True, exist_ok=True)
    packaged_registry.write_text(
        "\n".join(
            [
                "models:",
                "  - name: mock",
                "    family: mock",
                "    source:",
                "      type: local",
                "      repo_id: tollama/mock-runner",
                "      revision: main",
                "    license:",
                "      type: mit",
                "      needs_acceptance: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(registry_module, "DEFAULT_REGISTRY_PATH", tmp_path / "missing-registry.yaml")
    monkeypatch.setattr(registry_module, "PACKAGED_REGISTRY_PATH", packaged_registry)

    registry = registry_module.load_registry()

    assert set(registry) == {"mock"}


def test_packaged_registry_copy_matches_repo_registry() -> None:
    assert registry_module.PACKAGED_REGISTRY_PATH.read_text(
        encoding="utf-8"
    ) == registry_module.DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")


def test_registry_loads_required_model_specs() -> None:
    registry = load_registry()
    listed_names = [spec.name for spec in list_registry_models()]
    assert "lag-llama" in listed_names
    assert {
        "mock",
        "chronos2",
        "timesfm-2.5-200m",
        "moirai-2.0-R-small",
        "sundial-base-128m",
        "toto-open-base-1.0",
        "lag-llama",
        "granite-ttm-r2",
        "patchtst",
        "tide",
        "nhits",
        "nbeatsx",
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
    assert granite.source.revision == "512-96-ft-l1-r2.1"
    assert granite.capabilities is not None
    assert granite.capabilities.past_covariates_numeric is True
    assert granite.capabilities.past_covariates_categorical is False
    assert granite.metadata == {
        "implementation": "granite_ttm",
        "context_length": 512,
        "prediction_length": 96,
        "license": "apache-2.0",
    }

    timesfm = registry["timesfm-2.5-200m"]
    assert timesfm.family == "timesfm"
    assert timesfm.source.repo_id == "google/timesfm-2.5-200m-pytorch"
    assert timesfm.source.revision == "main"
    assert timesfm.license.type == "apache-2.0"
    assert timesfm.license.needs_acceptance is False
    assert timesfm.capabilities is not None
    assert timesfm.capabilities.past_covariates_numeric is True
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
    assert moirai.capabilities is not None
    assert moirai.capabilities.past_covariates_numeric is True
    assert moirai.capabilities.future_covariates_numeric is True
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
        "min_vram_gb": 4.0,
        "default_num_samples": 100,
    }

    timer = registry["timer-base"]
    assert timer.family == "timer"
    assert timer.source.repo_id == "thuml/timer-base-84m"
    assert timer.source.revision == "main"
    assert timer.license.type == "apache-2.0"
    assert timer.license.needs_acceptance is False
    assert timer.capabilities is not None
    assert timer.capabilities.past_covariates_numeric is False
    assert timer.metadata == {
        "implementation": "timer_base",
        "max_context": 2880,
        "max_horizon": 720,
        "min_vram_gb": 4.0,
        "install_extra": "runner_timer",
        "install_command": 'python -m pip install -e ".[dev,runner_timer]"',
        "notes": (
            "Timer is a generative pre-trained Transformer for time series "
            "(ICML 2024). Strong zero-shot transfer across domains."
        ),
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
        "min_vram_gb": 8.0,
        "default_num_samples": 256,
        "default_samples_per_batch": 256,
        "default_use_kv_cache": True,
    }

    patchtst = registry["patchtst"]
    assert patchtst.family == "patchtst"
    assert patchtst.source.repo_id == "ibm-granite/granite-timeseries-patchtst"
    assert patchtst.source.revision == "main"
    assert patchtst.license.type == "apache-2.0"
    assert patchtst.license.needs_acceptance is False
    assert patchtst.capabilities is not None
    assert patchtst.capabilities.past_covariates_numeric is False
    assert patchtst.metadata == {
        "implementation": "patchtst",
        "status": "phase2_inference",
        "support_level": "baseline",
        "default_context_length": 512,
        "install_extra": "runner_patchtst",
        "install_command": 'python -m pip install -e ".[dev,runner_patchtst]"',
        "notes": (
            "PatchTST runner supports canonical point forecasts (single/multi series) "
            "with optional quantiles when the backend exposes them; covariates are "
            "currently ignored."
        ),
    }

    tide = registry["tide"]
    assert tide.family == "tide"
    assert tide.source.repo_id == "tollama/tide-runner"
    assert tide.source.revision == "main"
    assert tide.metadata == {
        "implementation": "tide",
        "status": "phase3_probabilistic",
        "support_level": "baseline",
        "install_extra": "runner_tide",
        "install_command": 'python -m pip install -e ".[dev,runner_tide]"',
        "notes": (
            "TiDE runner supports deterministic forecasts and best-effort requested "
            "quantiles when runtime probabilistic outputs are available. Covariates and "
            "static features are currently unsupported; when quantiles cannot be produced, "
            "responses explicitly fall back to mean-only forecasts with warnings."
        ),
    }
    assert tide.capabilities is not None
    assert tide.capabilities.past_covariates_numeric is False
    assert tide.capabilities.future_covariates_numeric is False
    assert tide.capabilities.static_covariates is False

    nhits = registry["nhits"]
    assert nhits.family == "nhits"
    assert nhits.source.repo_id == "tollama/nhits-runner"
    assert nhits.source.revision == "main"
    assert nhits.metadata == {
        "implementation": "nhits",
        "status": "phase2_inference",
        "support_level": "baseline",
        "install_extra": "runner_nhits",
        "install_command": 'python -m pip install -e ".[dev,runner_nhits]"',
        "notes": (
            "N-HiTS runner supports canonical point forecasts (single/multi series) "
            "via runtime NeuralForecast inference with numeric past/future/static "
            "exogenous inputs in practical best-effort mode. Pull is manifest-only "
            "(local source, no Hugging Face snapshot/auth required). Requested "
            "quantiles use backend outputs when available and otherwise fall back to "
            "calibrated residual-based estimates with explicit warnings."
        ),
    }
    assert nhits.capabilities is not None
    assert nhits.capabilities.past_covariates_numeric is True
    assert nhits.capabilities.future_covariates_numeric is True
    assert nhits.capabilities.static_covariates is True

    nbeatsx = registry["nbeatsx"]
    assert nbeatsx.family == "nbeatsx"
    assert nbeatsx.source.repo_id == "tollama/nbeatsx-runner"
    assert nbeatsx.source.revision == "main"
    assert nbeatsx.metadata == {
        "implementation": "nbeatsx",
        "status": "phase3_hardened",
        "support_level": "baseline",
        "install_extra": "runner_nbeatsx",
        "install_command": 'python -m pip install -e ".[dev,runner_nbeatsx]"',
        "notes": (
            "N-BEATSx runner supports canonical single/multi-series forecasts via runtime "
            "NeuralForecast inference with stricter input validation and numeric "
            "past/future/static exogenous inputs in practical best-effort mode. Pull is "
            "manifest-only (local source, no Hugging Face snapshot/auth required). "
            "Requested quantiles are best-effort (returned when backend probabilistic "
            "outputs are exposed, otherwise omitted with explicit warnings)."
        ),
    }
    assert nbeatsx.capabilities is not None
    assert nbeatsx.capabilities.past_covariates_numeric is True
    assert nbeatsx.capabilities.future_covariates_numeric is True
    assert nbeatsx.capabilities.static_covariates is True

    lag_llama = registry["lag-llama"]
    assert lag_llama.family == "lag_llama"
    assert lag_llama.source.repo_id == "time-series-foundation-models/Lag-Llama"
    assert lag_llama.source.revision == "main"
    assert lag_llama.license.type == "apache-2.0"
    assert lag_llama.license.needs_acceptance is False
    assert lag_llama.capabilities is not None
    assert lag_llama.capabilities.past_covariates_numeric is False
    assert lag_llama.capabilities.future_covariates_numeric is False
    assert lag_llama.capabilities.static_covariates is False
    assert lag_llama.metadata == {
        "implementation": "lag_llama",
        "default_context_length": 1024,
        "default_num_samples": 100,
        "default_batch_size": 32,
        "checkpoint_filename": "lag-llama.ckpt",
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
