"""HTTP API tests for the tollama daemon."""

from __future__ import annotations

import importlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tollama.core.config import TollamaConfig, save_config
from tollama.core.storage import TollamaPaths, install_from_registry
from tollama.daemon.app import create_app
from tollama.daemon.supervisor import RunnerCallError, RunnerUnavailableError

daemon_app_module = importlib.import_module("tollama.daemon.app")


@pytest.fixture(autouse=True)
def _clear_pull_environment(monkeypatch) -> None:
    for key in (
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "TOLLAMA_HF_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)


def _sample_forecast_payload() -> dict[str, Any]:
    return {
        "model": "mock",
        "horizon": 2,
        "quantiles": [0.1, 0.9],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 3.0],
            },
            {
                "id": "s2",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [2.0, 4.0],
            },
        ],
        "options": {},
    }


def _install_model(monkeypatch, tmp_path, name: str = "mock") -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry(name, accept_license=True, paths=paths)


def _patch_fake_hf_download(
    monkeypatch,
    *,
    captures: dict[str, Any] | None = None,
    progressive_files: bool = False,
) -> None:
    class _FakeSibling:
        def __init__(self, size: int) -> None:
            self.size = size

    class _FakeModelInfo:
        sha = "fake-commit-sha"
        siblings = [_FakeSibling(64), _FakeSibling(64)]

    def _fake_model_info(*, repo_id: str, revision: str, token: str | None) -> _FakeModelInfo:
        assert repo_id
        assert revision
        if captures is not None:
            captures["model_info"] = {
                "repo_id": repo_id,
                "revision": revision,
                "token": token,
            }
        return _FakeModelInfo()

    def _fake_snapshot_download(
        *,
        repo_id: str,
        revision: str,
        local_dir: str,
        token: str | None,
        max_workers: int,
        tqdm_class,
        local_files_only: bool,
    ) -> str:
        assert repo_id
        assert revision
        assert max_workers == 8
        if captures is not None:
            captures["snapshot_download"] = {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": local_dir,
                "token": token,
                "local_files_only": local_files_only,
            }

        pull_root = Path(local_dir)
        pull_root.mkdir(parents=True, exist_ok=True)

        progress = tqdm_class(
            total=100,
            initial=0,
            unit="B",
            desc="huggingface_hub.snapshot_download",
        )
        if progressive_files:
            (pull_root / "weights-1.bin").write_bytes(b"x" * 32)
            progress.update(10)
            (pull_root / "weights-2.bin").write_bytes(b"x" * 48)
            progress.update(40)
            (pull_root / "weights-3.bin").write_bytes(b"x" * 48)
            progress.update(50)
        else:
            (pull_root / "weights-1.bin").write_bytes(b"x" * 128)
            progress.update(10)
            progress.update(90)
        progress.set_description("Download complete")
        progress.close()

        (pull_root / ".cache").mkdir(parents=True, exist_ok=True)
        (pull_root / ".cache" / "ignored.bin").write_bytes(b"x" * 1024)
        return str(pull_root)

    monkeypatch.setattr("tollama.core.hf_pull._hf_model_info", _fake_model_info)
    monkeypatch.setattr("tollama.core.hf_pull._hf_snapshot_download", _fake_snapshot_download)


def _capturing_env_override(capture: dict[str, Any]):
    @contextmanager
    def _capture(mapping: dict[str, str | None]):
        capture["env_mapping"] = dict(mapping)
        yield

    return _capture


def test_health_endpoint_returns_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint_returns_string_version() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/version")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("version"), str)


def test_api_info_returns_redacted_diagnostics(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    monkeypatch.setenv("HTTP_PROXY", "http://alice:secret@proxy.internal:3128")
    monkeypatch.setenv("TOLLAMA_HF_TOKEN", "top-secret-token")

    config_path = paths.config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "pull": {
                    "offline": True,
                    "http_proxy": "http://bob:hunter2@proxy.config:8080",
                    "token": "config-token",
                },
            },
        ),
        encoding="utf-8",
    )

    manifest = {
        "name": "dummy",
        "family": "mock",
        "resolved": {"commit_sha": "local", "snapshot_path": None},
        "size_bytes": 7,
        "pulled_at": "2026-02-16T00:00:00Z",
        "installed_at": "2026-02-16T00:00:00Z",
    }
    manifest_path = paths.manifest_path("dummy")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runner_statuses = [
        {
            "family": "mock",
            "command": ["tollama-runner-mock"],
            "installed": True,
            "running": False,
            "pid": None,
            "started_at": None,
            "last_used_at": None,
            "restarts": 0,
            "last_error": None,
        },
        {
            "family": "torch",
            "command": ["tollama-runner-torch"],
            "installed": False,
            "running": False,
            "pid": None,
            "started_at": None,
            "last_used_at": None,
            "restarts": 0,
            "last_error": None,
        },
        {
            "family": "timesfm",
            "command": ["tollama-runner-timesfm"],
            "installed": False,
            "running": False,
            "pid": None,
            "started_at": None,
            "last_used_at": None,
            "restarts": 0,
            "last_error": None,
        },
        {
            "family": "uni2ts",
            "command": ["tollama-runner-uni2ts"],
            "installed": False,
            "running": False,
            "pid": None,
            "started_at": None,
            "last_used_at": None,
            "restarts": 0,
            "last_error": None,
        },
    ]

    app = create_app()
    monkeypatch.setattr(app.state.runner_manager, "get_all_statuses", lambda: runner_statuses)

    with TestClient(app) as client:
        response = client.get("/api/info")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "daemon",
        "paths",
        "config",
        "env",
        "pull_defaults",
        "models",
        "runners",
    } <= set(payload)

    assert payload["paths"]["config_exists"] is True
    assert payload["env"]["HTTP_PROXY"] == "http://***:***@proxy.internal:3128"
    assert payload["config"]["pull"]["http_proxy"] == "http://***:***@proxy.config:8080"
    assert payload["pull_defaults"]["http_proxy"]["value"] == "http://***:***@proxy.internal:3128"
    assert payload["env"]["TOLLAMA_HF_TOKEN_present"] is True
    assert "TOLLAMA_HF_TOKEN" not in payload["env"]
    assert payload["models"]["installed"][0]["name"] == "dummy"
    available = payload["models"]["available"]
    mock_available = next(item for item in available if item["name"] == "mock")
    assert mock_available["capabilities"]["past_covariates_numeric"] is False
    assert payload["runners"] == runner_statuses
    assert isinstance(payload["daemon"]["uptime_seconds"], int)

    serialized = json.dumps(payload, sort_keys=True)
    assert "top-secret-token" not in serialized
    assert "config-token" not in serialized


def test_ollama_tags_and_show_endpoints_from_installed_manifest(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))

    manifest = {
        "name": "mock",
        "family": "mock",
        "source": {
            "repo_id": "tollama/mock-runner",
            "revision": "main",
            "entrypoint": "tollama-runner-mock",
        },
        "installed_at": "2026-02-16T00:00:00Z",
        "license": {"type": "mit", "needs_acceptance": False, "accepted": True},
        "size": 1234,
        "digest": "sha256:abc123",
    }
    manifest_path = paths.manifest_path("mock")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with TestClient(create_app()) as client:
        tags_response = client.get("/api/tags")
        show_response = client.post("/api/show", json={"model": "mock"})

    assert tags_response.status_code == 200
    assert tags_response.json() == {
        "models": [
            {
                "name": "mock",
                "model": "mock",
                "modified_at": "2026-02-16T00:00:00Z",
                "size": 1234,
                "digest": "sha256:abc123",
                "details": {
                    "format": "tollama",
                    "family": "mock",
                    "families": ["mock"],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        ]
    }

    assert show_response.status_code == 200
    assert show_response.json() == {
        "name": "mock",
        "model": "mock",
        "family": "mock",
        "source": {
            "repo_id": "tollama/mock-runner",
            "revision": "main",
            "entrypoint": "tollama-runner-mock",
        },
        "resolved": {},
        "digest": "sha256:abc123",
        "size": 1234,
        "snapshot_path": None,
        "license": {"type": "mit", "needs_acceptance": False, "accepted": True},
        "modelfile": "",
        "parameters": "",
    }


def test_ollama_show_returns_404_for_missing_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        response = client.post("/api/show", json={"model": "not-installed"})

    assert response.status_code == 404
    assert "is not installed" in response.json()["detail"]


def test_ollama_pull_non_stream_and_delete_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        pulled = client.post("/api/pull", json={"model": "mock", "stream": False})
        assert pulled.status_code == 200
        assert pulled.json()["status"] == "success"
        assert pulled.json()["model"] == "mock"

        tags = client.get("/api/tags")
        assert tags.status_code == 200
        assert [item["name"] for item in tags.json()["models"]] == ["mock"]

        deleted = client.request("DELETE", "/api/delete", json={"model": "mock"})
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "model": "mock"}

        deleted_again = client.request("DELETE", "/api/delete", json={"model": "mock"})
        assert deleted_again.status_code == 404


def test_ollama_pull_stream_returns_ndjson(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "mock"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2
    payloads = [json.loads(line) for line in lines]
    assert payloads[0].get("status") == "pulling manifest"
    assert payloads[-1] == {
        "status": "success",
        "model": "mock",
        "digest": "local",
        "size": 0,
    }


def test_ollama_pull_stream_downloads_hf_snapshot_with_progress(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    _patch_fake_hf_download(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "chronos2"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    assert payloads[0]["status"] == "pulling manifest"
    downloading_events = [entry for entry in payloads if entry.get("status") == "downloading"]
    assert downloading_events
    for event in downloading_events:
        assert {"completed_bytes", "total_bytes", "files_completed", "files_total"} <= set(event)
    assert payloads[-1]["status"] == "success"
    assert payloads[-1]["digest"] == "fake-commit-sha"
    assert payloads[-1]["size"] == 128

    manifest = json.loads(paths.manifest_path("chronos2").read_text(encoding="utf-8"))
    assert manifest["resolved"]["commit_sha"] == "fake-commit-sha"
    assert isinstance(manifest["resolved"]["snapshot_path"], str)
    assert manifest["size_bytes"] == 128
    assert isinstance(manifest["pulled_at"], str)

    with TestClient(create_app()) as client:
        tags_response = client.get("/api/tags")
        show_response = client.post("/api/show", json={"model": "chronos2"})

    assert tags_response.status_code == 200
    chronos_tag = next(
        item for item in tags_response.json()["models"] if item["name"] == "chronos2"
    )
    assert chronos_tag["digest"] == "fake-commit-sha"
    assert chronos_tag["size"] == 128

    assert show_response.status_code == 200
    show_body = show_response.json()
    assert show_body["digest"] == "fake-commit-sha"
    assert show_body["size"] == 128
    assert show_body["snapshot_path"] == manifest["resolved"]["snapshot_path"]


def test_ollama_pull_non_stream_returns_success_and_updates_manifest(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    _patch_fake_hf_download(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "chronos2", "stream": False})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "success",
        "model": "chronos2",
        "digest": "fake-commit-sha",
        "size": 128,
    }

    manifest = json.loads(paths.manifest_path("chronos2").read_text(encoding="utf-8"))
    assert manifest["resolved"]["commit_sha"] == "fake-commit-sha"
    assert manifest["size_bytes"] == 128


def test_ollama_pull_granite_uses_registry_repo_and_revision(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "granite-ttm-r2", "stream": False})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "granite-ttm-r2"
    assert captures["model_info"] == {
        "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": "90-30-ft-l1-r2.1",
        "token": None,
    }

    manifest = json.loads(paths.manifest_path("granite-ttm-r2").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "type": "huggingface",
        "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": "90-30-ft-l1-r2.1",
    }
    assert manifest["metadata"] == {
        "implementation": "granite_ttm",
        "context_length": 90,
        "prediction_length": 30,
        "license": "apache-2.0",
    }
    assert manifest["resolved"]["commit_sha"] == "fake-commit-sha"
    assert isinstance(manifest["resolved"]["snapshot_path"], str)
    assert isinstance(manifest["pulled_at"], str)
    assert manifest["size_bytes"] == 128


def test_ollama_pull_timesfm_uses_registry_repo_and_revision(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "timesfm-2.5-200m", "stream": False})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "timesfm-2.5-200m"
    assert captures["model_info"] == {
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "revision": "main",
        "token": None,
    }

    manifest = json.loads(paths.manifest_path("timesfm-2.5-200m").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "type": "huggingface",
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "revision": "main",
    }
    assert manifest["metadata"] == {
        "implementation": "timesfm_2p5_torch",
        "max_context": 1024,
        "max_horizon": 256,
        "use_quantiles_by_default": True,
    }
    assert manifest["resolved"]["commit_sha"] == "fake-commit-sha"
    assert isinstance(manifest["resolved"]["snapshot_path"], str)
    assert isinstance(manifest["pulled_at"], str)
    assert manifest["size_bytes"] == 128


def test_ollama_pull_moirai_requires_license_acceptance(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "moirai-1.1-R-base", "stream": False})

    assert response.status_code == 409
    assert "requires license acceptance" in response.json()["detail"]


def test_ollama_pull_moirai_with_accept_license_records_acceptance(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
            json={
                "model": "moirai-1.1-R-base",
                "stream": False,
                "accept_license": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["model"] == "moirai-1.1-R-base"
    assert captures["model_info"] == {
        "repo_id": "Salesforce/moirai-1.1-R-base",
        "revision": "main",
        "token": None,
    }

    manifest = json.loads(paths.manifest_path("moirai-1.1-R-base").read_text(encoding="utf-8"))
    assert manifest["license"]["type"] == "cc-by-nc-4.0"
    assert manifest["license"]["needs_acceptance"] is True
    assert manifest["license"]["accepted"] is True
    assert isinstance(manifest["license"]["accepted_at"], str)
    assert "Non-commercial" in manifest["license"]["notice"]
    assert isinstance(manifest["resolved"]["snapshot_path"], str)
    assert manifest["resolved"]["commit_sha"] == "fake-commit-sha"


def test_ollama_pull_stream_reports_file_count_progress(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    monkeypatch.setattr("tollama.core.hf_pull._THROTTLE_BYTES", 1)
    monkeypatch.setattr("tollama.core.hf_pull._FILE_SCAN_INTERVAL_SECONDS", 0.0)
    _patch_fake_hf_download(monkeypatch, progressive_files=True)

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "chronos2"})

    assert response.status_code == 200
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    downloading_events = [entry for entry in payloads if entry.get("status") == "downloading"]
    assert len(downloading_events) >= 2

    files_completed = [int(entry["files_completed"]) for entry in downloading_events]
    assert files_completed == sorted(files_completed)
    assert files_completed[-1] >= 3
    assert payloads[-1]["status"] == "success"


def test_ollama_pull_options_apply_overrides_and_token(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)
    monkeypatch.setattr(daemon_app_module, "set_env_temporarily", _capturing_env_override(captures))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
                json={
                    "model": "chronos2",
                    "stream": False,
                    "http_proxy": "http://proxy.internal:3128",
                    "https_proxy": "http://proxy.internal:3129",
                    "no_proxy": "localhost,127.0.0.1",
                    "hf_home": "/tmp/hf-cache",
                    "token": "secret-token",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert captures["env_mapping"] == {
        "HF_HOME": "/tmp/hf-cache",
        "HTTP_PROXY": "http://proxy.internal:3128",
        "HTTPS_PROXY": "http://proxy.internal:3129",
        "NO_PROXY": "localhost,127.0.0.1",
    }
    assert captures["model_info"]["token"] == "secret-token"
    assert captures["snapshot_download"]["token"] == "secret-token"
    assert captures["snapshot_download"]["local_files_only"] is False


def test_ollama_pull_uses_config_defaults_when_request_omits_options(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)
    monkeypatch.setattr(daemon_app_module, "set_env_temporarily", _capturing_env_override(captures))
    save_config(
        paths,
        TollamaConfig.model_validate(
            {
                "pull": {
                    "offline": True,
                    "https_proxy": "http://proxy.internal:3129",
                }
            },
        ),
    )

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
            json={"model": "chronos2", "stream": False},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert captures["env_mapping"]["HF_HUB_OFFLINE"] == "1"
    assert captures["env_mapping"]["HTTPS_PROXY"] == "http://proxy.internal:3129"
    assert captures["snapshot_download"]["local_files_only"] is True
    assert captures["snapshot_download"]["token"] is None


def test_ollama_pull_explicit_request_overrides_config_defaults(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)
    monkeypatch.setattr(daemon_app_module, "set_env_temporarily", _capturing_env_override(captures))
    save_config(paths, TollamaConfig.model_validate({"pull": {"offline": True}}))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
            json={"model": "chronos2", "stream": False, "offline": False},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert captures["env_mapping"] == {"HF_HUB_OFFLINE": None}
    assert captures["snapshot_download"]["local_files_only"] is False


def test_ollama_pull_local_files_only_without_offline_avoids_hf_hub_offline(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)
    monkeypatch.setattr(daemon_app_module, "set_env_temporarily", _capturing_env_override(captures))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
            json={
                "model": "chronos2",
                "stream": False,
                "local_files_only": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert captures["env_mapping"] == {}
    assert captures["snapshot_download"]["local_files_only"] is True
    assert "model_info" not in captures


def test_ollama_pull_offline_sets_hf_offline_and_forces_local_only(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    captures: dict[str, Any] = {}
    _patch_fake_hf_download(monkeypatch, captures=captures)
    monkeypatch.setattr(daemon_app_module, "set_env_temporarily", _capturing_env_override(captures))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/pull",
            json={
                "model": "chronos2",
                "stream": False,
                "offline": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert captures["env_mapping"] == {"HF_HUB_OFFLINE": "1"}
    assert "model_info" not in captures
    assert captures["snapshot_download"]["local_files_only"] is True


def test_ollama_pull_offline_returns_already_present_when_snapshot_exists(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    _patch_fake_hf_download(monkeypatch)

    with TestClient(create_app()) as client:
        first_pull = client.post("/api/pull", json={"model": "chronos2", "stream": False})
    assert first_pull.status_code == 200

    def _should_not_download(**kwargs: Any) -> str:
        raise AssertionError(f"unexpected download attempt in offline mode: {kwargs}")

    monkeypatch.setattr("tollama.core.hf_pull._hf_snapshot_download", _should_not_download)

    with TestClient(create_app()) as client:
        offline_pull = client.post("/api/pull", json={"model": "chronos2", "offline": True})

    assert offline_pull.status_code == 200
    payloads = [json.loads(line) for line in offline_pull.text.splitlines() if line.strip()]
    assert any(entry.get("status") == "already present" for entry in payloads)
    assert payloads[-1]["status"] == "success"
    assert payloads[-1]["model"] == "chronos2"


def test_ollama_pull_stream_emits_warning_when_insecure(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    _patch_fake_hf_download(monkeypatch)
    monkeypatch.setattr(
        "tollama.core.hf_pull._hf_client_tools",
        lambda: type(
            "_Tools",
            (),
            {
                "backend": "httpx",
                "request_hook": None,
                "default_factory": lambda: None,
                "set_factory": staticmethod(lambda _factory: None),
            },
        )(),
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "chronos2", "insecure": True})

    assert response.status_code == 200
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    warnings = [entry for entry in payloads if entry.get("status") == "warning"]
    assert warnings
    assert "SSL verification disabled" in warnings[0]["message"]


def test_ollama_pull_offline_missing_local_model_returns_friendly_error(
    monkeypatch,
    tmp_path,
) -> None:
    class LocalEntryNotFoundError(RuntimeError):
        pass

    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    monkeypatch.setattr(
        "tollama.core.hf_pull._hf_snapshot_download",
        lambda **kwargs: (_ for _ in ()).throw(
            LocalEntryNotFoundError("not found in local cache"),
        ),
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/pull", json={"model": "chronos2", "offline": True})

    assert response.status_code == 200
    payloads = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert "error" in payloads[-1]
    assert "without --offline or --local-files-only" in payloads[-1]["error"]["message"]


def test_ollama_pull_prevalidation_errors_are_raised_before_streaming(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        missing = client.post("/api/pull", json={"model": "does-not-exist"})
        assert missing.status_code == 404

        conflict = client.post("/api/pull", json={"model": "timesfm2p5"})
        assert conflict.status_code == 409


def test_api_ps_omits_model_after_forecast_keep_alive_zero(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["keep_alive"] = 0

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    assert ps_response.json() == {"models": []}


def test_api_ps_tracks_model_after_forecast_keep_alive_negative(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["keep_alive"] = -1

    with TestClient(create_app()) as client:
        forecast_response = client.post("/v1/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    models = ps_response.json().get("models")
    assert isinstance(models, list)
    assert len(models) == 1

    loaded = models[0]
    assert loaded["name"] == "mock"
    assert loaded["model"] == "mock"
    assert loaded["expires_at"] is None
    assert loaded["size"] == 0
    assert loaded["size_vram"] == 0
    assert loaded["context_length"] == 0
    assert loaded["details"] == {"family": "mock"}


def test_api_forecast_stream_false_returns_forecast_response(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["stream"] = False

    with TestClient(create_app()) as client:
        response = client.post("/api/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert isinstance(body["forecasts"], list)
    assert len(body["forecasts"]) == 2


def test_api_forecast_stream_true_returns_ndjson_with_done(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    with TestClient(create_app()) as client:
        response = client.post("/api/forecast", json=_sample_forecast_payload())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    lines = [line for line in response.text.splitlines() if line.strip()]
    assert len(lines) >= 2

    payloads = [json.loads(line) for line in lines]
    assert payloads[0] == {"status": "loading model"}
    assert payloads[1] == {"status": "running forecast"}
    assert payloads[-1].get("done") is True
    assert payloads[-1]["response"]["model"] == "mock"


def test_api_forecast_keep_alive_updates_loaded_model_tracker(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["stream"] = False
    payload["keep_alive"] = -1

    with TestClient(create_app()) as client:
        forecast_response = client.post("/api/forecast", json=payload)
        ps_response = client.get("/api/ps")

    assert forecast_response.status_code == 200
    assert ps_response.status_code == 200
    models = ps_response.json().get("models")
    assert isinstance(models, list)
    assert len(models) == 1
    assert models[0]["model"] == "mock"
    assert models[0]["expires_at"] is None


def test_forecast_routes_end_to_end_to_mock_runner(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock"
    assert len(body["forecasts"]) == 2

    first = body["forecasts"][0]
    assert first["id"] == "s1"
    assert first["mean"] == [3.0, 3.0]
    assert first["quantiles"] == {"0.1": [3.0, 3.0], "0.9": [3.0, 3.0]}
    assert first["start_timestamp"] == "2025-01-02"

    second = body["forecasts"][1]
    assert second["id"] == "s2"
    assert second["mean"] == [4.0, 4.0]
    assert second["quantiles"] == {"0.1": [4.0, 4.0], "0.9": [4.0, 4.0]}
    assert second["start_timestamp"] == "2025-01-02"


def test_forecast_invalid_payload_returns_400(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["horizon"] = "2"

    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_forecast_rejects_mismatched_future_covariate_length(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["series"][0]["past_covariates"] = {"promo": [1.0, 2.0]}
    payload["series"][0]["future_covariates"] = {"promo": [3.0]}

    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 400
    detail = json.dumps(response.json())
    assert "future_covariates" in detail
    assert "horizon" in detail


def test_forecast_rejects_mixed_covariate_types(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    payload = _sample_forecast_payload()
    payload["series"][0]["past_covariates"] = {"promo": [1.0, "mixed"]}

    with TestClient(create_app()) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 400
    detail = json.dumps(response.json())
    assert "covariate" in detail
    assert "mix" in detail


def test_forecast_best_effort_ignores_unsupported_covariates_with_warning(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("mock", accept_license=False, paths=paths)

    runner_manager = _CapturingRunnerManager()
    app = create_app(runner_manager=runner_manager)  # type: ignore[arg-type]
    payload = _sample_forecast_payload()
    payload["series"][0]["past_covariates"] = {"promo": [1.0, 2.0]}
    payload["series"][0]["future_covariates"] = {"promo": [3.0, 4.0]}

    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["warnings"]
    assert "does not support" in body["warnings"][0]
    passed_series = runner_manager.captured["params"]["series"][0]
    assert passed_series.get("past_covariates") is None
    assert passed_series.get("future_covariates") is None


def test_forecast_strict_rejects_unsupported_covariates_before_runner_call(
    monkeypatch,
    tmp_path,
) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("timesfm-2.5-200m", accept_license=False, paths=paths)

    runner_manager = _CapturingRunnerManager()
    app = create_app(runner_manager=runner_manager)  # type: ignore[arg-type]
    payload = {
        "model": "timesfm-2.5-200m",
        "horizon": 2,
        "quantiles": [],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 2.0],
                "past_covariates": {"event": ["off", "on"]},
                "future_covariates": {"event": ["off", "on"]},
            }
        ],
        "options": {},
        "parameters": {"covariates_mode": "strict"},
    }

    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 400
    assert "does not support" in response.json()["detail"]
    assert runner_manager.captured == {}


def test_models_endpoint_returns_available_and_installed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))
    install_from_registry(
        "mock",
        accept_license=False,
        paths=TollamaPaths(base_dir=tmp_path / ".tollama"),
    )

    with TestClient(create_app()) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    mock_available = next(spec for spec in body["available"] if spec["name"] == "mock")
    assert mock_available["family"] == "mock"
    assert mock_available["installed"] is True
    assert mock_available["license"] == {
        "type": "mit",
        "needs_acceptance": False,
        "accepted": True,
    }

    chronos = next(spec for spec in body["available"] if spec["name"] == "chronos2")
    assert chronos["installed"] is False
    assert chronos["license"]["needs_acceptance"] is False

    assert body["installed"] == [
        {
            "name": "mock",
            "family": "mock",
            "installed": True,
            "license": {
                "type": "mit",
                "needs_acceptance": False,
                "accepted": True,
            },
        }
    ]


def test_models_pull_and_delete_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        pull_response = client.post(
            "/v1/models/pull",
            json={"name": "mock", "accept_license": False},
        )
        assert pull_response.status_code == 200
        pulled = pull_response.json()
        assert pulled["name"] == "mock"

        listed = client.get("/v1/models")
        assert listed.status_code == 200
        assert [item["name"] for item in listed.json()["installed"]] == ["mock"]

        delete_response = client.delete("/v1/models/mock")
        assert delete_response.status_code == 200
        assert delete_response.json() == {"removed": True, "name": "mock"}

        delete_again = client.delete("/v1/models/mock")
        assert delete_again.status_code == 404


def test_models_pull_error_mapping(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))

    with TestClient(create_app()) as client:
        missing = client.post(
            "/v1/models/pull",
            json={"name": "does-not-exist", "accept_license": True},
        )
        assert missing.status_code == 404

        conflict = client.post(
            "/v1/models/pull",
            json={"name": "timesfm2p5", "accept_license": False},
        )
        assert conflict.status_code == 409

        bad_payload = client.post(
            "/v1/models/pull",
            json={"name": "mock", "accept_license": "false"},
        )
        assert bad_payload.status_code == 400


class _UnavailableRunnerManager:
    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        raise RunnerUnavailableError(f"runner unavailable for {family}:{method}")

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


class _BadGatewayRunnerManager:
    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        raise RunnerCallError(
            code=-32602,
            message="invalid params",
            data={"family": family, "method": method},
        )

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


class _BadRequestRunnerManager:
    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        raise RunnerCallError(
            code="BAD_REQUEST",
            message="Requested horizon exceeds model prediction_length.",
        )

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


class _CapturingRunnerManager:
    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        self.captured = {
            "family": family,
            "method": method,
            "params": params,
            "timeout": timeout,
        }
        series = params["series"][0]
        return {
            "model": params["model"],
            "forecasts": [
                {
                    "id": series["id"],
                    "freq": series["freq"],
                    "start_timestamp": series["timestamps"][-1],
                    "mean": [1.0] * int(params["horizon"]),
                }
            ],
        }

    def stop(self, family: str | None = None) -> None:
        return None

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        return None


def test_forecast_returns_503_when_runner_unavailable(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    app = create_app(runner_manager=_UnavailableRunnerManager())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 503
    assert "runner unavailable" in response.json()["detail"]


def test_forecast_returns_502_when_runner_returns_error(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    app = create_app(runner_manager=_BadGatewayRunnerManager())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == {
        "code": -32602,
        "message": "invalid params",
        "data": {"family": "mock", "method": "forecast"},
    }


def test_forecast_returns_400_when_runner_returns_bad_request(monkeypatch, tmp_path) -> None:
    _install_model(monkeypatch, tmp_path, "mock")
    app = create_app(runner_manager=_BadRequestRunnerManager())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=_sample_forecast_payload())

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "BAD_REQUEST",
        "message": "Requested horizon exceeds model prediction_length.",
    }


def test_forecast_passes_manifest_source_and_metadata_to_runner(monkeypatch, tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    install_from_registry("granite-ttm-r2", accept_license=False, paths=paths)

    runner_manager = _CapturingRunnerManager()
    app = create_app(runner_manager=runner_manager)  # type: ignore[arg-type]
    payload = {
        "model": "granite-ttm-r2",
        "horizon": 2,
        "quantiles": [],
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01", "2025-01-02"],
                "target": [1.0, 2.0],
            }
        ],
        "options": {},
    }
    with TestClient(app) as client:
        response = client.post("/v1/forecast", json=payload)

    assert response.status_code == 200
    captured_params = runner_manager.captured["params"]
    assert captured_params["model_source"] == {
        "type": "huggingface",
        "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": "90-30-ft-l1-r2.1",
    }
    assert captured_params["model_metadata"] == {
        "implementation": "granite_ttm",
        "context_length": 90,
        "prediction_length": 30,
        "license": "apache-2.0",
    }
