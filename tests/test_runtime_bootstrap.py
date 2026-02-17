"""Tests for per-family runtime bootstrap logic."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tollama.core.runtime_bootstrap import (
    FAMILY_EXTRAS,
    FAMILY_RUNNER_MODULES,
    BootstrapError,
    ensure_family_runtime,
    get_runtime_status,
    list_runtime_statuses,
    remove_family_runtime,
    runner_command_for_family,
)
from tollama.core.storage import TollamaPaths


def _paths(tmp_path: Path) -> TollamaPaths:
    return TollamaPaths(base_dir=tmp_path / ".tollama")


def _write_fake_state(paths: TollamaPaths, family: str, **overrides: object) -> None:
    """Write a fake installed.json so the runtime appears valid."""
    from tollama import __version__

    family_dir = paths.runtimes_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "tollama_version": __version__,
        "extra": FAMILY_EXTRAS[family],
        "python_version": platform.python_version(),
        "installed_at": "2026-01-01T00:00:00+00:00",
    }
    state.update(overrides)
    (family_dir / "installed.json").write_text(json.dumps(state), encoding="utf-8")


def _write_fake_python(paths: TollamaPaths, family: str) -> Path:
    """Create a fake python binary inside the venv directory."""
    venv_dir = paths.runtimes_dir / family / "venv"
    if platform.system() == "Windows":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    python_path.chmod(0o755)
    return python_path


# ---------------------------------------------------------------------------
# ensure_family_runtime
# ---------------------------------------------------------------------------


@patch("tollama.core.runtime_bootstrap._install_extras")
@patch("tollama.core.runtime_bootstrap._create_venv")
def test_ensure_creates_venv_when_missing(
    mock_create: MagicMock, mock_install: MagicMock, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    # After _create_venv is called, fake the python binary so the function works
    def create_side_effect(venv_dir: Path) -> None:
        _write_fake_python(paths, "torch")

    mock_create.side_effect = create_side_effect

    result = ensure_family_runtime("torch", paths=paths)

    mock_create.assert_called_once()
    mock_install.assert_called_once()
    assert "torch" in str(result)
    assert "venv" in str(result)


@patch("tollama.core.runtime_bootstrap._install_extras")
@patch("tollama.core.runtime_bootstrap._create_venv")
def test_ensure_skips_when_valid(
    mock_create: MagicMock, mock_install: MagicMock, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "torch")
    _write_fake_python(paths, "torch")

    result = ensure_family_runtime("torch", paths=paths)

    mock_create.assert_not_called()
    mock_install.assert_not_called()
    assert "python" in str(result)


@patch("tollama.core.runtime_bootstrap._install_extras")
@patch("tollama.core.runtime_bootstrap._create_venv")
def test_ensure_reinstalls_when_version_mismatch(
    mock_create: MagicMock, mock_install: MagicMock, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "torch", tollama_version="0.0.0-old")
    _write_fake_python(paths, "torch")

    def create_side_effect(venv_dir: Path) -> None:
        _write_fake_python(paths, "torch")

    mock_create.side_effect = create_side_effect

    ensure_family_runtime("torch", paths=paths)

    mock_create.assert_called_once()
    mock_install.assert_called_once()


@patch("tollama.core.runtime_bootstrap._install_extras")
@patch("tollama.core.runtime_bootstrap._create_venv")
def test_ensure_reinstalls_when_flag_set(
    mock_create: MagicMock, mock_install: MagicMock, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "torch")
    _write_fake_python(paths, "torch")

    def create_side_effect(venv_dir: Path) -> None:
        _write_fake_python(paths, "torch")

    mock_create.side_effect = create_side_effect

    ensure_family_runtime("torch", paths=paths, reinstall=True)

    mock_create.assert_called_once()
    mock_install.assert_called_once()


def test_ensure_rejects_unknown_family(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    with pytest.raises(ValueError, match="does not support"):
        ensure_family_runtime("unknown_family", paths=paths)


# ---------------------------------------------------------------------------
# remove_family_runtime
# ---------------------------------------------------------------------------


def test_remove_deletes_family_dir(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "torch")
    _write_fake_python(paths, "torch")

    removed = remove_family_runtime("torch", paths=paths)

    assert removed is True
    assert not (paths.runtimes_dir / "torch").exists()


def test_remove_returns_false_when_not_installed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    removed = remove_family_runtime("torch", paths=paths)
    assert removed is False


# ---------------------------------------------------------------------------
# get_runtime_status / list_runtime_statuses
# ---------------------------------------------------------------------------


def test_get_status_installed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "sundial")
    _write_fake_python(paths, "sundial")

    status = get_runtime_status("sundial", paths=paths)

    assert status["installed"] is True
    assert status["family"] == "sundial"
    assert status["extra"] == "runner_sundial"
    assert status["tollama_version"] is not None


def test_get_status_not_installed(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    status = get_runtime_status("sundial", paths=paths)

    assert status["installed"] is False
    assert status["family"] == "sundial"


def test_list_statuses_covers_all_families(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    statuses = list_runtime_statuses(paths=paths)
    families = {s["family"] for s in statuses}
    assert families == set(FAMILY_EXTRAS.keys())


# ---------------------------------------------------------------------------
# runner_command_for_family
# ---------------------------------------------------------------------------


def test_runner_command_for_family_returns_tuple(tmp_path: Path) -> None:
    python_path = tmp_path / "bin" / "python"
    cmd = runner_command_for_family("torch", python_path)
    assert cmd == (str(python_path), "-m", FAMILY_RUNNER_MODULES["torch"])


def test_runner_command_rejects_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no runner module"):
        runner_command_for_family("nonexistent", tmp_path / "python")


# ---------------------------------------------------------------------------
# _resolve_install_spec (indirectly via ensure_family_runtime error path)
# ---------------------------------------------------------------------------


@patch("tollama.core.runtime_bootstrap.subprocess.run")
@patch("tollama.core.runtime_bootstrap.venv.create")
def test_pip_failure_raises_bootstrap_error(
    mock_venv_create: MagicMock, mock_run: MagicMock, tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    def venv_side_effect(venv_dir: str, **kwargs: object) -> None:
        _write_fake_python(paths, "torch")

    mock_venv_create.side_effect = venv_side_effect
    mock_run.return_value = MagicMock(returncode=1, stderr="ERROR: no matching distribution")

    with pytest.raises(BootstrapError, match="pip install failed"):
        ensure_family_runtime("torch", paths=paths)


# ---------------------------------------------------------------------------
# Staleness detection edge cases
# ---------------------------------------------------------------------------


def test_state_file_with_wrong_extra_triggers_reinstall(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_fake_state(paths, "torch", extra="wrong_extra")
    _write_fake_python(paths, "torch")

    with (
        patch("tollama.core.runtime_bootstrap._create_venv") as mock_create,
        patch("tollama.core.runtime_bootstrap._install_extras"),
    ):
        mock_create.side_effect = lambda vd: _write_fake_python(paths, "torch")
        ensure_family_runtime("torch", paths=paths)
        mock_create.assert_called_once()


def test_corrupted_state_json_triggers_reinstall(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    family_dir = paths.runtimes_dir / "torch"
    family_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "installed.json").write_text("{invalid json", encoding="utf-8")
    _write_fake_python(paths, "torch")

    with (
        patch("tollama.core.runtime_bootstrap._create_venv") as mock_create,
        patch("tollama.core.runtime_bootstrap._install_extras"),
    ):
        mock_create.side_effect = lambda vd: _write_fake_python(paths, "torch")
        ensure_family_runtime("torch", paths=paths)
        mock_create.assert_called_once()
