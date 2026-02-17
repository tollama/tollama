"""Tests for ``tollama runtime`` CLI subcommands."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tollama import __version__
from tollama.cli.main import app
from tollama.core.runtime_bootstrap import FAMILY_EXTRAS
from tollama.core.storage import TollamaPaths

runner = CliRunner()


def _paths(tmp_path: Path) -> TollamaPaths:
    return TollamaPaths(base_dir=tmp_path / ".tollama")


def _write_fake_runtime(paths: TollamaPaths, family: str) -> None:
    """Write enough state that the runtime looks installed."""
    family_dir = paths.runtimes_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "tollama_version": __version__,
        "extra": FAMILY_EXTRAS[family],
        "python_version": platform.python_version(),
        "installed_at": "2026-01-01T00:00:00+00:00",
    }
    (family_dir / "installed.json").write_text(json.dumps(state), encoding="utf-8")
    venv_dir = family_dir / "venv"
    if platform.system() == "Windows":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.write_text("#!/bin/sh\n", encoding="utf-8")
    python_path.chmod(0o755)


class TestRuntimeList:
    def test_list_empty(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "list"])
        assert result.exit_code == 0
        for family in FAMILY_EXTRAS:
            assert family in result.output
        assert "✗" in result.output

    def test_list_json(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        families = {entry["family"] for entry in data}
        assert families == set(FAMILY_EXTRAS.keys())

    def test_list_shows_installed(self, monkeypatch: object, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))  # type: ignore[attr-defined]
        _write_fake_runtime(paths, "torch")
        result = runner.invoke(app, ["runtime", "list"])
        assert result.exit_code == 0
        assert "✓" in result.output


class TestRuntimeInstall:
    @patch("tollama.cli.main.ensure_family_runtime")
    def test_install_one(self, mock_ensure: object, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        mock_ensure.return_value = Path("/fake/venv/bin/python")  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "install", "torch"])
        assert result.exit_code == 0
        assert "torch" in result.output
        mock_ensure.assert_called_once()  # type: ignore[attr-defined]

    def test_install_requires_family_or_all(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "install"])
        assert result.exit_code != 0

    def test_install_unknown_family(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "install", "bogus"])
        assert result.exit_code != 0
        assert "unknown family" in result.output

    @patch("tollama.cli.main.ensure_family_runtime")
    def test_install_all(self, mock_ensure: object, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        mock_ensure.return_value = Path("/fake/venv/bin/python")  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "install", "--all"])
        assert result.exit_code == 0
        assert mock_ensure.call_count == len(FAMILY_EXTRAS)  # type: ignore[attr-defined]


class TestRuntimeRemove:
    def test_remove_installed(self, monkeypatch: object, tmp_path: Path) -> None:
        paths = _paths(tmp_path)
        monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))  # type: ignore[attr-defined]
        _write_fake_runtime(paths, "torch")
        result = runner.invoke(app, ["runtime", "remove", "torch"])
        assert result.exit_code == 0
        assert "removed" in result.output
        assert not (paths.runtimes_dir / "torch").exists()

    def test_remove_not_installed(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "remove", "torch"])
        assert result.exit_code == 0
        assert "not installed" in result.output

    def test_remove_requires_family_or_all(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "remove"])
        assert result.exit_code != 0


class TestRuntimeUpdate:
    @patch("tollama.cli.main.ensure_family_runtime")
    def test_update_one(self, mock_ensure: object, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        mock_ensure.return_value = Path("/fake/venv/bin/python")  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "update", "torch"])
        assert result.exit_code == 0
        mock_ensure.assert_called_once()  # type: ignore[attr-defined]
        call_kwargs = mock_ensure.call_args  # type: ignore[attr-defined]
        assert call_kwargs.kwargs.get("reinstall") is True

    def test_update_requires_family_or_all(self, monkeypatch: object, tmp_path: Path) -> None:
        monkeypatch.setenv("TOLLAMA_HOME", str(tmp_path / ".tollama"))  # type: ignore[attr-defined]
        result = runner.invoke(app, ["runtime", "update"])
        assert result.exit_code != 0
