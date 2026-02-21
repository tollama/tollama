"""Tests for ``tollama dev`` CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tollama.cli.main import app


def _seed_project_root(root: Path) -> None:
    (root / "src" / "tollama" / "core").mkdir(parents=True, exist_ok=True)
    (root / "src" / "tollama" / "runners").mkdir(parents=True, exist_ok=True)
    (root / "model-registry").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)

    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                "name = \"tollama\"",
                "",
                "[project.scripts]",
                "tollama = \"tollama.cli.main:main\"",
                "tollamad = \"tollama.daemon.main:main\"",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (root / "src" / "tollama" / "core" / "runtime_bootstrap.py").write_text(
        "\n".join(
            [
                "FAMILY_RUNNER_MODULES: dict[str, str] = {",
                '    "mock": "tollama.runners.mock.main",',
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (root / "model-registry" / "registry.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  - name: mock",
                "    family: mock",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_dev_scaffold_generates_runner_files_without_registration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _seed_project_root(tmp_path)
    monkeypatch.setattr("tollama.cli.dev._resolve_project_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["dev", "scaffold", "acme-family"])

    assert result.exit_code == 0
    package_dir = tmp_path / "src" / "tollama" / "runners" / "acme_family_runner"
    assert (package_dir / "__init__.py").is_file()
    assert (package_dir / "main.py").is_file()
    assert (package_dir / "adapter.py").is_file()
    assert (tmp_path / "tests" / "test_acme_family_runner.py").is_file()

    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert "tollama-runner-acme-family" not in pyproject

    runtime_bootstrap = (tmp_path / "src" / "tollama" / "core" / "runtime_bootstrap.py").read_text(
        encoding="utf-8",
    )
    assert '"acme_family": "tollama.runners.acme_family_runner.main"' not in runtime_bootstrap

    registry = (tmp_path / "model-registry" / "registry.yaml").read_text(encoding="utf-8")
    assert "name: acme_family-example" not in registry


def test_dev_scaffold_with_register_updates_project_files(monkeypatch, tmp_path: Path) -> None:
    _seed_project_root(tmp_path)
    monkeypatch.setattr("tollama.cli.dev._resolve_project_root", lambda: tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["dev", "scaffold", "acme_family", "--register"])

    assert result.exit_code == 0

    pyproject = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'tollama-runner-acme-family = "tollama.runners.acme_family_runner.main:main"'
        in pyproject
    )

    runtime_bootstrap = (tmp_path / "src" / "tollama" / "core" / "runtime_bootstrap.py").read_text(
        encoding="utf-8",
    )
    assert '"acme_family": "tollama.runners.acme_family_runner.main"' in runtime_bootstrap

    registry = (tmp_path / "model-registry" / "registry.yaml").read_text(encoding="utf-8")
    assert "name: acme_family-example" in registry
    assert "family: acme_family" in registry


def test_dev_scaffold_fails_when_files_already_exist_without_force(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _seed_project_root(tmp_path)
    monkeypatch.setattr("tollama.cli.dev._resolve_project_root", lambda: tmp_path)

    runner = CliRunner()
    first = runner.invoke(app, ["dev", "scaffold", "delta"])
    assert first.exit_code == 0

    second = runner.invoke(app, ["dev", "scaffold", "delta"])
    assert second.exit_code == 1
    assert "already exists" in second.output
