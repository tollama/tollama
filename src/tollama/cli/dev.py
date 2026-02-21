"""Developer-focused CLI helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import typer

from .templates.scaffold import (
    REGISTRY_ENTRY_TEMPLATE,
    RUNNER_ADAPTER_TEMPLATE,
    RUNNER_INIT_TEMPLATE,
    RUNNER_MAIN_TEMPLATE,
    RUNNER_TEST_TEMPLATE,
)

dev_app = typer.Typer(help="Developer commands for extending tollama.")

_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SCRIPTS_SECTION = "[project.scripts]"
_RUNNER_MODULE_MAP_NAME = "FAMILY_RUNNER_MODULES"


@dataclass(frozen=True)
class ScaffoldPaths:
    """Resolved output paths for one scaffolded runner family."""

    package_dir: Path
    init_py: Path
    main_py: Path
    adapter_py: Path
    test_py: Path
    pyproject_toml: Path
    runtime_bootstrap_py: Path
    registry_yaml: Path


@dev_app.command("scaffold")
def scaffold_runner(
    family_name: str = typer.Argument(..., help="Runner family name in snake_case."),
    register: bool = typer.Option(
        False,
        "--register",
        help="Also register runner script, family module map, and registry template entry.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite generated files when they already exist.",
    ),
) -> None:
    """Generate a new runner skeleton with tests and optional registration edits."""
    family = _normalize_family_name(family_name)
    try:
        root = _resolve_project_root()
        paths = _scaffold_paths(root=root, family=family)

        _ensure_parent_dirs(paths)
        class_name = _adapter_class_name(family)
        _write_file(
            paths.init_py,
            RUNNER_INIT_TEMPLATE.format(family=family),
            force=force,
        )
        _write_file(
            paths.adapter_py,
            RUNNER_ADAPTER_TEMPLATE.format(family=family, class_name=class_name),
            force=force,
        )
        _write_file(
            paths.main_py,
            RUNNER_MAIN_TEMPLATE.format(family=family, class_name=class_name),
            force=force,
        )
        _write_file(
            paths.test_py,
            RUNNER_TEST_TEMPLATE.format(family=family),
            force=force,
        )

        modified_files: list[Path] = [paths.init_py, paths.adapter_py, paths.main_py, paths.test_py]

        if register:
            script_name = f"tollama-runner-{family.replace('_', '-')}"
            script_target = f"tollama.runners.{family}_runner.main:main"
            if _upsert_toml_script_entry(paths.pyproject_toml, script_name, script_target):
                modified_files.append(paths.pyproject_toml)
            if _upsert_runner_module_entry(paths.runtime_bootstrap_py, family):
                modified_files.append(paths.runtime_bootstrap_py)
            model_name = f"{family}-example"
            if _append_registry_entry(paths.registry_yaml, family=family, model_name=model_name):
                modified_files.append(paths.registry_yaml)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Scaffolded runner family {family!r}.")
    typer.echo("")
    typer.echo("Modified files:")
    for path in modified_files:
        typer.echo(f"  - {path}")

    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Implement adapter.forecast in adapter.py.")
    typer.echo("  2. Run tests: pytest -q tests/test_" + family + "_runner.py")
    typer.echo("  3. If you used --register, reinstall: python -m pip install -e \".[dev]\"")
    typer.echo("  4. Add real model metadata/capabilities in model-registry/registry.yaml.")


def _normalize_family_name(raw: str) -> str:
    family = raw.strip().lower().replace("-", "_")
    if not _FAMILY_RE.fullmatch(family):
        raise typer.BadParameter(
            "family_name must be snake_case and start with a letter (example: acme)",
        )
    return family


def _resolve_project_root() -> Path:
    resolved = Path(__file__).resolve()
    if len(resolved.parents) < 4:
        raise RuntimeError("unable to resolve project root from module location")
    return resolved.parents[3]


def _scaffold_paths(*, root: Path, family: str) -> ScaffoldPaths:
    package_dir = root / "src" / "tollama" / "runners" / f"{family}_runner"
    return ScaffoldPaths(
        package_dir=package_dir,
        init_py=package_dir / "__init__.py",
        main_py=package_dir / "main.py",
        adapter_py=package_dir / "adapter.py",
        test_py=root / "tests" / f"test_{family}_runner.py",
        pyproject_toml=root / "pyproject.toml",
        runtime_bootstrap_py=root / "src" / "tollama" / "core" / "runtime_bootstrap.py",
        registry_yaml=root / "model-registry" / "registry.yaml",
    )


def _adapter_class_name(family: str) -> str:
    return "".join(chunk.capitalize() for chunk in family.split("_")) + "Adapter"


def _ensure_parent_dirs(paths: ScaffoldPaths) -> None:
    paths.package_dir.mkdir(parents=True, exist_ok=True)
    paths.test_py.parent.mkdir(parents=True, exist_ok=True)


def _write_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise RuntimeError(f"{path} already exists (use --force to overwrite)")
    path.write_text(content, encoding="utf-8")


def _upsert_toml_script_entry(path: Path, script_name: str, target: str) -> bool:
    line = f'{script_name} = "{target}"'
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    if any(current.strip().startswith(f"{script_name} =") for current in lines):
        return False

    section_index = _find_line_index(lines, _SCRIPTS_SECTION)
    if section_index is None:
        raise RuntimeError(f"unable to find {_SCRIPTS_SECTION} in {path}")

    insert_index = len(lines)
    for idx in range(section_index + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            insert_index = idx
            break

    lines.insert(insert_index, line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _find_line_index(lines: list[str], marker: str) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip() == marker:
            return idx
    return None


def _upsert_runner_module_entry(path: Path, family: str) -> bool:
    marker = f'    "{family}": "tollama.runners.{family}_runner.main",'
    content = path.read_text(encoding="utf-8")
    if marker in content:
        return False

    map_start = content.find(f"{_RUNNER_MODULE_MAP_NAME}: dict[str, str] = {{")
    if map_start < 0:
        raise RuntimeError(f"unable to find {_RUNNER_MODULE_MAP_NAME} in {path}")

    map_end = content.find("\n}\n", map_start)
    if map_end < 0:
        raise RuntimeError(f"unable to locate end of {_RUNNER_MODULE_MAP_NAME} in {path}")

    updated = content[:map_end] + marker + "\n" + content[map_end:]
    path.write_text(updated, encoding="utf-8")
    return True


def _append_registry_entry(path: Path, *, family: str, model_name: str) -> bool:
    existing = path.read_text(encoding="utf-8")
    if f"name: {model_name}" in existing:
        return False
    entry = REGISTRY_ENTRY_TEMPLATE.format(family=family, model_name=model_name).rstrip()
    if not existing.endswith("\n"):
        existing += "\n"
    updated = existing.rstrip() + "\n" + entry + "\n"
    path.write_text(updated, encoding="utf-8")
    return True
