"""Architecture fitness checks for key Tollama boundaries."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from tollama.core.errors import TollamaError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runners_do_not_import_daemon_modules() -> None:
    violations: list[str] = []
    runners_root = REPO_ROOT / "src" / "tollama" / "runners"

    for path in runners_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tollama.daemon"):
                        violations.append(f"{path}: import {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("tollama.daemon"):
                    violations.append(f"{path}: from {node.module} import ...")

    assert violations == []


def test_runner_error_modules_export_shared_error_subclasses() -> None:
    runners_root = REPO_ROOT / "src" / "tollama" / "runners"

    for path in runners_root.glob("*/errors.py"):
        module_name = f"tollama.runners.{path.parent.name}.errors"
        module = importlib.import_module(module_name)
        exported_error_types = [
            obj
            for name, obj in inspect.getmembers(module, inspect.isclass)
            if name.endswith("Error") and obj.__module__.startswith("tollama.")
        ]
        assert exported_error_types, f"{module_name} did not export any error types"
        assert all(issubclass(obj, TollamaError) for obj in exported_error_types)


def test_direct_tollama_env_accesses_are_allowlisted() -> None:
    allowlisted = {
        "src/tollama/daemon/secrets.py",
    }
    violations: list[str] = []
    for base in ("src/tollama/daemon", "src/tollama/core", "src/tollama/cli"):
        for path in (REPO_ROOT / base).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in allowlisted:
                continue
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if 'os.environ.get("TOLLAMA_' in line or 'os.getenv("TOLLAMA_' in line:
                    violations.append(f"{rel}:{line_number}")

    assert violations == []
