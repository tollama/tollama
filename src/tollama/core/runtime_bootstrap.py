"""Per-family virtual environment bootstrap for runner dependency isolation.

Each runner family (
    torch, timesfm, uni2ts, sundial, toto, lag_llama, patchtst, tide, nhits, nbeatsx
) can be installed into its own virtualenv under ``~/.tollama/runtimes/<family>/venv/``.
This
keeps heavyweight and potentially conflicting ML dependencies from interfering
with one another.

The bootstrap is *lazy* by default: when the daemon needs a runner that has no
valid venv yet, it creates one on-the-fly.  Users can also trigger it eagerly
via ``tollama runtime install <family>``.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import venv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tollama import __version__ as _tollama_version

from .storage import TollamaPaths

logger = logging.getLogger(__name__)

# Maps runner family names to the ``pyproject.toml`` optional-extra name that
# carries the family's heavy dependencies.
FAMILY_EXTRAS: dict[str, str] = {
    "torch": "runner_torch",
    "timesfm": "runner_timesfm",
    "uni2ts": "runner_uni2ts",
    "sundial": "runner_sundial",
    "toto": "runner_toto",
    "lag_llama": "runner_lag_llama",
    "patchtst": "runner_patchtst",
    "tide": "runner_tide",
    "nhits": "runner_nhits",
    "nbeatsx": "runner_nbeatsx",
    "timer": "runner_timer",
    "timemixer": "runner_timemixer",
    "forecastpfn": "runner_forecastpfn",
}

# Increment when runtime state compatibility changes even without a package
# version bump (for example, extra-name normalization or bootstrap semantics).
_RUNTIME_STATE_SCHEMA_VERSION = 2

# Families that require specific Python versions.
# uni2ts and timesfm have build-time or runtime failures on Python 3.12+.
# The constraint is checked before creating the venv so users get a clear
# error early rather than a cryptic pip failure.
FAMILY_PYTHON_CONSTRAINTS: dict[str, str] = {
    "uni2ts": "<3.12",
    "timesfm": "<3.12",
}

# Module paths used to build runner commands.
FAMILY_RUNNER_MODULES: dict[str, str] = {
    "mock": "tollama.runners.mock.main",
    "torch": "tollama.runners.torch_runner.main",
    "timesfm": "tollama.runners.timesfm_runner.main",
    "uni2ts": "tollama.runners.uni2ts_runner.main",
    "sundial": "tollama.runners.sundial_runner.main",
    "toto": "tollama.runners.toto_runner.main",
    "lag_llama": "tollama.runners.lag_llama_runner.main",
    "patchtst": "tollama.runners.patchtst_runner.main",
    "tide": "tollama.runners.tide_runner.main",
    "nhits": "tollama.runners.nhits_runner.main",
    "nbeatsx": "tollama.runners.nbeatsx_runner.main",
    "timer": "tollama.runners.timer_runner.main",
    "timemixer": "tollama.runners.timemixer_runner.main",
    "forecastpfn": "tollama.runners.forecastpfn_runner.main",
}

_STATE_FILENAME = "installed.json"


class BootstrapError(RuntimeError):
    """Raised when venv creation or dependency installation fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_family_runtime(
    family: str,
    *,
    paths: TollamaPaths | None = None,
    reinstall: bool = False,
) -> Path:
    """Return the Python interpreter path for *family*'s isolated venv.

    If the venv does not exist or is stale (tollama version mismatch) it is
    (re-)created automatically.

    Returns the absolute ``Path`` to the venv's ``python`` binary.
    """
    if family not in FAMILY_EXTRAS:
        raise ValueError(f"family {family!r} does not support isolated runtimes")

    resolved_paths = paths or TollamaPaths.default()
    family_dir = resolved_paths.runtimes_dir / family
    venv_dir = family_dir / "venv"
    python_path = _venv_python(venv_dir)

    if not reinstall and _is_runtime_valid(family_dir, family):
        logger.debug("runtime for family %r is up-to-date at %s", family, venv_dir)
        return python_path

    _check_python_version(family)
    logger.info("bootstrapping isolated runtime for family %r …", family)
    _create_venv(venv_dir)
    _install_extras(python_path, family)
    _write_runtime_state(family_dir, family)
    logger.info("runtime for family %r ready at %s", family, venv_dir)
    return python_path


def _check_python_version(family: str) -> None:
    """Verify that the current Python interpreter meets the family's constraints."""
    constraint = FAMILY_PYTHON_CONSTRAINTS.get(family)
    if not constraint:
        return

    # Simple check for "<3.12" style constraints.
    py_ver = sys.version_info[:2]

    if constraint == "<3.12" and py_ver >= (3, 12):
        raise BootstrapError(
            f"Family {family!r} requires Python {constraint}, but current python is "
            f"{sys.version.split()[0]}.  Please run tollama with Python 3.11."
        )


def remove_family_runtime(
    family: str,
    *,
    paths: TollamaPaths | None = None,
) -> bool:
    """Remove the isolated venv for *family*.  Returns ``True`` if removed."""
    resolved_paths = paths or TollamaPaths.default()
    family_dir = resolved_paths.runtimes_dir / family
    if not family_dir.exists():
        return False
    shutil.rmtree(family_dir)
    logger.info("removed runtime for family %r at %s", family, family_dir)
    return True


def get_runtime_status(
    family: str,
    *,
    paths: TollamaPaths | None = None,
) -> dict[str, Any]:
    """Return a status dict for one family's runtime directory."""
    resolved_paths = paths or TollamaPaths.default()
    family_dir = resolved_paths.runtimes_dir / family
    state_path = family_dir / _STATE_FILENAME

    python_constraint = FAMILY_PYTHON_CONSTRAINTS.get(family)

    if not state_path.is_file():
        return {
            "family": family,
            "installed": False,
            "venv_path": str(family_dir / "venv"),
            "tollama_version": None,
            "extra": FAMILY_EXTRAS.get(family),
            "python_version": None,
            "python_constraint": python_constraint,
            "installed_at": None,
            "schema_version": None,
        }

    state = _read_state(state_path)
    return {
        "family": family,
        "installed": True,
        "venv_path": str(family_dir / "venv"),
        "tollama_version": state.get("tollama_version"),
        "extra": state.get("extra"),
        "python_version": state.get("python_version"),
        "python_constraint": python_constraint,
        "installed_at": state.get("installed_at"),
        "schema_version": state.get("schema_version"),
    }


def list_runtime_statuses(
    *,
    paths: TollamaPaths | None = None,
) -> list[dict[str, Any]]:
    """Return status dicts for every bootstrappable family."""
    return [get_runtime_status(family, paths=paths) for family in FAMILY_EXTRAS]


def runner_command_for_family(family: str, python_path: Path) -> tuple[str, ...]:
    """Build a runner command tuple that uses the given Python interpreter."""
    module = FAMILY_RUNNER_MODULES.get(family)
    if module is None:
        raise ValueError(f"no runner module registered for family {family!r}")
    return (str(python_path), "-m", module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _venv_python(venv_dir: Path) -> Path:
    """Return the expected python binary path inside a venv."""
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _is_runtime_valid(family_dir: Path, family: str) -> bool:
    """Check that a family runtime exists and was built with the current tollama version."""
    state_path = family_dir / _STATE_FILENAME
    if not state_path.is_file():
        return False

    venv_dir = family_dir / "venv"
    python_path = _venv_python(venv_dir)
    if not python_path.is_file():
        return False

    state = _read_state(state_path)
    if state.get("tollama_version") != _tollama_version:
        logger.debug(
            "runtime for %r has tollama %s (current %s); will re-bootstrap",
            family,
            state.get("tollama_version"),
            _tollama_version,
        )
        return False

    if state.get("extra") != FAMILY_EXTRAS.get(family):
        return False

    if state.get("schema_version") != _RUNTIME_STATE_SCHEMA_VERSION:
        logger.debug(
            "runtime for %r has schema_version=%s (expected %s); will re-bootstrap",
            family,
            state.get("schema_version"),
            _RUNTIME_STATE_SCHEMA_VERSION,
        )
        return False

    return True


def _read_state(state_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _create_venv(venv_dir: Path) -> None:
    """Create (or recreate) a virtual environment at *venv_dir*."""
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("creating venv at %s", venv_dir)
    try:
        venv.create(str(venv_dir), with_pip=True, clear=True)
    except Exception as exc:
        raise BootstrapError(f"failed to create venv at {venv_dir}: {exc}") from exc


def _install_extras(python_path: Path, family: str) -> None:
    """Install tollama with the runner extra into the family venv."""
    extra = FAMILY_EXTRAS[family]

    # Determine the install source.  If tollama is installed in editable mode
    # (development), use the project root so editable changes propagate.
    # Otherwise install from PyPI.
    install_spec = _resolve_install_spec(extra)
    cmd: list[str] = [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--upgrade",
        install_spec,
    ]
    logger.info("installing %s: %s", extra, " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        raise BootstrapError(
            f"pip install failed for family {family!r} (exit {result.returncode}):\n{stderr_tail}"
        )


def _resolve_install_spec(extra: str) -> str:
    """Determine the pip install specifier for the given extra.

    Preference order:
    1) local source tree containing ``pyproject.toml`` (works even when the
       currently-running tollama install is non-editable),
    2) editable-install ``direct_url.json`` metadata,
    3) published package name from PyPI.

    NOTE: pip normalizes extra names in metadata (`_` -> `-`). Keep internal
    names unchanged for user-facing messages, but normalize when building the
    install target.
    """
    normalized_extra = extra.replace("_", "-")

    local_root = _resolve_local_project_root()
    if local_root is not None:
        return f"{local_root}[{normalized_extra}]"

    # Try to detect an editable / local install by inspecting our own package path.
    try:
        from importlib.metadata import distribution

        dist = distribution("tollama")
        # ``direct_url.json`` is present for editable / ``pip install -e .`` installs.
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            url: str = direct_url.get("url", "")
            if url.startswith("file://"):
                project_root = url[len("file://"):]
                return f"{project_root}[{normalized_extra}]"
    except Exception:
        pass

    return f"tollama[{normalized_extra}]"


def _resolve_local_project_root() -> str | None:
    """Return a local repo root when running from a source checkout."""

    def _candidate_root(root: Path) -> str | None:
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            return None
        try:
            content = pyproject.read_text(encoding="utf-8")
        except OSError:
            return None
        if "name = \"tollama\"" in content:
            return str(root)
        return None

    # Prefer the current working directory (or one of its parents). This lets
    # local development checkouts work even when the active ``tollama`` import
    # comes from a previously-installed wheel in a different environment.
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        resolved = _candidate_root(parent)
        if resolved is not None:
            return resolved

    current = Path(__file__).resolve()
    for parent in current.parents:
        resolved = _candidate_root(parent)
        if resolved is not None:
            return resolved
    return None


def _write_runtime_state(family_dir: Path, family: str) -> None:
    """Persist metadata about the bootstrap so we can detect staleness."""
    state = {
        "tollama_version": _tollama_version,
        "extra": FAMILY_EXTRAS[family],
        "python_version": platform.python_version(),
        "installed_at": datetime.now(UTC).isoformat(),
        "schema_version": _RUNTIME_STATE_SCHEMA_VERSION,
    }
    state_path = family_dir / _STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
