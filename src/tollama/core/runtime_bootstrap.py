"""Per-family virtual environment bootstrap for runner dependency isolation.

Each bootstrappable runner family can be installed into its own virtualenv
under ``~/.tollama/runtimes/<family>/venv/``. The exact family-to-extra and
family-to-module mappings live in ``FAMILY_EXTRAS`` and
``FAMILY_RUNNER_MODULES`` below, which avoids stale hardcoded family lists in
the module docstring. This keeps heavyweight and potentially conflicting ML
dependencies from interfering with one another.

The bootstrap is *lazy* by default: when the daemon needs a runner that has no
valid venv yet, it creates one on-the-fly.  Users can also trigger it eagerly
via ``tollama runtime install <family>``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tomllib
import venv
from datetime import UTC, datetime
from hashlib import sha256
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
_RUNTIME_STATE_SCHEMA_VERSION = 4

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
BOOTSTRAP_WHEELHOUSE_ENV_NAME = "TOLLAMA_RUNTIME_WHEELHOUSE"
_DEPENDENCY_FINGERPRINT_ALGORITHM = "sha256"
_SOURCE_FINGERPRINT_ALGORITHM = "sha256"
_SOURCE_FINGERPRINT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    }
)


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
            "dependency_fingerprint": None,
            "dependency_fingerprint_algorithm": _DEPENDENCY_FINGERPRINT_ALGORITHM,
            "extra_dependencies": None,
            "source_fingerprint": None,
            "source_fingerprint_algorithm": _SOURCE_FINGERPRINT_ALGORITHM,
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
        "dependency_fingerprint": state.get("dependency_fingerprint"),
        "dependency_fingerprint_algorithm": state.get("dependency_fingerprint_algorithm"),
        "extra_dependencies": state.get("extra_dependencies"),
        "source_fingerprint": state.get("source_fingerprint"),
        "source_fingerprint_algorithm": state.get("source_fingerprint_algorithm"),
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

    expected_dependency_fingerprint = _runtime_dependency_fingerprint(family)
    if state.get("dependency_fingerprint") != expected_dependency_fingerprint:
        logger.debug(
            "runtime for %r has dependency_fingerprint=%s (expected %s); will re-bootstrap",
            family,
            state.get("dependency_fingerprint"),
            expected_dependency_fingerprint,
        )
        return False

    expected_source_fingerprint = _runtime_source_fingerprint()
    if (
        expected_source_fingerprint is not None
        and state.get("source_fingerprint") != expected_source_fingerprint
    ):
        logger.debug(
            "runtime for %r has source_fingerprint=%s (expected %s); will re-bootstrap",
            family,
            state.get("source_fingerprint"),
            expected_source_fingerprint,
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
        venv.create(
            str(venv_dir),
            with_pip=True,
            clear=True,
            symlinks=platform.system() != "Windows",
        )
        return
    except Exception as exc:
        uv_binary = shutil.which("uv")
        if uv_binary:
            logger.warning(
                "stdlib venv bootstrap failed for %s; retrying with uv: %s",
                venv_dir,
                exc,
            )
            cmd = [
                uv_binary,
                "venv",
                "--seed",
                "--clear",
                str(venv_dir),
                "--python",
                sys.executable,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return
            stderr_tail = ((result.stderr or "") or (result.stdout or ""))[-2000:]
            raise BootstrapError(
                f"failed to create venv at {venv_dir}: {exc}\n"
                f"uv fallback failed (exit {result.returncode}):\n{stderr_tail}"
            ) from exc
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
    ]
    cmd.extend(_wheelhouse_install_args())
    cmd.append(install_spec)
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


def _wheelhouse_install_args() -> list[str]:
    """Return optional pip args for an offline wheelhouse when configured."""
    raw_path = os.environ.get(BOOTSTRAP_WHEELHOUSE_ENV_NAME, "").strip()
    if not raw_path:
        return []

    wheelhouse = Path(raw_path).expanduser()
    if not wheelhouse.is_dir():
        logger.warning(
            "ignoring %s=%r because the directory does not exist",
            BOOTSTRAP_WHEELHOUSE_ENV_NAME,
            raw_path,
        )
        return []

    # Prefer bundled/local wheels when available without blocking index fallback
    # for families whose extras are not present in the wheelhouse.
    return ["--find-links", str(wheelhouse)]


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
                project_root = url[len("file://") :]
                return f"{project_root}[{normalized_extra}]"
    except Exception:
        pass

    return f"tollama[{normalized_extra}]=={_tollama_version}"


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
        if 'name = "tollama"' in content:
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
    extra_dependencies = _extra_dependency_specs(FAMILY_EXTRAS[family])
    source_fingerprint = _runtime_source_fingerprint()
    state = {
        "tollama_version": _tollama_version,
        "extra": FAMILY_EXTRAS[family],
        "dependency_fingerprint": _dependency_fingerprint(extra_dependencies),
        "dependency_fingerprint_algorithm": _DEPENDENCY_FINGERPRINT_ALGORITHM,
        "extra_dependencies": extra_dependencies,
        "python_version": platform.python_version(),
        "installed_at": datetime.now(UTC).isoformat(),
        "schema_version": _RUNTIME_STATE_SCHEMA_VERSION,
    }
    if source_fingerprint is not None:
        state["source_fingerprint"] = source_fingerprint
        state["source_fingerprint_algorithm"] = _SOURCE_FINGERPRINT_ALGORITHM
    state_path = family_dir / _STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _runtime_dependency_fingerprint(family: str) -> str:
    return _dependency_fingerprint(_extra_dependency_specs(FAMILY_EXTRAS[family]))


def _runtime_source_fingerprint() -> str | None:
    local_root = _resolve_local_project_root()
    if local_root is None:
        return None

    root = Path(local_root)
    candidates = (root / "pyproject.toml", root / "src" / "tollama")
    digest = sha256()
    hashed_any = False

    for candidate in candidates:
        if candidate.is_file():
            hashed_any = _update_source_digest(digest, root, candidate) or hashed_any
            continue
        if not candidate.is_dir():
            continue
        for source_path in sorted(candidate.rglob("*.py")):
            relative_parts = source_path.relative_to(root).parts
            if any(part in _SOURCE_FINGERPRINT_EXCLUDED_DIRS for part in relative_parts):
                continue
            hashed_any = _update_source_digest(digest, root, source_path) or hashed_any

    return digest.hexdigest() if hashed_any else None


def _update_source_digest(digest: Any, root: Path, source_path: Path) -> bool:
    try:
        relative = source_path.relative_to(root).as_posix()
        content = source_path.read_bytes()
    except OSError:
        return False

    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    digest.update(b"\0")
    return True


def _dependency_fingerprint(dependencies: list[str]) -> str:
    payload = json.dumps(dependencies, separators=(",", ":"), sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _extra_dependency_specs(extra: str) -> list[str]:
    local_root = _resolve_local_project_root()
    if local_root is not None:
        pyproject = Path(local_root) / "pyproject.toml"
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        optional_dependencies = data.get("project", {}).get("optional-dependencies", {})
        if isinstance(optional_dependencies, dict):
            for extra_key in _extra_key_candidates(extra):
                if extra_key in optional_dependencies:
                    return _normalize_dependency_specs(optional_dependencies.get(extra_key))

    try:
        from importlib.metadata import distribution

        dist = distribution("tollama")
        requirements = dist.requires or []
    except Exception:
        return []

    normalized_extra = extra.replace("_", "-").lower()
    specs: list[str] = []
    for requirement in requirements:
        dependency, separator, marker = requirement.partition(";")
        if not separator:
            continue
        normalized_marker = marker.replace('"', "'").replace("_", "-").lower()
        if f"extra == '{normalized_extra}'" in normalized_marker:
            specs.append(_normalize_dependency_spec(dependency))
    return sorted(spec for spec in specs if spec)


def _extra_key_candidates(extra: str) -> tuple[str, ...]:
    normalized = extra.replace("_", "-")
    underscored = extra.replace("-", "_")
    candidates: list[str] = []
    for candidate in (extra, normalized, underscored):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _normalize_dependency_specs(raw_specs: object) -> list[str]:
    if not isinstance(raw_specs, list):
        return []
    return sorted(
        spec
        for spec in (_normalize_dependency_spec(raw_spec) for raw_spec in raw_specs)
        if spec
    )


def _normalize_dependency_spec(raw_spec: object) -> str:
    if not isinstance(raw_spec, str):
        return ""
    return " ".join(raw_spec.strip().split())
