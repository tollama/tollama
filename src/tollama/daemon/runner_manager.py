"""Family-aware runner process management for daemon forecast calls."""

from __future__ import annotations

import logging
import shutil
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tollama.core.runtime_bootstrap import (
    FAMILY_EXTRAS,
    FAMILY_RUNNER_MODULES,
    BootstrapError,
    ensure_family_runtime,
    runner_command_for_family,
)
from tollama.core.storage import TollamaPaths

from .supervisor import RunnerCallError, RunnerSupervisor, RunnerUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_RUNNER_COMMANDS: dict[str, tuple[str, ...]] = {
    family: (sys.executable, "-m", module)
    for family, module in FAMILY_RUNNER_MODULES.items()
}

UNIMPLEMENTED_FAMILIES = frozenset()


@dataclass(frozen=True)
class _RunnerConfig:
    family: str
    command: tuple[str, ...]


class RunnerManager:
    """Manage lazy RunnerSupervisor instances keyed by model family."""

    def __init__(
        self,
        *,
        runner_commands: Mapping[str, Sequence[str]] | None = None,
        supervisors: Mapping[str, RunnerSupervisor] | None = None,
        auto_bootstrap: bool = False,
        paths: TollamaPaths | None = None,
    ) -> None:
        self._runner_configs = _build_runner_configs(runner_commands)
        self._supervisors: dict[str, RunnerSupervisor] = dict(supervisors or {})
        self._lock = threading.Lock()
        self._auto_bootstrap = auto_bootstrap
        self._paths = paths or TollamaPaths.default()
        # Track families whose commands were explicitly overridden by the caller
        # so that auto-bootstrap never overrides them.
        self._explicit_families: frozenset[str] = (
            frozenset(runner_commands.keys()) if runner_commands else frozenset()
        )

    def list_families(self) -> list[str]:
        """List all configured runner families without starting runners."""
        return list(self._runner_configs.keys())

    def get_all_statuses(self) -> list[dict[str, Any]]:
        """Return runner statuses for all families without starting runners."""
        with self._lock:
            runner_configs = dict(self._runner_configs)
            supervisors = dict(self._supervisors)

        statuses: list[dict[str, Any]] = []
        for family in runner_configs:
            supervisor = supervisors.get(family)
            if supervisor is not None:
                statuses.append(supervisor.get_status(family=family))
                continue

            command = list(runner_configs[family].command)
            installed = bool(command) and shutil.which(command[0]) is not None
            statuses.append(
                {
                    "family": family,
                    "command": command,
                    "installed": installed,
                    "running": False,
                    "pid": None,
                    "started_at": None,
                    "last_used_at": None,
                    "restarts": 0,
                    "last_error": None,
                },
            )
        return statuses

    def call(
        self,
        family: str,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Call one runner method for the specified family."""
        supervisor = self._supervisor_for_family(family)
        try:
            return supervisor.call(method=method, params=params, timeout=timeout)
        except RunnerUnavailableError as exc:
            raise self._rewrite_runner_unavailable_error(family=family, error=exc) from exc

    def unload(self, family: str, *, model: str | None = None, timeout: float) -> None:
        """Unload one model from a family runner, or stop the family runner as fallback."""
        params: dict[str, Any] = {}
        if model is not None:
            params["model"] = model

        try:
            self.call(family=family, method="unload", params=params, timeout=timeout)
            return
        except RunnerCallError as exc:
            if exc.code != -32601:
                raise

        self.stop(family=family)

    def stop(self, family: str | None = None) -> None:
        """Stop one family runner or all runner families."""
        with self._lock:
            if family is None:
                supervisors = list(self._supervisors.values())
                self._supervisors.clear()
            else:
                supervisor = self._supervisors.pop(family, None)
                supervisors = [supervisor] if supervisor is not None else []

        for supervisor in supervisors:
            supervisor.stop()

    def _supervisor_for_family(self, family: str) -> RunnerSupervisor:
        if family in UNIMPLEMENTED_FAMILIES:
            raise RunnerUnavailableError(
                f"runner family {family!r} is not implemented yet",
            )

        with self._lock:
            existing = self._supervisors.get(family)
            if existing is not None:
                return existing

            config = self._runner_configs.get(family)
            if config is None:
                raise RunnerUnavailableError(f"runner family {family!r} is not supported")

            command = config.command

            # Auto-bootstrap: create an isolated venv for this family if the
            # caller did not provide an explicit runner_commands override.
            if (
                self._auto_bootstrap
                and family in FAMILY_EXTRAS
                and family not in self._explicit_families
            ):
                try:
                    venv_python = ensure_family_runtime(family, paths=self._paths)
                    command = runner_command_for_family(family, venv_python)
                    logger.info(
                        "using bootstrapped runtime for %r: %s",
                        family,
                        command[0],
                    )
                except BootstrapError as exc:
                    logger.warning(
                        "auto-bootstrap failed for family %r, falling back to "
                        "default command: %s",
                        family,
                        exc,
                    )

            supervisor = RunnerSupervisor(runner_command=command)
            self._supervisors[family] = supervisor
            return supervisor

    def _rewrite_runner_unavailable_error(
        self,
        *,
        family: str,
        error: RunnerUnavailableError,
    ) -> RunnerUnavailableError:
        config = self._runner_configs.get(family)
        if config is None:
            return error

        message = str(error)
        lowered = message.lower()
        missing_command = "no such file or directory" in lowered or "not found" in lowered
        if not missing_command:
            return error

        extra_name = FAMILY_EXTRAS.get(family)
        if extra_name is None:
            return RunnerUnavailableError(
                f"runner command {config.command[0]!r} is not installed",
            )

        return RunnerUnavailableError(
            "runner command "
            f"{config.command[0]!r} is not installed; install it with "
            f"`pip install -e \".[dev,{extra_name}]\"`",
        )


def _build_runner_configs(
    runner_commands: Mapping[str, Sequence[str]] | None,
) -> dict[str, _RunnerConfig]:
    merged: dict[str, tuple[str, ...]] = dict(DEFAULT_RUNNER_COMMANDS)
    if runner_commands is not None:
        for family, command in runner_commands.items():
            normalized = (command,) if isinstance(command, str) else tuple(command)
            if not normalized:
                raise ValueError(f"runner command for family {family!r} cannot be empty")
            merged[family] = normalized

    return {
        family: _RunnerConfig(family=family, command=command)
        for family, command in merged.items()
    }
