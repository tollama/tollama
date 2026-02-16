"""Family-aware runner process management for daemon forecast calls."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .supervisor import RunnerCallError, RunnerSupervisor, RunnerUnavailableError

DEFAULT_RUNNER_COMMANDS: dict[str, tuple[str, ...]] = {
    "mock": ("tollama-runner-mock",),
    "torch": ("tollama-runner-torch",),
    "timesfm": ("tollama-runner-timesfm",),
    "uni2ts": ("tollama-runner-uni2ts",),
}

FAMILY_EXTRAS: dict[str, str] = {
    "torch": "runner_torch",
}

UNIMPLEMENTED_FAMILIES = frozenset({"timesfm", "uni2ts"})


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
    ) -> None:
        self._runner_configs = _build_runner_configs(runner_commands)
        self._supervisors: dict[str, RunnerSupervisor] = dict(supervisors or {})
        self._lock = threading.Lock()

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

            supervisor = RunnerSupervisor(runner_command=config.command)
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
