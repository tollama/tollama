"""Phase-1 placeholder adapter for N-HiTS runner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tollama.core.schemas import ForecastRequest

from .errors import DependencyMissingError, NotImplementedRunnerError

_INSTALL_HINT = 'python -m pip install -e ".[dev,runner_nhits]"'


class NhitsAdapter:
    """Capability-gated placeholder adapter for initial N-HiTS integration."""

    def unload(self, model_name: str | None = None) -> None:
        del model_name

    def forecast(
        self,
        request: ForecastRequest,
        *,
        model_local_dir: str | None = None,
        model_source: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        del request, model_local_dir, model_source

        metadata = dict(model_metadata or {})
        status = str(metadata.get("status") or "").strip().lower()
        if status and status != "phase1_placeholder":
            raise NotImplementedRunnerError(
                "N-HiTS runner implementation is not available yet for status "
                f"{status!r}. Track progress in tollama milestones.",
            )

        if metadata.get("dependency_probe", True):
            raise DependencyMissingError(
                "N-HiTS Phase-1 baseline is scaffolded but optional runtime dependencies "
                f"are not installed yet. Install with `{_INSTALL_HINT}` and retry.",
            )

        raise NotImplementedRunnerError(
            "N-HiTS Phase-1 baseline runner is discoverable and routable, but forecast "
            "execution is intentionally disabled. Full inference will land in a follow-up "
            "phase.",
        )
