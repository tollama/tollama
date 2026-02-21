"""Overview screen for the Textual dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..client import DashboardAPIClient

try:
    from textual.containers import Grid
    from textual.screen import Screen
    from textual.widgets import Header, RichLog, Static
except Exception:  # pragma: no cover - optional dependency fallback
    class Screen:  # type: ignore[no-redef]
        pass

    class Grid:  # type: ignore[no-redef]
        pass

    class Static:  # type: ignore[no-redef]
        pass

    class RichLog:  # type: ignore[no-redef]
        pass

    class Header:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class DashboardConfig:
    """Runtime connection config for dashboard screens."""

    base_url: str
    timeout: float
    api_key: str | None


class DashboardScreen(Screen):
    """Operational dashboard overview with polling and event log."""

    BINDINGS = [
        ("f", "open_forecast", "Forecast"),
        ("enter", "open_model_detail", "Model detail"),
    ]

    def __init__(self, config: DashboardConfig) -> None:
        super().__init__()
        self._config = config
        self._client = DashboardAPIClient(
            base_url=config.base_url,
            timeout=config.timeout,
            api_key=config.api_key,
        )

    def compose(self):  # type: ignore[override]
        yield Header(show_clock=True)
        with Grid(id="dashboard-grid"):
            yield Static("Loading installed models...", id="installed-models")
            yield Static("Loading loaded models...", id="loaded-models")
            yield Static("Loading quick stats...", id="quick-stats")
            yield RichLog(highlight=True, id="events")

    async def on_mount(self) -> None:
        if hasattr(self, "set_interval"):
            self.set_interval(5, self.refresh_snapshot)
        await self.refresh_snapshot()
        if hasattr(self, "run_worker"):
            self.run_worker(self._consume_events(), exclusive=True, thread=False)

    async def refresh_snapshot(self) -> None:
        snapshot = await self._client.dashboard_snapshot()
        info = snapshot.state.get("info", {})
        ps = snapshot.state.get("ps", {})
        usage = snapshot.state.get("usage", {})
        tags = snapshot.tags.get("models", []) if isinstance(snapshot.tags, dict) else []
        models = ps.get("models") if isinstance(ps, dict) else []
        warnings = snapshot.state.get("warnings", [])

        installed_target = self.query_one("#installed-models", Static)
        loaded_target = self.query_one("#loaded-models", Static)
        quick_stats_target = self.query_one("#quick-stats", Static)

        installed_target.update(
            json.dumps(
                {"installed": tags},
                indent=2,
                sort_keys=True,
            ),
        )
        loaded_target.update(
            json.dumps(
                {"loaded": models if isinstance(models, list) else []},
                indent=2,
                sort_keys=True,
            ),
        )
        quick_stats_target.update(
            json.dumps(
                {
                    "daemon": info.get("daemon", {}),
                    "usage": usage.get("summary") if isinstance(usage, dict) else {},
                    "warnings": warnings,
                },
                indent=2,
                sort_keys=True,
            ),
        )

    async def _consume_events(self) -> None:
        log = self.query_one("#events", RichLog)
        async for event in self._client.stream_events():
            name = event.get("event", "message")
            data = event.get("data", "")
            log.write(f"{name}: {data}")

    def action_open_forecast(self) -> None:
        from .forecast import ForecastScreen

        self.app.push_screen(ForecastScreen(self._config))

    def action_open_model_detail(self) -> None:
        from .model_detail import ModelDetailScreen

        self.app.push_screen(ModelDetailScreen(self._config))
