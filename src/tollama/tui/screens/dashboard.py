"""Overview screen for the Textual dashboard."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from ..client import DashboardAPIClient, DashboardAPIError
from ..widgets import EventLogWidget, ModelTableWidget

try:
    from textual.containers import Grid
    from textual.screen import Screen
    from textual.widgets import Header, Static
except Exception:  # pragma: no cover - optional dependency fallback
    class Screen:  # type: ignore[no-redef]
        pass

    class Grid:  # type: ignore[no-redef]
        pass

    class Static:  # type: ignore[no-redef]
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
            yield ModelTableWidget(id="installed-models")
            yield ModelTableWidget(id="loaded-models")
            yield Static("Loading quick stats...", id="quick-stats")
            yield EventLogWidget(highlight=True, id="events")

    async def on_mount(self) -> None:
        installed_table = self.query_one("#installed-models", ModelTableWidget)
        installed_table.setup_columns("Name", "Family", "Size")
        loaded_table = self.query_one("#loaded-models", ModelTableWidget)
        loaded_table.setup_columns("Model", "Family", "Expires")
        if hasattr(self, "set_interval"):
            self.set_interval(5, self.refresh_snapshot)
        await self.refresh_snapshot()
        if hasattr(self, "run_worker"):
            self.run_worker(self._consume_events(), exclusive=True, thread=False)

    async def refresh_snapshot(self) -> None:
        installed_table = self.query_one("#installed-models", ModelTableWidget)
        loaded_table = self.query_one("#loaded-models", ModelTableWidget)
        quick_stats_target = self.query_one("#quick-stats", Static)
        try:
            snapshot = await self._client.dashboard_snapshot()
        except DashboardAPIError as exc:
            hint = "Verify TOLLAMA_API_KEY." if exc.status_code == 401 else exc.hint
            quick_stats_target.update(
                json.dumps(
                    {"error": str(exc), "hint": hint},
                    indent=2,
                    sort_keys=True,
                ),
            )
            return
        except Exception as exc:  # noqa: BLE001
            quick_stats_target.update(
                json.dumps(
                    {"error": str(exc)},
                    indent=2,
                    sort_keys=True,
                ),
            )
            return

        info = snapshot.state.get("info", {})
        ps = snapshot.state.get("ps", {})
        usage = snapshot.state.get("usage", {})
        tags = snapshot.tags.get("models", []) if isinstance(snapshot.tags, dict) else []
        models = ps.get("models") if isinstance(ps, dict) else []
        warnings = snapshot.state.get("warnings", []) if isinstance(snapshot.state, dict) else []

        installed_rows: list[tuple[str, str, str]] = []
        if isinstance(tags, list):
            for item in tags:
                if not isinstance(item, dict):
                    continue
                details = item.get("details")
                family = ""
                if isinstance(details, dict):
                    family = str(details.get("family") or "")
                if not family:
                    family = str(item.get("family") or "")
                installed_rows.append(
                    (
                        str(item.get("name") or item.get("model") or ""),
                        family,
                        str(item.get("size") or 0),
                    ),
                )
        installed_table.set_rows(installed_rows, empty_label="No installed models")

        loaded_rows: list[tuple[str, str, str]] = []
        if isinstance(models, list):
            for item in models:
                if not isinstance(item, dict):
                    continue
                details = item.get("details")
                family = str(details.get("family") or "") if isinstance(details, dict) else ""
                loaded_rows.append(
                    (
                        str(item.get("model") or item.get("name") or ""),
                        family,
                        str(item.get("expires_at") or "-"),
                    ),
                )
        loaded_table.set_rows(loaded_rows, empty_label="No loaded models")

        daemon = info.get("daemon", {}) if isinstance(info, dict) else {}
        usage_summary = usage.get("summary", {}) if isinstance(usage, dict) else {}
        quick_stats_target.update(
            json.dumps(
                {
                    "daemon": {
                        "version": daemon.get("version") if isinstance(daemon, dict) else None,
                        "uptime_seconds": (
                            daemon.get("uptime_seconds") if isinstance(daemon, dict) else None
                        ),
                    },
                    "usage": usage_summary if isinstance(usage_summary, dict) else {},
                    "counts": {
                        "installed_models": len(installed_rows),
                        "loaded_models": len(loaded_rows),
                    },
                    "warnings": warnings,
                },
                indent=2,
                sort_keys=True,
            ),
        )

    async def _consume_events(self) -> None:
        log = self.query_one("#events", EventLogWidget)
        retry_delay_seconds = 1.0
        while True:
            try:
                async for event in self._client.stream_events():
                    name = event.get("event", "message")
                    data = event.get("data", "")
                    if isinstance(data, str):
                        try:
                            normalized = json.loads(data)
                        except json.JSONDecodeError:
                            normalized = data
                    else:
                        normalized = data
                    if isinstance(normalized, (dict, list)):
                        rendered = json.dumps(normalized, sort_keys=True)
                    else:
                        rendered = str(normalized)
                    log.write(f"{name}: {rendered}")
                retry_delay_seconds = 1.0
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except DashboardAPIError as exc:
                log.write(f"stream_error: {exc}")
                await asyncio.sleep(retry_delay_seconds)
                retry_delay_seconds = min(retry_delay_seconds * 2, 10.0)
            except Exception as exc:  # noqa: BLE001
                log.write(f"stream_error: {exc}")
                await asyncio.sleep(retry_delay_seconds)
                retry_delay_seconds = min(retry_delay_seconds * 2, 10.0)

    def action_open_forecast(self) -> None:
        from .forecast import ForecastScreen

        self.app.push_screen(ForecastScreen(self._config))

    def action_open_model_detail(self) -> None:
        from .model_detail import ModelDetailScreen

        self.app.push_screen(ModelDetailScreen(self._config))
