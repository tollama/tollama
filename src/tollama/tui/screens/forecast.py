"""Forecast playground screen for TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..client import DashboardAPIClient, DashboardAPIError
from ..widgets.forecast_chart import ForecastChartWidget
from .dashboard import DashboardConfig

try:
    from textual.containers import Horizontal, Vertical
    from textual.screen import Screen
    from textual.widgets import Button, Input, Static, TextArea
except Exception:  # pragma: no cover - optional dependency fallback
    class Screen:  # type: ignore[no-redef]
        pass

    class Vertical:  # type: ignore[no-redef]
        pass

    class Horizontal:  # type: ignore[no-redef]
        pass

    class Button:  # type: ignore[no-redef]
        pass

    class Input:  # type: ignore[no-redef]
        pass

    class Static:  # type: ignore[no-redef]
        def update(self, *_args, **_kwargs) -> None:
            return

    class TextArea:  # type: ignore[no-redef]
        pass


class ForecastScreen(Screen):
    """Submit forecast requests and render ASCII chart output."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(self, config: DashboardConfig) -> None:
        super().__init__()
        self._client = DashboardAPIClient(
            base_url=config.base_url,
            timeout=config.timeout,
            api_key=config.api_key,
        )
        self._last_response: dict[str, Any] | None = None

    def compose(self):  # type: ignore[override]
        with Vertical(id="forecast-layout"):
            yield Input(value="mock", id="forecast-model", placeholder="Model")
            yield Input(value="3", id="forecast-horizon", placeholder="Horizon")
            yield TextArea(
                text='[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[10,11]}]',
                id="forecast-series",
            )
            with Horizontal():
                yield Button("Run forecast", id="forecast-submit")
                yield Button("Export JSON", id="forecast-export-json")
                yield Button("Export CSV", id="forecast-export-csv")
            yield ForecastChartWidget(id="forecast-chart")
            yield Static("", id="forecast-output")

    async def on_button_pressed(self, event: Any) -> None:
        button_id = getattr(event.button, "id", None)
        if button_id == "forecast-export-json":
            await self._export_json()
            return
        if button_id == "forecast-export-csv":
            await self._export_csv()
            return
        if button_id != "forecast-submit":
            return

        model = self.query_one("#forecast-model", Input).value.strip()
        horizon_raw = self.query_one("#forecast-horizon", Input).value.strip()
        series_raw = self.query_one("#forecast-series", TextArea).text
        output = self.query_one("#forecast-output", Static)
        chart = self.query_one("#forecast-chart", ForecastChartWidget)

        try:
            horizon = int(horizon_raw)
            series = json.loads(series_raw)
            payload = {
                "model": model,
                "horizon": horizon,
                "series": series,
                "options": {},
            }
            response = await self._client.forecast(payload)
            self._last_response = response
            first = response.get("forecasts", [{}])[0]
            mean = first.get("mean", []) if isinstance(first, dict) else []
            if isinstance(mean, list):
                chart.render_series([float(item) for item in mean])
            output.update(json.dumps(response, indent=2, sort_keys=True))
        except DashboardAPIError as exc:
            if exc.status_code == 401:
                output.update("Forecast failed: unauthorized. Verify TOLLAMA_API_KEY.")
                return
            output.update(f"Forecast failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            output.update(f"Forecast failed: {exc}")

    async def _export_json(self) -> None:
        output = self.query_one("#forecast-output", Static)
        if self._last_response is None:
            output.update("No forecast response to export.")
            return
        destination = Path.cwd() / "tollama_forecast.json"
        destination.write_text(
            json.dumps(self._last_response, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output.update(f"Saved JSON to {destination}")

    async def _export_csv(self) -> None:
        output = self.query_one("#forecast-output", Static)
        if self._last_response is None:
            output.update("No forecast response to export.")
            return
        forecast = self._last_response.get("forecasts", [{}])[0]
        if not isinstance(forecast, dict):
            output.update("Forecast response has unexpected shape.")
            return
        mean = forecast.get("mean", [])
        timestamps = forecast.get("timestamps", [])
        if not isinstance(mean, list):
            output.update("Forecast response has no mean values.")
            return
        rows = ["step,timestamp,mean"]
        for index, value in enumerate(mean, start=1):
            timestamp = ""
            if isinstance(timestamps, list) and index - 1 < len(timestamps):
                timestamp = timestamps[index - 1]
            rows.append(f"{index},{timestamp},{value}")
        destination = Path.cwd() / "tollama_forecast.csv"
        destination.write_text("\\n".join(rows), encoding="utf-8")
        output.update(f"Saved CSV to {destination}")
