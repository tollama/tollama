"""Model detail/action screen for TUI."""

from __future__ import annotations

import json
from typing import Any

from ..client import DashboardAPIClient
from .dashboard import DashboardConfig

try:
    from textual.containers import Horizontal, Vertical
    from textual.screen import Screen
    from textual.widgets import Button, Input, RichLog, Static
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

    class RichLog:  # type: ignore[no-redef]
        def write(self, *_args, **_kwargs) -> None:
            return


class ModelDetailScreen(Screen):
    """Inspect model metadata and trigger model actions."""

    BINDINGS = [("escape", "pop_screen", "Back")]

    def __init__(self, config: DashboardConfig) -> None:
        super().__init__()
        self._client = DashboardAPIClient(
            base_url=config.base_url,
            timeout=config.timeout,
            api_key=config.api_key,
        )

    def compose(self):  # type: ignore[override]
        with Vertical(id="model-detail-layout"):
            yield Input(value="mock", id="model-name", placeholder="Model")
            with Horizontal():
                yield Button("Show", id="model-show")
                yield Button("Pull", id="model-pull")
                yield Button("Delete", id="model-delete")
            yield Static("", id="model-output")
            yield RichLog(id="model-events")

    async def on_button_pressed(self, event: Any) -> None:
        button_id = getattr(event.button, "id", "")
        model = self.query_one("#model-name", Input).value.strip()
        output = self.query_one("#model-output", Static)
        events = self.query_one("#model-events", RichLog)

        try:
            if button_id == "model-show":
                payload = await self._client.show_model(model)
                output.update(json.dumps(payload, indent=2, sort_keys=True))
                return
            if button_id == "model-delete":
                payload = await self._client.delete_model(model)
                output.update(json.dumps(payload, indent=2, sort_keys=True))
                return
            if button_id == "model-pull":
                pull_events = await self._client.pull_model_events(model)
                for entry in pull_events:
                    events.write(json.dumps(entry, sort_keys=True))
                output.update(f"Pull events: {len(pull_events)}")
                return
        except Exception as exc:  # noqa: BLE001
            output.update(f"Action failed: {exc}")
