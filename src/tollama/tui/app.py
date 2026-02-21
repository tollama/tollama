"""Main Textual app entrypoint for Tollama dashboard."""

from __future__ import annotations

from typing import Any

_TEXTUAL_IMPORT_ERROR: Exception | None = None

try:
    from textual.app import App
    from textual.widgets import Footer

    from .screens.dashboard import DashboardConfig, DashboardScreen
except Exception as exc:  # pragma: no cover - optional dependency fallback
    _TEXTUAL_IMPORT_ERROR = exc
    App = object  # type: ignore[assignment]
    Footer = object  # type: ignore[assignment]


if _TEXTUAL_IMPORT_ERROR is None:

    class TollamaDashboardApp(App[None]):
        """Top-level Textual application for Tollama operations monitoring."""

        CSS_PATH = "styles/dashboard.tcss"
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("f", "open_forecast", "Forecast"),
            ("tab", "focus_next", "Next"),
        ]

        def __init__(
            self,
            *,
            base_url: str,
            timeout: float,
            api_key: str | None = None,
        ) -> None:
            super().__init__()
            self._config = DashboardConfig(
                base_url=base_url,
                timeout=timeout,
                api_key=api_key,
            )

        def compose(self):  # type: ignore[override]
            yield Footer()

        async def on_mount(self) -> None:
            await self.push_screen(DashboardScreen(self._config))

        def action_open_forecast(self) -> None:
            screen = self.screen
            if hasattr(screen, "action_open_forecast"):
                screen.action_open_forecast()

else:

    class TollamaDashboardApp:  # pragma: no cover - fallback object
        """Fallback shim when Textual is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            message = (
                "Textual is required for `tollama dashboard`. "
                "Install with: python -m pip install -e '.[tui]'"
            )
            if _TEXTUAL_IMPORT_ERROR is not None:
                message = f"{message} ({_TEXTUAL_IMPORT_ERROR})"
            raise RuntimeError(message)


def run_dashboard_app(*, base_url: str, timeout: float, api_key: str | None = None) -> None:
    """Run the Textual dashboard application."""
    app = TollamaDashboardApp(base_url=base_url, timeout=timeout, api_key=api_key)
    run = getattr(app, "run", None)
    if not callable(run):
        raise RuntimeError("Textual dashboard is unavailable")
    run()


def main() -> None:
    """Console entrypoint for `tollama-dashboard`."""
    run_dashboard_app(base_url="http://localhost:11435", timeout=10.0, api_key=None)
