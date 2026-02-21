"""Forecast form widget wrappers."""

from __future__ import annotations

try:
    from textual.widgets import Static
except Exception:  # pragma: no cover - optional dependency fallback
    class Static:  # type: ignore[no-redef]
        pass


class ForecastFormWidget(Static):
    """Container marker for forecast form controls."""

    pass
