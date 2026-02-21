"""ASCII forecast chart widget."""

from __future__ import annotations

from collections.abc import Iterable

try:
    from textual.widgets import Static
except Exception:  # pragma: no cover - optional dependency fallback
    class Static:  # type: ignore[no-redef]
        def update(self, *_args, **_kwargs) -> None:
            return


class ForecastChartWidget(Static):
    """Render forecast values as a compact sparkline-like bar chart."""

    def render_series(self, values: Iterable[float]) -> None:
        points = [float(value) for value in values]
        if not points:
            self.update("No forecast values.")
            return

        minimum = min(points)
        maximum = max(points)
        spread = maximum - minimum or 1.0
        glyphs = "▁▂▃▄▅▆▇█"
        rendered = ""
        for point in points:
            normalized = (point - minimum) / spread
            index = min(int(round(normalized * (len(glyphs) - 1))), len(glyphs) - 1)
            rendered += glyphs[index]
        self.update(f"{rendered}  min={minimum:.4g} max={maximum:.4g}")
