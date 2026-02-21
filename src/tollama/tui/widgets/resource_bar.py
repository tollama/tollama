"""Resource meter widget wrappers."""

from __future__ import annotations

try:
    from textual.widgets import Static
except Exception:  # pragma: no cover - optional dependency fallback
    class Static:  # type: ignore[no-redef]
        def update(self, *_args, **_kwargs) -> None:
            return


class ResourceBarWidget(Static):
    """Simple text-based resource bar for loaded models."""

    def set_value(self, label: str, fraction: float) -> None:
        width = 16
        clamped = min(max(fraction, 0.0), 1.0)
        filled = int(round(width * clamped))
        self.update(f"{label}: [{'#' * filled}{'.' * (width - filled)}] {int(clamped * 100)}%")
