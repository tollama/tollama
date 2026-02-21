"""Event log widget wrappers."""

from __future__ import annotations

try:
    from textual.widgets import RichLog
except Exception:  # pragma: no cover - optional dependency fallback
    class RichLog:  # type: ignore[no-redef]
        pass


class EventLogWidget(RichLog):
    """Scrollable live event output widget."""

    pass
