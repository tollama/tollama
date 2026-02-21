"""Model table widget wrappers."""

from __future__ import annotations

try:
    from textual.widgets import DataTable
except Exception:  # pragma: no cover - optional dependency fallback
    class DataTable:  # type: ignore[no-redef]
        pass


class ModelTableWidget(DataTable):
    """Table used for installed and loaded model listings."""

    pass
