"""Model table widget wrappers."""

from __future__ import annotations

from collections.abc import Iterable

try:
    from textual.widgets import DataTable
except Exception:  # pragma: no cover - optional dependency fallback
    class DataTable:  # type: ignore[no-redef]
        pass


class ModelTableWidget(DataTable):
    """Table used for installed and loaded model listings."""

    _column_count = 0

    def setup_columns(self, *columns: str) -> None:
        """Configure table columns once."""
        if not columns or getattr(self, "_column_count", 0) > 0:
            return
        if hasattr(self, "add_columns"):
            self.add_columns(*columns)  # type: ignore[attr-defined]
            self._column_count = len(columns)

    def set_rows(
        self,
        rows: Iterable[tuple[str, ...]],
        *,
        empty_label: str = "(none)",
    ) -> None:
        """Replace current rows with deterministic string values."""
        if hasattr(self, "clear"):
            self.clear()  # type: ignore[attr-defined]

        prepared = [tuple(str(value) for value in row) for row in rows]
        if not prepared:
            if self._column_count <= 0:
                return
            blank_cells = ("",) * (self._column_count - 1)
            if hasattr(self, "add_row"):
                self.add_row(empty_label, *blank_cells)  # type: ignore[attr-defined]
            return

        if hasattr(self, "add_row"):
            for row in prepared:
                self.add_row(*row)  # type: ignore[attr-defined]
