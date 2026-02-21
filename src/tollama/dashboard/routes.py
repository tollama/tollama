"""HTML partial routes used by the dashboard web frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


def create_dashboard_html_router(*, partials_dir: Path) -> APIRouter:
    """Serve static HTML partial templates by name."""
    router = APIRouter(prefix="/dashboard/partials", include_in_schema=False)

    @router.get("/{name}")
    def partial(name: str) -> FileResponse:
        safe_name = name.strip().replace("..", "")
        if not safe_name.endswith(".html"):
            safe_name = f"{safe_name}.html"
        target = partials_dir / safe_name
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail=f"partial {safe_name!r} not found")
        return FileResponse(target)

    return router
