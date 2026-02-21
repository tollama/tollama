"""Dashboard-focused daemon routes and aggregation helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from tollama.core.schemas import DashboardStateResponse, DashboardStateWarning

DashboardProvider = Callable[[], dict[str, Any]]
DashboardRequestProvider = Callable[[Request], dict[str, Any]]


def create_dashboard_router(
    *,
    info_provider: DashboardProvider,
    ps_provider: DashboardProvider,
    usage_provider: DashboardRequestProvider,
) -> APIRouter:
    """Build dashboard API routes using existing daemon providers."""
    router = APIRouter()

    @router.get(
        "/api/dashboard/state",
        response_model=DashboardStateResponse,
        tags=["runtime"],
        summary="Dashboard bootstrap state",
        description=(
            "Aggregate info, loaded-model, and usage snapshots for dashboard bootstrap "
            "with partial-failure warnings."
        ),
    )
    def dashboard_state(request: Request) -> DashboardStateResponse:
        info, info_warning = _call_provider(source="info", provider=info_provider)
        ps, ps_warning = _call_provider(source="ps", provider=ps_provider)
        usage, usage_warning = _call_request_provider(
            source="usage",
            provider=usage_provider,
            request=request,
        )
        warnings = [
            warning
            for warning in (info_warning, ps_warning, usage_warning)
            if warning is not None
        ]
        return DashboardStateResponse(
            info=info,
            ps=ps,
            usage=usage,
            warnings=warnings,
        )

    return router


def _call_provider(
    *,
    source: str,
    provider: DashboardProvider,
) -> tuple[dict[str, Any] | None, DashboardStateWarning | None]:
    try:
        payload = provider()
    except HTTPException as exc:
        return None, _warning_from_error(
            source=source,
            status_code=exc.status_code,
            detail=exc.detail,
        )
    except Exception as exc:  # noqa: BLE001
        return None, _warning_from_error(source=source, status_code=500, detail=str(exc))
    return _coerce_provider_payload(source=source, payload=payload)


def _call_request_provider(
    *,
    source: str,
    provider: DashboardRequestProvider,
    request: Request,
) -> tuple[dict[str, Any] | None, DashboardStateWarning | None]:
    try:
        payload = provider(request)
    except HTTPException as exc:
        return None, _warning_from_error(
            source=source,
            status_code=exc.status_code,
            detail=exc.detail,
        )
    except Exception as exc:  # noqa: BLE001
        return None, _warning_from_error(source=source, status_code=500, detail=str(exc))
    return _coerce_provider_payload(source=source, payload=payload)


def _coerce_provider_payload(
    *,
    source: str,
    payload: Any,
) -> tuple[dict[str, Any] | None, DashboardStateWarning | None]:
    if isinstance(payload, dict):
        return payload, None
    warning = DashboardStateWarning(
        source=_coerce_source(source),
        status_code=502,
        detail="provider returned non-object payload",
    )
    return None, warning


def _warning_from_error(*, source: str, status_code: int, detail: Any) -> DashboardStateWarning:
    normalized_status = status_code if 400 <= status_code <= 599 else 500
    detail_text = _detail_to_text(detail)
    if not detail_text:
        detail_text = "unexpected provider error"
    return DashboardStateWarning(
        source=_coerce_source(source),
        status_code=normalized_status,
        detail=detail_text,
    )


def _coerce_source(value: str) -> str:
    if value in {"info", "ps", "usage"}:
        return value
    return "info"


def _detail_to_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, (int, float, bool)):
        return str(detail)
    if detail is None:
        return ""
    try:
        return json.dumps(detail, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(detail)
