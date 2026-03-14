"""Role-Based Access Control (RBAC) middleware for Tollama daemon.

Provides enterprise-grade authorization with predefined roles:
- admin: Full access to all endpoints including configuration
- analyst: Read/write access to forecasts, analysis, XAI
- auditor: Read-only access to audit trails and compliance reports
- viewer: Read-only access to forecasts and dashboards

Roles are configured via TOLLAMA_RBAC_POLICY env var (JSON file path)
or tollama.toml ``[auth.rbac]`` section.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

_RBAC_POLICY_ENV = "TOLLAMA_RBAC_POLICY"


class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    VIEWER = "viewer"


# Default permission matrix: role → set of allowed path prefixes
_DEFAULT_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"*"},
    Role.ANALYST: {
        "/api/forecast",
        "/api/auto-forecast",
        "/api/compare",
        "/api/analyze",
        "/api/what-if",
        "/api/counterfactual",
        "/api/pipeline",
        "/api/report",
        "/api/explain",
        "/api/xai",
        "/v1/forecast",
        "/api/dashboard",
        "/docs",
    },
    Role.AUDITOR: {
        "/api/xai",
        "/api/dashboard",
        "/docs",
    },
    Role.VIEWER: {
        "/api/forecast",
        "/api/dashboard",
        "/docs",
    },
}

# Methods allowed per role (beyond GET which is always allowed for
# permitted paths)
_DEFAULT_WRITE_ROLES: set[Role] = {Role.ADMIN, Role.ANALYST}


@dataclass(frozen=True)
class RBACPolicy:
    """RBAC policy mapping API keys to roles with path permissions."""

    key_roles: dict[str, Role] = field(default_factory=dict)
    role_permissions: dict[Role, set[str]] = field(
        default_factory=lambda: dict(_DEFAULT_PERMISSIONS)
    )
    write_roles: set[Role] = field(
        default_factory=lambda: set(_DEFAULT_WRITE_ROLES)
    )


def load_rbac_policy(
    *,
    policy_path: str | Path | None = None,
) -> RBACPolicy | None:
    """Load RBAC policy from file or environment.

    Parameters
    ----------
    policy_path : str or Path, optional
        Explicit path to RBAC policy JSON. Falls back to
        ``TOLLAMA_RBAC_POLICY`` env var.

    Returns
    -------
    RBACPolicy or None
        Policy if configured, None if RBAC is disabled.
    """
    path = policy_path
    if path is None:
        env_path = os.environ.get(_RBAC_POLICY_ENV, "").strip()
        if env_path:
            path = env_path

    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        logger.warning("RBAC policy file not found: %s", path)
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load RBAC policy: %s", exc)
        return None

    return _parse_policy(data)


def _parse_policy(data: dict[str, Any]) -> RBACPolicy:
    """Parse a raw policy dict into an RBACPolicy."""
    key_roles: dict[str, Role] = {}
    for key_id, role_str in data.get("key_roles", {}).items():
        try:
            key_roles[key_id] = Role(role_str)
        except ValueError:
            logger.warning("Unknown role %r for key %s, skipping", role_str, key_id)

    # Custom permissions override defaults
    role_permissions = dict(_DEFAULT_PERMISSIONS)
    for role_str, paths in data.get("role_permissions", {}).items():
        try:
            role = Role(role_str)
            role_permissions[role] = set(paths)
        except ValueError:
            logger.warning("Unknown role %r in permissions, skipping", role_str)

    write_roles = set(_DEFAULT_WRITE_ROLES)
    if "write_roles" in data:
        write_roles = set()
        for role_str in data["write_roles"]:
            try:
                write_roles.add(Role(role_str))
            except ValueError:
                pass

    return RBACPolicy(
        key_roles=key_roles,
        role_permissions=role_permissions,
        write_roles=write_roles,
    )


def check_rbac(
    request: Request,
    policy: RBACPolicy,
) -> None:
    """Check RBAC authorization for the current request.

    Parameters
    ----------
    request : Request
        FastAPI request with ``auth_principal`` on state.
    policy : RBACPolicy
        Active RBAC policy.

    Raises
    ------
    HTTPException
        403 if the principal's role does not permit the request.
    """
    from tollama.daemon.auth import AuthPrincipal

    principal = getattr(request.state, "auth_principal", None)
    if principal is None:
        # Anonymous access: check if viewer-level is sufficient
        role = Role.VIEWER
    elif isinstance(principal, AuthPrincipal):
        role = policy.key_roles.get(principal.key_id, Role.VIEWER)
    else:
        role = Role.VIEWER

    # Check path permissions
    allowed_paths = policy.role_permissions.get(role, set())
    if "*" not in allowed_paths:
        path = request.url.path
        if not any(path.startswith(prefix) for prefix in allowed_paths):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role.value}' does not have access to {path}",
            )

    # Check write permission for mutating methods
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        if role not in policy.write_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role.value}' does not have write access",
            )
