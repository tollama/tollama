"""
tollama.xai.connectors.versions — Version compatibility matrix for federation.

Tracks the ``audit.formula_version`` and ``audit.agent_version`` fields
across the federated agent ecosystem so that breaking changes can be
detected before they cause silent data corruption.

Usage::

    from tollama.xai.connectors.versions import check_compatibility

    compat = check_compatibility(
        agent_name="news_agent",
        formula_version="news-v2",
        agent_version="0.2.0",
    )
    if not compat.compatible:
        log.warning("Incompatible agent version: %s", compat.message)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    """Result of a version compatibility check."""

    compatible: bool
    message: str
    agent_name: str
    formula_version: str
    agent_version: str


# ── known formula versions per agent ──────────────────────────

# Maps agent_name → set of formula_versions this tollama release supports.
# When an agent bumps its formula_version in a breaking way, add the new
# version here *and* keep the old one during the backward-compatible period.
SUPPORTED_FORMULA_VERSIONS: dict[str, frozenset[str]] = {
    "news_agent": frozenset({
        "news-v1",
        "news-v2",
    }),
    "financial_agent": frozenset({
        "financial-v1",
        "financial-v2",
    }),
    "market_calibration": frozenset({
        "mc-v1",
        "mc-v2",
    }),
    # Degraded results produced by the router itself
    "_degraded": frozenset({
        "degraded-v1",
    }),
    # Multi-agent aggregate results
    "_aggregate": frozenset({
        "aggregate-v1",
    }),
}

# Minimum agent_version tollama requires for each agent.
# Agents below this version may return incompatible schemas.
MIN_AGENT_VERSIONS: dict[str, str] = {
    "news_agent": "0.1.0",
    "financial_agent": "0.1.0",
    "market_calibration": "0.1.0",
}


def _parse_semver(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into a tuple of ints."""
    parts: list[int] = []
    for segment in version.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def check_compatibility(
    *,
    agent_name: str,
    formula_version: str,
    agent_version: str,
) -> CompatibilityResult:
    """Check whether an agent's version info is compatible.

    Returns a ``CompatibilityResult`` with ``compatible=True`` when:
    1. The ``formula_version`` is in the supported set (or agent is unknown).
    2. The ``agent_version`` meets the minimum requirement.

    Unknown agents are always considered compatible (forward-compatible).
    """
    # Check formula version
    supported = SUPPORTED_FORMULA_VERSIONS.get(agent_name)
    if supported is not None and formula_version not in supported:
        return CompatibilityResult(
            compatible=False,
            message=(
                f"Agent {agent_name!r} returned unsupported "
                f"formula_version={formula_version!r}. "
                f"Supported: {sorted(supported)}"
            ),
            agent_name=agent_name,
            formula_version=formula_version,
            agent_version=agent_version,
        )

    # Check minimum agent version
    min_ver = MIN_AGENT_VERSIONS.get(agent_name)
    if min_ver is not None:
        if _parse_semver(agent_version) < _parse_semver(min_ver):
            return CompatibilityResult(
                compatible=False,
                message=(
                    f"Agent {agent_name!r} version {agent_version!r} "
                    f"is below minimum {min_ver!r}."
                ),
                agent_name=agent_name,
                formula_version=formula_version,
                agent_version=agent_version,
            )

    return CompatibilityResult(
        compatible=True,
        message="OK",
        agent_name=agent_name,
        formula_version=formula_version,
        agent_version=agent_version,
    )


__all__ = [
    "MIN_AGENT_VERSIONS",
    "SUPPORTED_FORMULA_VERSIONS",
    "CompatibilityResult",
    "check_compatibility",
]
