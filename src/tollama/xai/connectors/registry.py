"""
tollama.xai.connectors.registry — Registry for data feed connectors.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.connectors.protocol import DataConnector


class ConnectorRegistry:
    """In-memory registry for data connectors."""

    def __init__(self) -> None:
        self._connectors: list[DataConnector] = []

    def register(self, connector: DataConnector) -> None:
        if not hasattr(connector, "connector_name") or not hasattr(connector, "domain"):
            raise ValueError("Connector must define connector_name and domain")
        if not callable(getattr(connector, "supports", None)):
            raise ValueError("Connector must implement supports(identifier, context)")
        if not callable(getattr(connector, "fetch", None)):
            raise ValueError("Connector must implement fetch(identifier, context)")
        self._connectors.append(connector)

    def get(
        self,
        domain: str,
        identifier: str,
        context: dict[str, Any] | None = None,
    ) -> DataConnector | None:
        """Return the first connector matching domain that supports the identifier."""
        ctx = context or {}
        for connector in self._connectors:
            if connector.domain == domain and connector.supports(identifier, ctx):
                return connector
        return None

    def get_all(self, domain: str) -> list[DataConnector]:
        """Return all connectors for a domain."""
        return [c for c in self._connectors if c.domain == domain]

    @property
    def connectors(self) -> list[DataConnector]:
        return list(self._connectors)


__all__ = ["ConnectorRegistry"]
