"""
tollama.xai.connectors.registry — Registry for data feed connectors.
"""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar

from tollama.xai.connectors.protocol import AsyncDataConnector, DataConnector


class _RegisteredConnector(Protocol):
    connector_name: str
    domain: str

    def supports(self, identifier: str, context: dict[str, Any]) -> bool: ...


ConnectorT = TypeVar("ConnectorT", bound=_RegisteredConnector)


class _ConnectorRegistryBase(Generic[ConnectorT]):
    def __init__(self) -> None:
        self._connectors: list[ConnectorT] = []

    def register(self, connector: ConnectorT) -> None:
        self._validate_connector(connector)
        self._connectors.append(connector)

    @staticmethod
    def _validate_connector(connector: Any) -> None:
        if not hasattr(connector, "connector_name") or not hasattr(connector, "domain"):
            raise ValueError("Connector must define connector_name and domain")
        if not callable(getattr(connector, "supports", None)):
            raise ValueError("Connector must implement supports(identifier, context)")
        if not callable(getattr(connector, "fetch", None)):
            raise ValueError("Connector must implement fetch(identifier, context)")

    def get(
        self,
        domain: str,
        identifier: str,
        context: dict[str, Any] | None = None,
    ) -> ConnectorT | None:
        """Return the first connector matching domain that supports the identifier."""
        ctx = context or {}
        for connector in self._connectors:
            if connector.domain == domain and connector.supports(identifier, ctx):
                return connector
        return None

    def get_all(self, domain: str) -> list[ConnectorT]:
        """Return all connectors for a domain."""
        return [connector for connector in self._connectors if connector.domain == domain]

    @property
    def connectors(self) -> list[ConnectorT]:
        return list(self._connectors)


class ConnectorRegistry(_ConnectorRegistryBase[DataConnector]):
    """In-memory registry for data connectors."""


class AsyncConnectorRegistry(_ConnectorRegistryBase[AsyncDataConnector]):
    """In-memory registry for async data connectors."""


__all__ = ["AsyncConnectorRegistry", "ConnectorRegistry"]
