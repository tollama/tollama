"""
tollama.xai.connectors.helpers — Factory functions for connector setup.
"""

from __future__ import annotations

from tollama.xai.connectors.assembler import AsyncPayloadAssembler, PayloadAssembler
from tollama.xai.connectors.registry import AsyncConnectorRegistry, ConnectorRegistry
from tollama.xai.connectors.stubs import (
    MockFinancialConnector,
    MockGeopoliticalConnector,
    MockNewsConnector,
    MockRegulatoryConnector,
    MockSupplyChainConnector,
)


def build_default_connector_registry() -> ConnectorRegistry:
    """Build a registry with all mock connectors registered."""
    registry = ConnectorRegistry()
    registry.register(MockFinancialConnector())
    registry.register(MockNewsConnector())
    registry.register(MockSupplyChainConnector())
    registry.register(MockGeopoliticalConnector())
    registry.register(MockRegulatoryConnector())
    return registry


def build_default_assembler() -> PayloadAssembler:
    """Build an assembler with the default mock connector registry."""
    return PayloadAssembler(build_default_connector_registry())


def build_default_async_connector_registry() -> AsyncConnectorRegistry:
    """Build an async registry with all async HTTP connectors registered."""
    from tollama.xai.connectors.live import (
        AsyncHttpFinancialConnector,
        AsyncHttpGeopoliticalConnector,
        AsyncHttpNewsConnector,
        AsyncHttpRegulatoryConnector,
        AsyncHttpSupplyChainConnector,
    )

    registry = AsyncConnectorRegistry()
    registry.register(AsyncHttpFinancialConnector(base_url="http://localhost:8080"))
    registry.register(AsyncHttpNewsConnector(base_url="http://localhost:8080"))
    registry.register(AsyncHttpSupplyChainConnector(base_url="http://localhost:8080"))
    registry.register(AsyncHttpGeopoliticalConnector(base_url="http://localhost:8080"))
    registry.register(AsyncHttpRegulatoryConnector(base_url="http://localhost:8080"))
    return registry


def build_default_async_assembler() -> AsyncPayloadAssembler:
    """Build an async assembler with the default async connector registry."""
    return AsyncPayloadAssembler(build_default_async_connector_registry())


__all__ = [
    "build_default_assembler",
    "build_default_async_assembler",
    "build_default_async_connector_registry",
    "build_default_connector_registry",
]
