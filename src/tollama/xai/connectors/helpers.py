"""
tollama.xai.connectors.helpers — Factory functions for connector setup.
"""

from __future__ import annotations

from tollama.xai.connectors.assembler import PayloadAssembler
from tollama.xai.connectors.registry import ConnectorRegistry
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


__all__ = ["build_default_assembler", "build_default_connector_registry"]
