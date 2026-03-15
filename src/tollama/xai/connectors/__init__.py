"""
tollama.xai.connectors — External data feed connector protocol and stubs.
"""

from tollama.xai.connectors.assembler import AssemblyResult, PayloadAssembler
from tollama.xai.connectors.helpers import (
    build_default_assembler,
    build_default_connector_registry,
)
from tollama.xai.connectors.protocol import (
    ConnectorError,
    ConnectorFetchError,
    ConnectorResult,
    DataConnector,
)
from tollama.xai.connectors.registry import ConnectorRegistry
from tollama.xai.connectors.stubs import (
    MockFinancialConnector,
    MockNewsConnector,
    MockSupplyChainConnector,
)

__all__ = [
    "AssemblyResult",
    "ConnectorError",
    "ConnectorFetchError",
    "ConnectorRegistry",
    "ConnectorResult",
    "DataConnector",
    "MockFinancialConnector",
    "MockNewsConnector",
    "MockSupplyChainConnector",
    "PayloadAssembler",
    "build_default_assembler",
    "build_default_connector_registry",
]
