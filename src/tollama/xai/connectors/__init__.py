"""
tollama.xai.connectors — External data feed connector protocol, stubs, and live connectors.
"""

from tollama.xai.connectors.assembler import AssemblyResult, PayloadAssembler
from tollama.xai.connectors.helpers import (
    build_default_assembler,
    build_default_connector_registry,
)
from tollama.xai.connectors.live import (
    AsyncHttpFinancialConnector,
    AsyncHttpGeopoliticalConnector,
    AsyncHttpNewsConnector,
    AsyncHttpRegulatoryConnector,
    AsyncHttpSupplyChainConnector,
    HttpFinancialConnector,
    HttpGeopoliticalConnector,
    HttpNewsConnector,
    HttpRegulatoryConnector,
    HttpSupplyChainConnector,
)
from tollama.xai.connectors.protocol import (
    AsyncDataConnector,
    ConnectorError,
    ConnectorFetchError,
    ConnectorResult,
    DataConnector,
)
from tollama.xai.connectors.registry import ConnectorRegistry
from tollama.xai.connectors.stubs import (
    MockFinancialConnector,
    MockGeopoliticalConnector,
    MockNewsConnector,
    MockRegulatoryConnector,
    MockSupplyChainConnector,
)

__all__ = [
    "AssemblyResult",
    "AsyncDataConnector",
    "AsyncHttpFinancialConnector",
    "AsyncHttpGeopoliticalConnector",
    "AsyncHttpNewsConnector",
    "AsyncHttpRegulatoryConnector",
    "AsyncHttpSupplyChainConnector",
    "ConnectorError",
    "ConnectorFetchError",
    "ConnectorRegistry",
    "ConnectorResult",
    "DataConnector",
    "HttpFinancialConnector",
    "HttpGeopoliticalConnector",
    "HttpNewsConnector",
    "HttpRegulatoryConnector",
    "HttpSupplyChainConnector",
    "MockFinancialConnector",
    "MockGeopoliticalConnector",
    "MockNewsConnector",
    "MockRegulatoryConnector",
    "MockSupplyChainConnector",
    "PayloadAssembler",
    "build_default_assembler",
    "build_default_connector_registry",
]
