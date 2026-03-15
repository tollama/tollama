"""
tollama.xai.connectors.stubs — Mock connector implementations for testing and development.
"""

from __future__ import annotations

from typing import Any

from tollama.xai.connectors.protocol import ConnectorResult


class MockFinancialConnector:
    """Mock connector producing FinancialTrustPayload-compatible data."""

    connector_name = "mock_financial"
    domain = "financial_market"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={
                "instrument_id": identifier,
                "liquidity_depth": 0.75,
                "bid_ask_spread_bps": 15.0,
                "realized_volatility": 0.25,
                "execution_risk": 0.3,
                "data_freshness": 0.9,
            },
            source_id=identifier,
            source_type="equity_market",
            freshness_seconds=5.0,
            metadata={"connector": "mock_financial", "version": "0.1.0"},
        )


class MockNewsConnector:
    """Mock connector producing NewsTrustPayload-compatible data."""

    connector_name = "mock_news"
    domain = "news"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={
                "story_id": identifier,
                "source_credibility": 0.8,
                "corroboration": 0.7,
                "contradiction_score": 0.15,
                "propagation_delay_seconds": 120.0,
                "freshness_score": 0.85,
            },
            source_id=identifier,
            source_type="news_feed",
            freshness_seconds=120.0,
            metadata={"connector": "mock_news", "version": "0.1.0"},
        )


class MockSupplyChainConnector:
    """Mock connector producing SupplyChainTrustPayload-compatible data."""

    connector_name = "mock_supply_chain"
    domain = "supply_chain"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={
                "network_id": identifier,
                "lead_time_reliability": 0.8,
                "inventory_visibility": 0.7,
                "disruption_risk": 0.25,
                "sensor_quality": 0.85,
                "data_freshness": 0.75,
            },
            source_id=identifier,
            source_type="supply_chain_iot",
            freshness_seconds=300.0,
            metadata={"connector": "mock_supply_chain", "version": "0.1.0"},
        )


class MockGeopoliticalConnector:
    """Mock connector producing GeopoliticalTrustPayload-compatible data."""

    connector_name = "mock_geopolitical"
    domain = "geopolitical"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={
                "region_id": identifier,
                "political_stability": 0.7,
                "sanctions_exposure": 0.2,
                "conflict_proximity": 0.15,
                "regulatory_alignment": 0.8,
                "data_freshness": 0.85,
            },
            source_id=identifier,
            source_type="country_risk",
            freshness_seconds=600.0,
            metadata={"connector": "mock_geopolitical", "version": "0.1.0"},
        )


class MockRegulatoryConnector:
    """Mock connector producing RegulatoryTrustPayload-compatible data."""

    connector_name = "mock_regulatory"
    domain = "regulatory"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return True

    def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={
                "jurisdiction_id": identifier,
                "compliance_score": 0.85,
                "enforcement_risk": 0.2,
                "reporting_quality": 0.75,
                "audit_recency": 0.8,
                "data_freshness": 0.7,
            },
            source_id=identifier,
            source_type="compliance",
            freshness_seconds=3600.0,
            metadata={"connector": "mock_regulatory", "version": "0.1.0"},
        )


__all__ = [
    "MockFinancialConnector",
    "MockGeopoliticalConnector",
    "MockNewsConnector",
    "MockRegulatoryConnector",
    "MockSupplyChainConnector",
]
