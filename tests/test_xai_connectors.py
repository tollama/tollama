"""Tests for tollama.xai.connectors — protocol, registry, assembler, stubs."""

from __future__ import annotations

from typing import Any

import pytest

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
from tollama.xai.trust_contract import (
    coerce_financial_payload,
    coerce_news_payload,
    coerce_supply_chain_payload,
)

# ── Protocol conformance ──────────────────────────────────────────────


class TestProtocolConformance:
    def test_mock_financial_satisfies_protocol(self):
        assert isinstance(MockFinancialConnector(), DataConnector)

    def test_mock_news_satisfies_protocol(self):
        assert isinstance(MockNewsConnector(), DataConnector)

    def test_mock_supply_chain_satisfies_protocol(self):
        assert isinstance(MockSupplyChainConnector(), DataConnector)

    def test_class_missing_fetch_does_not_satisfy_protocol(self):
        class BadConnector:
            connector_name = "bad"
            domain = "test"

            def supports(self, identifier: str, context: dict[str, Any]) -> bool:
                return True

        assert not isinstance(BadConnector(), DataConnector)


# ── ConnectorResult / ConnectorError validation ────────────────────


class TestConnectorResultValidation:
    def test_connector_result_required_fields(self):
        result = ConnectorResult(
            domain="test",
            payload={"key": "value"},
            source_id="src-1",
        )
        assert result.domain == "test"
        assert result.source_type == "unknown"
        assert result.freshness_seconds is None
        assert result.metadata == {}
        assert result.fetched_at  # auto-populated

    def test_connector_error_fields(self):
        err = ConnectorError(
            domain="financial_market",
            source_id="AAPL",
            error_type="network",
            message="Connection timeout",
            retryable=True,
        )
        assert err.error_type == "network"
        assert err.retryable is True

    def test_connector_fetch_error_wraps_error(self):
        inner = ConnectorError(domain="test", message="fail")
        exc = ConnectorFetchError(inner)
        assert exc.error is inner
        assert str(exc) == "fail"


# ── Registry ──────────────────────────────────────────────────────────


class TestConnectorRegistry:
    def test_register_and_get(self):
        registry = ConnectorRegistry()
        registry.register(MockFinancialConnector())
        connector = registry.get("financial_market", "AAPL")
        assert connector is not None
        assert connector.connector_name == "mock_financial"

    def test_returns_none_for_unknown_domain(self):
        registry = ConnectorRegistry()
        registry.register(MockFinancialConnector())
        assert registry.get("unknown_domain", "X") is None

    def test_rejects_invalid_connector(self):
        registry = ConnectorRegistry()

        class NoMethods:
            connector_name = "bad"
            domain = "test"

        with pytest.raises(ValueError, match="supports"):
            registry.register(NoMethods())

    def test_get_all_returns_multiple(self):
        registry = ConnectorRegistry()
        registry.register(MockFinancialConnector())
        # Register a second financial connector
        c2 = MockFinancialConnector()
        c2.connector_name = "mock_financial_2"
        registry.register(c2)
        all_fin = registry.get_all("financial_market")
        assert len(all_fin) == 2

    def test_connectors_property(self):
        registry = ConnectorRegistry()
        registry.register(MockFinancialConnector())
        registry.register(MockNewsConnector())
        assert len(registry.connectors) == 2


# ── Assembler ─────────────────────────────────────────────────────────


class TestPayloadAssembler:
    def test_assemble_financial_payload(self):
        assembler = build_default_assembler()
        result = assembler.assemble("financial_market", "AAPL")
        assert isinstance(result, AssemblyResult)
        assert result.connector_name == "mock_financial"
        assert result.trust_context["domain"] == "financial_market"
        # Verify payload is valid for FinancialTrustPayload
        payload = coerce_financial_payload(result.payload)
        assert payload.instrument_id == "AAPL"

    def test_assemble_news_payload(self):
        assembler = build_default_assembler()
        result = assembler.assemble("news", "breaking-story-123")
        payload = coerce_news_payload(result.payload)
        assert payload.story_id == "breaking-story-123"

    def test_assemble_supply_chain_payload(self):
        assembler = build_default_assembler()
        result = assembler.assemble("supply_chain", "SHIP-001")
        payload = coerce_supply_chain_payload(result.payload)
        assert payload.network_id == "SHIP-001"

    def test_evidence_has_freshness(self):
        assembler = build_default_assembler()
        result = assembler.assemble("financial_market", "MSFT")
        assert result.evidence.freshness_seconds == 5.0
        assert result.evidence.source_ids == ["MSFT"]
        assert result.evidence.source_type == "equity_market"

    def test_raises_on_unknown_domain(self):
        assembler = build_default_assembler()
        with pytest.raises(ConnectorFetchError) as exc_info:
            assembler.assemble("unknown_domain", "region-1")
        assert exc_info.value.error.error_type == "not_found"
        assert exc_info.value.error.domain == "unknown_domain"


# ── End-to-end ────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_assembled_payload_flows_through_trust_router(self):
        from tollama.xai.trust_router import build_default_trust_router

        assembler = build_default_assembler()
        assembly = assembler.assemble("financial_market", "TSLA")

        router = build_default_trust_router()
        trust_result = router.analyze(
            context=assembly.trust_context,
            payload=assembly.payload,
        )
        assert trust_result is not None
        assert 0.0 <= trust_result.trust_score <= 1.0
        assert trust_result.agent_name == "financial_market"


# ── Error handling ────────────────────────────────────────────────────


class TestErrorHandling:
    def test_failing_connector_raises_structured_error(self):
        class FailingConnector:
            connector_name = "failing"
            domain = "financial_market"

            def supports(self, identifier: str, context: dict[str, Any]) -> bool:
                return True

            def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
                raise ConnectorFetchError(
                    ConnectorError(
                        domain=self.domain,
                        source_id=identifier,
                        error_type="network",
                        message="API timeout",
                        retryable=True,
                    )
                )

        registry = ConnectorRegistry()
        registry.register(FailingConnector())
        assembler = PayloadAssembler(registry)

        with pytest.raises(ConnectorFetchError) as exc_info:
            assembler.assemble("financial_market", "AAPL")
        assert exc_info.value.error.error_type == "network"
        assert exc_info.value.error.retryable is True


# ── Factory helpers ───────────────────────────────────────────────────


class TestFactoryHelpers:
    def test_build_default_registry_has_all_domains(self):
        registry = build_default_connector_registry()
        assert registry.get("financial_market", "X") is not None
        assert registry.get("news", "X") is not None
        assert registry.get("supply_chain", "X") is not None

    def test_build_default_assembler_works(self):
        assembler = build_default_assembler()
        result = assembler.assemble("news", "test-story")
        assert result.connector_name == "mock_news"
