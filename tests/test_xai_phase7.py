"""Tests for Phase 7 features — async registry/assembler, calibration auto-persist,
live geo/reg connectors, CLI XAI commands.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tollama.xai.connectors.assembler import AsyncPayloadAssembler
from tollama.xai.connectors.helpers import build_default_async_connector_registry
from tollama.xai.connectors.live import (
    AsyncHttpFinancialConnector,
    AsyncHttpGeopoliticalConnector,
    AsyncHttpNewsConnector,
    AsyncHttpRegulatoryConnector,
    AsyncHttpSupplyChainConnector,
    HttpGeopoliticalConnector,
    HttpRegulatoryConnector,
)
from tollama.xai.connectors.protocol import (
    AsyncDataConnector,
    ConnectorFetchError,
    ConnectorResult,
    DataConnector,
)
from tollama.xai.connectors.registry import AsyncConnectorRegistry
from tollama.xai.trust_agents.calibration import (
    CalibrationTracker,
    default_calibration_path,
)
from tollama.xai.trust_router import TrustRouter, build_default_trust_router

# ── AsyncConnectorRegistry ──────────────────────────────────────────


class TestAsyncConnectorRegistry:
    def test_register_and_get(self):
        registry = AsyncConnectorRegistry()
        connector = AsyncHttpFinancialConnector(base_url="http://test:8080")
        registry.register(connector)
        found = registry.get("financial_market", "AAPL")
        assert found is not None
        assert found.connector_name == "async_http_financial"

    def test_get_returns_none_for_unknown_domain(self):
        registry = AsyncConnectorRegistry()
        registry.register(AsyncHttpFinancialConnector(base_url="http://test:8080"))
        assert registry.get("unknown_domain", "X") is None

    def test_get_all(self):
        registry = AsyncConnectorRegistry()
        registry.register(AsyncHttpFinancialConnector(base_url="http://test:8080"))
        registry.register(AsyncHttpNewsConnector(base_url="http://test:8080"))
        assert len(registry.get_all("financial_market")) == 1
        assert len(registry.get_all("news")) == 1
        assert len(registry.get_all("missing")) == 0

    def test_connectors_property(self):
        registry = AsyncConnectorRegistry()
        assert registry.connectors == []
        registry.register(AsyncHttpFinancialConnector(base_url="http://test:8080"))
        assert len(registry.connectors) == 1

    def test_register_rejects_invalid_connector(self):
        registry = AsyncConnectorRegistry()
        with pytest.raises(ValueError, match="connector_name"):
            registry.register(object())  # type: ignore[arg-type]

    def test_register_rejects_no_supports(self):
        class BadConnector:
            connector_name = "bad"
            domain = "test"

            async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
                pass  # pragma: no cover

        registry = AsyncConnectorRegistry()
        with pytest.raises(ValueError, match="supports"):
            registry.register(BadConnector())  # type: ignore[arg-type]


class TestBuildDefaultAsyncRegistry:
    def test_builds_with_5_connectors(self):
        registry = build_default_async_connector_registry()
        assert len(registry.connectors) == 5
        domains = {c.domain for c in registry.connectors}
        assert domains == {"financial_market", "news", "supply_chain", "geopolitical", "regulatory"}


# ── AsyncPayloadAssembler ──────────────────────────────────────────


class _FakeAsyncConnector:
    connector_name = "fake_async"
    domain = "test_domain"

    def supports(self, identifier: str, context: dict[str, Any]) -> bool:
        return identifier == "test-id"

    async def fetch(self, identifier: str, context: dict[str, Any]) -> ConnectorResult:
        return ConnectorResult(
            domain=self.domain,
            payload={"key": "value"},
            source_id=identifier,
            source_type="test_source",
        )


class TestAsyncPayloadAssembler:
    def test_assemble_success(self):
        registry = AsyncConnectorRegistry()
        registry.register(_FakeAsyncConnector())  # type: ignore[arg-type]
        assembler = AsyncPayloadAssembler(registry)
        result = asyncio.run(
            assembler.assemble("test_domain", "test-id")
        )
        assert result.payload == {"key": "value"}
        assert result.connector_name == "fake_async"
        assert result.trust_context["domain"] == "test_domain"

    def test_assemble_no_connector_raises(self):
        registry = AsyncConnectorRegistry()
        assembler = AsyncPayloadAssembler(registry)
        with pytest.raises(ConnectorFetchError):
            asyncio.run(
                assembler.assemble("missing", "x")
            )


# ── Async Connector Protocol Conformance ──────────────────────────


class TestAsyncProtocolConformance:
    def test_async_financial_satisfies_protocol(self):
        assert isinstance(AsyncHttpFinancialConnector(base_url="http://test"), AsyncDataConnector)

    def test_async_news_satisfies_protocol(self):
        assert isinstance(AsyncHttpNewsConnector(base_url="http://test"), AsyncDataConnector)

    def test_async_supply_chain_satisfies_protocol(self):
        assert isinstance(AsyncHttpSupplyChainConnector(base_url="http://test"), AsyncDataConnector)

    def test_async_geopolitical_satisfies_protocol(self):
        conn = AsyncHttpGeopoliticalConnector(base_url="http://test")
        assert isinstance(conn, AsyncDataConnector)

    def test_async_regulatory_satisfies_protocol(self):
        assert isinstance(AsyncHttpRegulatoryConnector(base_url="http://test"), AsyncDataConnector)


# ── Live Geopolitical + Regulatory Connectors ──────────────────────


class TestHttpGeopoliticalConnector:
    def test_satisfies_protocol(self):
        c = HttpGeopoliticalConnector(base_url="http://test")
        assert isinstance(c, DataConnector)
        assert c.domain == "geopolitical"
        assert c.connector_name == "http_geopolitical"

    def test_supports_returns_true(self):
        c = HttpGeopoliticalConnector(base_url="http://test")
        assert c.supports("us", {}) is True


class TestHttpRegulatoryConnector:
    def test_satisfies_protocol(self):
        c = HttpRegulatoryConnector(base_url="http://test")
        assert isinstance(c, DataConnector)
        assert c.domain == "regulatory"
        assert c.connector_name == "http_regulatory"

    def test_supports_returns_true(self):
        c = HttpRegulatoryConnector(base_url="http://test")
        assert c.supports("eu", {}) is True


# ── CalibrationTracker Persistence ─────────────────────────────────


class TestCalibrationPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        tracker = CalibrationTracker()
        for i in range(5):
            tracker.record(
                agent_name="test_agent",
                domain="test",
                predicted_score=0.7,
                actual_outcome=0.6 + i * 0.02,
                component_scores={"metric_a": 0.8},
            )

        path = tmp_path / "calibration.json"
        tracker.save(path)
        assert path.is_file()

        loaded = CalibrationTracker.load(path)
        stats = loaded.get_calibration_stats("test_agent")
        assert stats.record_count == 5
        assert stats.agent_name == "test_agent"

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        tracker = CalibrationTracker.load(tmp_path / "nonexistent.json")
        assert tracker.agents == []

    def test_save_atomic_write(self, tmp_path: Path):
        tracker = CalibrationTracker()
        tracker.record("a", "d", 0.5, 0.5, {})
        path = tmp_path / "cal.json"
        tracker.save(path)
        # .tmp should not remain
        assert not (tmp_path / "cal.json.tmp").exists()
        assert path.exists()

    def test_default_calibration_path(self):
        path = default_calibration_path()
        assert str(path).endswith("calibration.json")
        assert "xai" in str(path)

    def test_window_size_respected(self, tmp_path: Path):
        tracker = CalibrationTracker(window_size=3)
        for i in range(10):
            tracker.record("a", "d", 0.5, 0.5, {})
        path = tmp_path / "cal.json"
        tracker.save(path)

        loaded = CalibrationTracker.load(path, window_size=3)
        stats = loaded.get_calibration_stats("a")
        assert stats.record_count == 3


# ── TrustRouter Calibration Integration ────────────────────────────


class TestTrustRouterCalibration:
    def test_build_default_with_calibration(self):
        router = build_default_trust_router(enable_calibration=True)
        assert router.calibration_tracker is not None
        assert router._calibration_path is not None

    def test_build_default_without_calibration(self):
        router = build_default_trust_router(enable_calibration=False)
        assert router.calibration_tracker is None
        assert router._calibration_path is None

    def test_record_outcome(self, tmp_path: Path):
        from tollama.xai.trust_agents.calibration import CalibrationTracker
        from tollama.xai.trust_router import TrustAgentRegistry

        tracker = CalibrationTracker()
        registry = TrustAgentRegistry()
        router = TrustRouter(
            registry,
            calibration_tracker=tracker,
            calibration_path=tmp_path / "cal.json",
            auto_persist_every=2,
        )
        router.record_outcome("agent_a", "test", 0.7, 0.6, {"m": 0.8})
        stats = tracker.get_calibration_stats("agent_a")
        assert stats.record_count == 1

    def test_auto_persist(self, tmp_path: Path):
        from tollama.xai.trust_agents.calibration import CalibrationTracker
        from tollama.xai.trust_router import TrustAgentRegistry

        cal_path = tmp_path / "cal.json"
        tracker = CalibrationTracker()
        registry = TrustAgentRegistry()
        router = TrustRouter(
            registry,
            calibration_tracker=tracker,
            calibration_path=cal_path,
            auto_persist_every=3,
        )
        # _maybe_auto_persist increments _analyze_count each call;
        # persists when _analyze_count % auto_persist_every == 0
        router.record_outcome("a", "d", 0.5, 0.5, {})  # count=1
        router.record_outcome("a", "d", 0.6, 0.6, {})  # count=2
        assert not cal_path.exists()
        router.record_outcome("a", "d", 0.7, 0.7, {})  # count=3 → persist
        assert cal_path.exists()

    def test_persist_calibration_manual(self, tmp_path: Path):
        from tollama.xai.trust_agents.calibration import CalibrationTracker
        from tollama.xai.trust_router import TrustAgentRegistry

        cal_path = tmp_path / "cal.json"
        tracker = CalibrationTracker()
        tracker.record("a", "d", 0.5, 0.5, {})
        registry = TrustAgentRegistry()
        router = TrustRouter(
            registry,
            calibration_tracker=tracker,
            calibration_path=cal_path,
        )
        router.persist_calibration()
        assert cal_path.exists()

    def test_persist_calibration_no_tracker(self):
        from tollama.xai.trust_router import TrustAgentRegistry

        registry = TrustAgentRegistry()
        router = TrustRouter(registry)
        # Should be a no-op, no error
        router.persist_calibration()


# ── Re-exports ─────────────────────────────────────────────────────


class TestReexports:
    def test_async_connector_registry_from_package(self):
        from tollama.xai.connectors import AsyncConnectorRegistry
        assert AsyncConnectorRegistry is not None

    def test_async_payload_assembler_from_package(self):
        from tollama.xai.connectors import AsyncPayloadAssembler
        assert AsyncPayloadAssembler is not None

    def test_build_helpers_from_package(self):
        from tollama.xai.connectors import (
            build_default_async_assembler,
            build_default_async_connector_registry,
        )
        assert build_default_async_assembler is not None
        assert build_default_async_connector_registry is not None
