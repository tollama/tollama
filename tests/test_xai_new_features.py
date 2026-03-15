"""Tests for Phase 5: new domain agents, live connectors, learned calibration."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tollama.xai.connectors.helpers import build_default_connector_registry
from tollama.xai.connectors.protocol import ConnectorFetchError, DataConnector
from tollama.xai.connectors.stubs import MockGeopoliticalConnector, MockRegulatoryConnector
from tollama.xai.trust_agents.calibration import (
    CalibrationRecord,
    CalibrationStats,
    CalibrationTracker,
)
from tollama.xai.trust_agents.heuristic import (
    GeopoliticalTrustAgent,
    RegulatoryTrustAgent,
    _build_result,
)
from tollama.xai.trust_contract import (
    GeopoliticalTrustPayload,
    RegulatoryTrustPayload,
    TrustComponent,
    coerce_geopolitical_payload,
    coerce_regulatory_payload,
)
from tollama.xai.trust_router import build_default_trust_router


# ── Geopolitical Agent Tests ──────────────────────────────────────────


class TestGeopoliticalPayload:
    def test_alias_normalization(self):
        p1 = coerce_geopolitical_payload({"country_code": "US", "political_stability": 0.9})
        assert p1.region_id == "US"
        p2 = coerce_geopolitical_payload({"geo_zone": "EU-WEST"})
        assert p2.region_id == "EU-WEST"

    def test_requires_region_id(self):
        with pytest.raises(ValidationError):
            coerce_geopolitical_payload({"political_stability": 0.9})


class TestGeopoliticalAgent:
    def test_green_for_stable_region(self):
        agent = GeopoliticalTrustAgent()
        result = agent.analyze({
            "region_id": "CH",
            "political_stability": 0.9,
            "sanctions_exposure": 0.05,
            "conflict_proximity": 0.05,
            "regulatory_alignment": 0.9,
            "data_freshness": 0.95,
        })
        assert result.risk_category == "GREEN"
        assert result.trust_score >= 0.75
        assert not result.violations

    def test_red_for_sanctioned_conflict_zone(self):
        agent = GeopoliticalTrustAgent()
        result = agent.analyze({
            "region_id": "CONFLICT-ZONE",
            "political_stability": 0.1,
            "sanctions_exposure": 0.85,
            "conflict_proximity": 0.85,
            "regulatory_alignment": 0.2,
            "data_freshness": 0.5,
        })
        assert result.risk_category == "RED"
        critical_names = [v.name for v in result.violations if v.severity == "critical"]
        assert "sanctions_exposure_extreme" in critical_names
        assert "conflict_proximity_extreme" in critical_names
        assert "political_stability_critical" in critical_names


# ── Regulatory Agent Tests ────────────────────────────────────────────


class TestRegulatoryPayload:
    def test_alias_normalization(self):
        p1 = coerce_regulatory_payload({"regulation_id": "GDPR", "compliance_score": 0.9})
        assert p1.jurisdiction_id == "GDPR"
        p2 = coerce_regulatory_payload({"compliance_scope": "SOX"})
        assert p2.jurisdiction_id == "SOX"

    def test_requires_jurisdiction_id(self):
        with pytest.raises(ValidationError):
            coerce_regulatory_payload({"compliance_score": 0.9})


class TestRegulatoryAgent:
    def test_green_for_compliant_entity(self):
        agent = RegulatoryTrustAgent()
        result = agent.analyze({
            "jurisdiction_id": "SEC-COMPLIANT",
            "compliance_score": 0.9,
            "enforcement_risk": 0.1,
            "reporting_quality": 0.85,
            "audit_recency": 0.9,
            "data_freshness": 0.95,
        })
        assert result.risk_category == "GREEN"
        assert result.trust_score >= 0.75
        assert not result.violations

    def test_red_for_noncompliant_entity(self):
        agent = RegulatoryTrustAgent()
        result = agent.analyze({
            "jurisdiction_id": "NONCOMPLIANT",
            "compliance_score": 0.1,
            "enforcement_risk": 0.9,
            "reporting_quality": 0.15,
            "audit_recency": 0.05,
            "data_freshness": 0.5,
        })
        assert result.risk_category == "RED"
        critical_names = [v.name for v in result.violations if v.severity == "critical"]
        assert "compliance_critical" in critical_names
        assert "enforcement_risk_extreme" in critical_names
        assert "reporting_quality_critical" in critical_names
        assert "audit_recency_stale" in critical_names


# ── Router Integration ────────────────────────────────────────────────


class TestRouterNewAgents:
    def test_selects_geopolitical_agent(self):
        router = build_default_trust_router()
        agent = router.select_agent({"domain": "geopolitical"})
        assert agent is not None
        assert agent.agent_name == "geopolitical"

    def test_selects_regulatory_agent(self):
        router = build_default_trust_router()
        agent = router.select_agent({"domain": "regulatory"})
        assert agent is not None
        assert agent.agent_name == "regulatory"


# ── Connector Protocol Tests ─────────────────────────────────────────


class TestNewConnectors:
    def test_mock_geopolitical_satisfies_protocol(self):
        assert isinstance(MockGeopoliticalConnector(), DataConnector)

    def test_mock_regulatory_satisfies_protocol(self):
        assert isinstance(MockRegulatoryConnector(), DataConnector)

    def test_default_registry_includes_new_domains(self):
        registry = build_default_connector_registry()
        assert registry.get("geopolitical", "US") is not None
        assert registry.get("regulatory", "GDPR") is not None

    def test_http_financial_timeout_error(self):
        from tollama.xai.connectors.live import HttpFinancialConnector

        # Use an unreachable address to trigger timeout
        connector = HttpFinancialConnector(
            base_url="http://192.0.2.1:1",  # TEST-NET, guaranteed unreachable
            timeout=0.1,
        )
        with pytest.raises(ConnectorFetchError) as exc_info:
            connector.fetch("AAPL", {})
        assert exc_info.value.error.error_type == "network"
        assert exc_info.value.error.retryable is True


# ── Calibration Tests ─────────────────────────────────────────────────


class TestCalibrationRecord:
    def test_creation(self):
        rec = CalibrationRecord(
            agent_name="financial_market",
            domain="financial_market",
            predicted_score=0.8,
            actual_outcome=0.7,
            component_scores={"liquidity_depth": 0.9, "spread_slippage": 0.7},
        )
        assert rec.predicted_score == 0.8
        assert rec.actual_outcome == 0.7
        assert rec.recorded_at  # auto-populated


class TestCalibrationTracker:
    def _make_tracker_with_records(self, n: int = 10) -> CalibrationTracker:
        tracker = CalibrationTracker(window_size=100)
        for i in range(n):
            # Simulate slightly overconfident predictions
            predicted = 0.5 + (i / n) * 0.4
            actual = predicted - 0.05  # consistently 5% optimistic
            tracker.record(
                agent_name="test_agent",
                domain="test",
                predicted_score=predicted,
                actual_outcome=actual,
                component_scores={"comp_a": 0.7, "comp_b": 0.3 + i * 0.05},
            )
        return tracker

    def test_record_and_stats(self):
        tracker = self._make_tracker_with_records(10)
        stats = tracker.get_calibration_stats("test_agent")
        assert stats.record_count == 10
        assert stats.mean_bias > 0  # overconfident → positive bias
        assert stats.ece >= 0.0
        assert "comp_a" in stats.adjustment_factors
        assert "comp_b" in stats.adjustment_factors

    def test_few_records_empty_adjustments(self):
        tracker = CalibrationTracker()
        tracker.record("a", "d", 0.5, 0.5, {"x": 0.5})
        tracker.record("a", "d", 0.6, 0.6, {"x": 0.6})
        assert tracker.get_weight_adjustments("a") == {}

    def test_window_size_eviction(self):
        tracker = CalibrationTracker(window_size=5)
        for i in range(10):
            tracker.record("a", "d", 0.5, 0.5, {"x": float(i)})
        stats = tracker.get_calibration_stats("a")
        assert stats.record_count == 5

    def test_build_result_with_calibration(self):
        tracker = CalibrationTracker(window_size=100)
        # Create records that make comp_a correlate with under-estimation
        for i in range(10):
            tracker.record(
                agent_name="test",
                domain="test",
                predicted_score=0.5,
                actual_outcome=0.3 + (i / 10) * 0.5,  # varies with comp_a
                component_scores={"comp_a": 0.3 + (i / 10) * 0.5, "comp_b": 0.5},
            )

        components = {
            "comp_a": TrustComponent(score=0.8, weight=0.5),
            "comp_b": TrustComponent(score=0.6, weight=0.5),
        }

        # Without calibration
        result_no_cal = _build_result(
            agent_name="test",
            domain="test",
            component_breakdown=dict(components),
            violations=[],
            why_trusted="test",
            source_type="test",
            source_id="test-1",
        )

        # With calibration
        result_with_cal = _build_result(
            agent_name="test",
            domain="test",
            component_breakdown={
                "comp_a": TrustComponent(score=0.8, weight=0.5),
                "comp_b": TrustComponent(score=0.6, weight=0.5),
            },
            violations=[],
            why_trusted="test",
            source_type="test",
            source_id="test-1",
            calibration_tracker=tracker,
        )

        # Scores should differ because calibration adjusts weights
        assert result_with_cal.trust_score != result_no_cal.trust_score


# ── End-to-end ────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_assembled_geopolitical_through_router(self):
        from tollama.xai.connectors.helpers import build_default_assembler

        assembler = build_default_assembler()
        assembly = assembler.assemble("geopolitical", "US")

        router = build_default_trust_router()
        result = router.analyze(
            context=assembly.trust_context,
            payload=assembly.payload,
        )
        assert result is not None
        assert 0.0 <= result.trust_score <= 1.0
        assert result.agent_name == "geopolitical"
