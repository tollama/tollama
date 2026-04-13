"""Tests for hierarchical forecast reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from tollama.core.reconciliation import (
    HierarchySpec,
    build_summing_matrix,
    check_coherency,
    reconcile,
)
from tollama.core.schemas import ForecastResponse, SeriesForecast


@pytest.fixture()
def simple_hierarchy() -> HierarchySpec:
    return HierarchySpec(tree={"total": ["region_a", "region_b"]})


@pytest.fixture()
def simple_response() -> ForecastResponse:
    return ForecastResponse(
        model="test",
        forecasts=[
            SeriesForecast(
                id="total",
                freq="D",
                start_timestamp="2025-01-01",
                mean=[100.0, 200.0, 300.0],
            ),
            SeriesForecast(
                id="region_a",
                freq="D",
                start_timestamp="2025-01-01",
                mean=[60.0, 120.0, 180.0],
            ),
            SeriesForecast(
                id="region_b",
                freq="D",
                start_timestamp="2025-01-01",
                mean=[50.0, 90.0, 130.0],
            ),
        ],
    )


def test_hierarchy_leaves(simple_hierarchy: HierarchySpec) -> None:
    assert simple_hierarchy.leaves() == ["region_a", "region_b"]


def test_hierarchy_all_series_ids(simple_hierarchy: HierarchySpec) -> None:
    ids = simple_hierarchy.all_series_ids()
    assert ids == ["total", "region_a", "region_b"]


def test_build_summing_matrix(simple_hierarchy: HierarchySpec) -> None:
    s = build_summing_matrix(simple_hierarchy)
    assert s.shape == (3, 2)  # 3 total series, 2 bottom
    # total = region_a + region_b
    np.testing.assert_array_equal(s[0], [1.0, 1.0])
    # region_a and region_b are identity
    np.testing.assert_array_equal(s[1], [1.0, 0.0])
    np.testing.assert_array_equal(s[2], [0.0, 1.0])


def test_reconcile_bottom_up(
    simple_hierarchy: HierarchySpec,
    simple_response: ForecastResponse,
) -> None:
    result = reconcile(simple_response, simple_hierarchy, method="bottom_up")
    fc_map = {fc.id: fc for fc in result.forecasts}
    # After bottom-up, total = region_a + region_b
    for i in range(3):
        total = fc_map["total"].mean[i]
        child_sum = fc_map["region_a"].mean[i] + fc_map["region_b"].mean[i]
        assert abs(total - child_sum) < 1e-10


def test_reconcile_ols(
    simple_hierarchy: HierarchySpec,
    simple_response: ForecastResponse,
) -> None:
    result = reconcile(simple_response, simple_hierarchy, method="ols")
    fc_map = {fc.id: fc for fc in result.forecasts}
    for i in range(3):
        total = fc_map["total"].mean[i]
        child_sum = fc_map["region_a"].mean[i] + fc_map["region_b"].mean[i]
        assert abs(total - child_sum) < 1e-6


def test_check_coherency_incoherent(
    simple_hierarchy: HierarchySpec,
    simple_response: ForecastResponse,
) -> None:
    # Original response is incoherent (100 != 60 + 50)
    violations = check_coherency(simple_response, simple_hierarchy)
    assert len(violations) > 0


def test_check_coherency_after_reconcile(
    simple_hierarchy: HierarchySpec,
    simple_response: ForecastResponse,
) -> None:
    reconciled = reconcile(simple_response, simple_hierarchy, method="bottom_up")
    violations = check_coherency(reconciled, simple_hierarchy)
    assert len(violations) == 0


def test_reconcile_passes_through_outside_series(
    simple_hierarchy: HierarchySpec,
) -> None:
    response = ForecastResponse(
        model="test",
        forecasts=[
            SeriesForecast(id="total", freq="D", start_timestamp="2025-01-01", mean=[100.0]),
            SeriesForecast(id="region_a", freq="D", start_timestamp="2025-01-01", mean=[60.0]),
            SeriesForecast(id="region_b", freq="D", start_timestamp="2025-01-01", mean=[40.0]),
            SeriesForecast(id="unrelated", freq="D", start_timestamp="2025-01-01", mean=[999.0]),
        ],
    )
    result = reconcile(response, simple_hierarchy, method="bottom_up")
    fc_ids = {fc.id for fc in result.forecasts}
    assert "unrelated" in fc_ids
    unrelated = next(fc for fc in result.forecasts if fc.id == "unrelated")
    assert unrelated.mean == [999.0]


def test_validate_series_ids(simple_hierarchy: HierarchySpec) -> None:
    missing = simple_hierarchy.validate_series_ids({"total", "region_a"})
    assert len(missing) > 0  # region_b missing

    ok = simple_hierarchy.validate_series_ids({"total", "region_a", "region_b"})
    assert len(ok) == 0
