"""Tests for tollama.preprocess.transforms."""

from __future__ import annotations

import numpy as np
import pytest

from tollama.preprocess.transforms import (
    DifferencingTransform,
    LogTransform,
    MinMaxScaler1D,
    StandardScaler1D,
    build_scaler,
    chronological_split,
)

# ---------------------------------------------------------------------------
# StandardScaler1D
# ---------------------------------------------------------------------------


def test_standard_scaler_round_trip() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sc = StandardScaler1D().fit(y)
    transformed = sc.transform(y)
    recovered = sc.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, y, atol=1e-12)


def test_standard_scaler_zero_variance() -> None:
    y = np.array([5.0, 5.0, 5.0])
    sc = StandardScaler1D().fit(y)
    assert sc.std_ == 1.0
    transformed = sc.transform(y)
    assert np.all(transformed == 0.0)


def test_standard_scaler_unfitted_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        StandardScaler1D().transform(np.array([1.0]))


def test_standard_scaler_to_dict() -> None:
    sc = StandardScaler1D().fit(np.array([0.0, 10.0]))
    d = sc.to_dict()
    assert d["method"] == "standard"
    assert "mean" in d and "std" in d


# ---------------------------------------------------------------------------
# MinMaxScaler1D
# ---------------------------------------------------------------------------


def test_minmax_scaler_round_trip() -> None:
    y = np.array([2.0, 4.0, 6.0, 8.0])
    sc = MinMaxScaler1D().fit(y)
    transformed = sc.transform(y)
    assert pytest.approx(transformed[0]) == 0.0
    assert pytest.approx(transformed[-1]) == 1.0
    recovered = sc.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, y, atol=1e-12)


def test_minmax_scaler_constant_input() -> None:
    y = np.array([3.0, 3.0, 3.0])
    sc = MinMaxScaler1D().fit(y)
    assert sc.max_ == sc.min_ + 1.0


def test_minmax_scaler_to_dict() -> None:
    sc = MinMaxScaler1D().fit(np.array([1.0, 5.0]))
    d = sc.to_dict()
    assert d["method"] == "minmax"
    assert d["min"] == 1.0 and d["max"] == 5.0


# ---------------------------------------------------------------------------
# DifferencingTransform
# ---------------------------------------------------------------------------


def test_differencing_round_trip() -> None:
    y = np.array([10.0, 12.0, 15.0, 13.0])
    dt = DifferencingTransform().fit(y)
    diffed = dt.transform(y)
    assert len(diffed) == len(y) - 1
    recovered = dt.inverse_transform(diffed)
    np.testing.assert_allclose(recovered, y, atol=1e-12)


def test_differencing_too_short() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        DifferencingTransform().fit(np.array([1.0]))


# ---------------------------------------------------------------------------
# LogTransform
# ---------------------------------------------------------------------------


def test_log_round_trip() -> None:
    y = np.array([0.0, 1.0, 2.0, 10.0])
    lt = LogTransform().fit(y)
    transformed = lt.transform(y)
    recovered = lt.inverse_transform(transformed)
    np.testing.assert_allclose(recovered, y, atol=1e-12)


def test_log_negative_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LogTransform().fit(np.array([-1.0, 2.0]))


# ---------------------------------------------------------------------------
# build_scaler factory
# ---------------------------------------------------------------------------


def test_build_scaler_standard() -> None:
    sc = build_scaler("standard")
    assert isinstance(sc, StandardScaler1D)


def test_build_scaler_minmax() -> None:
    sc = build_scaler("minmax")
    assert isinstance(sc, MinMaxScaler1D)


def test_build_scaler_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_scaler("robust")


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_shapes() -> None:
    y = np.arange(100, dtype=float)
    train, val, test, (train_end, val_end) = chronological_split(y)
    assert len(train) == train_end
    assert len(val) == val_end - train_end
    assert len(test) == len(y) - val_end
    assert len(train) + len(val) + len(test) == len(y)


def test_chronological_split_bad_ratios() -> None:
    with pytest.raises(ValueError, match="< 1.0"):
        chronological_split(np.arange(10, dtype=float), train_ratio=0.8, val_ratio=0.3)


def test_chronological_split_too_short() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        chronological_split(np.array([1.0, 2.0]))
