"""Property-based schema serialization checks."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - optional dev dependency
    pytest.skip("hypothesis is not installed", allow_module_level=True)

from tollama.core.schemas import ForecastRequest


@given(
    horizon=st.integers(min_value=1, max_value=8),
    values=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=3,
        max_size=6,
    ),
)
def test_forecast_request_json_round_trip(horizon: int, values: list[float]) -> None:
    request = ForecastRequest.model_validate(
        {
            "model": "mock",
            "horizon": horizon,
            "series": [
                {
                    "id": "series-1",
                    "freq": "D",
                    "timestamps": [f"2025-01-0{index + 1}" for index in range(len(values))],
                    "target": values,
                }
            ],
            "options": {},
        }
    )

    dumped = request.model_dump(mode="json")
    assert ForecastRequest.model_validate(dumped) == request
