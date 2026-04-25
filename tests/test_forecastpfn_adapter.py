from __future__ import annotations

from tollama.core.schemas import ForecastRequest, SeriesInput
from tollama.runners.forecastpfn_runner.adapter import ForecastPFNAdapter
from tollama.runners.forecastpfn_runner.errors import UnsupportedModelError


def _series_input() -> SeriesInput:
    return SeriesInput(
        id="series-1",
        freq="D",
        timestamps=["2025-01-01", "2025-01-02", "2025-01-03"],
        target=[1.0, 2.0, 3.0],
    )


def test_forecastpfn_manifest_only_source_raises_clear_error() -> None:
    adapter = ForecastPFNAdapter()
    request = ForecastRequest(
        model="forecastpfn",
        horizon=2,
        series=[_series_input()],
    )

    try:
        adapter.forecast(
            request,
            model_local_dir="/tmp/tollama-empty-forecastpfn",
            model_source={
                "type": "local",
                "repo_id": "tollama/forecastpfn-runner",
                "revision": "main",
            },
        )
    except UnsupportedModelError as exc:
        assert "manifest-only" in str(exc)
        assert "does not publish an installable Python package" in str(exc)
        assert "timer-base" in str(exc)
    else:
        raise AssertionError("expected ForecastPFN manifest-only source to be unsupported")
