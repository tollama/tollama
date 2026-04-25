from __future__ import annotations

import sys
import types

from tollama.core.schemas import ForecastRequest, SeriesInput
from tollama.runners.forecastpfn_runner.adapter import ForecastPFNAdapter
from tollama.runners.forecastpfn_runner.errors import DependencyMissingError, UnsupportedModelError


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


def test_forecastpfn_default_manifest_only_model_raises_clear_error() -> None:
    adapter = ForecastPFNAdapter()
    request = ForecastRequest(
        model="forecastpfn",
        horizon=2,
        series=[_series_input()],
    )

    try:
        adapter.forecast(request)
    except UnsupportedModelError as exc:
        assert "manifest-only" in str(exc)
        assert "does not publish an installable Python package" in str(exc)
        assert "timer-base" in str(exc)
    else:
        raise AssertionError("expected default ForecastPFN model to be unsupported")


def test_forecastpfn_missing_python_module_is_dependency_error(monkeypatch) -> None:
    def _array(values: object, dtype: object | None = None) -> object:
        del dtype
        return values

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.float32 = object()
    fake_numpy.array = _array

    fake_torch = types.ModuleType("torch")

    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ForecastPFN", None)

    adapter = ForecastPFNAdapter()
    request = ForecastRequest(
        model="custom-forecastpfn",
        horizon=2,
        series=[_series_input()],
    )

    try:
        adapter.forecast(
            request,
            model_source={
                "type": "huggingface",
                "repo_id": "example/forecastpfn",
                "revision": "main",
            },
        )
    except DependencyMissingError as exc:
        assert "ForecastPFN Python module is not installed or importable" in str(exc)
    else:
        raise AssertionError("expected missing ForecastPFN import to be a dependency error")
