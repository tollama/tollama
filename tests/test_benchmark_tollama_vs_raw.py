"""Tests for benchmark helper script."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "tollama_vs_raw.py"
_MODULE_SPEC = spec_from_file_location("benchmarks_tollama_vs_raw", _MODULE_PATH)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _MODULE
_MODULE_SPEC.loader.exec_module(_MODULE)

BenchmarkRow = _MODULE.BenchmarkRow
benchmark_method = _MODULE.benchmark_method
build_raw_request = _MODULE.build_raw_request
count_effective_loc = _MODULE.count_effective_loc
raw_example_loc = _MODULE.raw_example_loc
render_results_table = _MODULE.render_results_table
sdk_example_loc = _MODULE.sdk_example_loc


def test_count_effective_loc_ignores_blanks_and_comments() -> None:
    snippet = """
    # comment

    line1 = 1
      # comment2
    line2 = 2
    """
    assert count_effective_loc(snippet) == 2


def test_sdk_example_requires_fewer_loc_than_raw_example() -> None:
    assert sdk_example_loc() < raw_example_loc()


def test_benchmark_method_runs_warmup_and_iterations() -> None:
    calls = {"count": 0}

    def _call() -> dict[str, str]:
        calls["count"] += 1
        return {"ok": "true"}

    first_ms, mean_ms = benchmark_method(call=_call, iterations=3, warmup=2)

    assert calls["count"] == 5
    assert first_ms >= 0.0
    assert mean_ms >= 0.0


def test_build_raw_request_has_canonical_shape() -> None:
    payload = build_raw_request(model="mock", horizon=2, values=[1.0, 2.0, 3.0])

    assert payload["model"] == "mock"
    assert payload["horizon"] == 2
    assert payload["series"][0]["target"] == [1.0, 2.0, 3.0]
    assert payload["options"] == {}


def test_render_results_table_contains_headers() -> None:
    table = render_results_table(
        [
            BenchmarkRow(method="SDK (Tollama)", loc=4, first_ms=12.34, mean_ms=12.0),
            BenchmarkRow(method="Raw Client", loc=11, first_ms=13.2, mean_ms=13.0),
        ],
    )

    assert "Method" in table
    assert "First forecast (ms)" in table
    assert "SDK (Tollama)" in table
    assert "Raw Client" in table
