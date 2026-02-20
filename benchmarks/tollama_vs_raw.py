"""Compare SDK ergonomics and time-to-first-forecast vs raw HTTP client calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tollama import Tollama
from tollama.client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS, TollamaClient

SDK_EXAMPLE = """
from tollama import Tollama

t = Tollama(base_url=base_url, timeout=timeout)
result = t.forecast(
    model=model,
    series={"target": values, "freq": "D"},
    horizon=horizon,
)
"""

RAW_EXAMPLE = """
from tollama.client import TollamaClient

client = TollamaClient(base_url=base_url, timeout=timeout)
payload = {
    "model": model,
    "horizon": horizon,
    "series": [
        {
            "id": "s1",
            "freq": "D",
            "timestamps": timestamps,
            "target": values,
        }
    ],
    "options": {},
}
response = client.forecast(payload, stream=False)
"""


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    method: str
    loc: int
    first_ms: float
    mean_ms: float


def count_effective_loc(snippet: str) -> int:
    """Count non-empty, non-comment code lines."""
    return sum(
        1
        for raw_line in snippet.splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    )


def sdk_example_loc() -> int:
    """Return effective LOC for the SDK example."""
    return count_effective_loc(SDK_EXAMPLE)


def raw_example_loc() -> int:
    """Return effective LOC for the raw-client example."""
    return count_effective_loc(RAW_EXAMPLE)


def build_values(*, context_length: int) -> list[float]:
    """Build a deterministic toy series for benchmarking."""
    if context_length < 2:
        raise ValueError("context_length must be >= 2")
    return [float(10 + index) for index in range(context_length)]


def build_raw_request(*, model: str, horizon: int, values: list[float]) -> dict[str, Any]:
    """Build a canonical non-streaming forecast request payload."""
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    timestamps = [str(index + 1) for index in range(len(values))]
    return {
        "model": model,
        "horizon": horizon,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": timestamps,
                "target": values,
            }
        ],
        "options": {},
    }


def run_sdk_call(
    *,
    sdk: Tollama,
    model: str,
    horizon: int,
    values: list[float],
) -> dict[str, Any]:
    """Run one SDK forecast and return canonical JSON response."""
    result = sdk.forecast(
        model=model,
        series={"target": values, "freq": "D"},
        horizon=horizon,
    )
    return result.response.model_dump(mode="json", exclude_none=True)


def run_raw_call(
    *,
    client: TollamaClient,
    model: str,
    horizon: int,
    values: list[float],
) -> dict[str, Any]:
    """Run one raw non-streaming forecast call and return response payload."""
    payload = build_raw_request(model=model, horizon=horizon, values=values)
    response = client.forecast(payload, stream=False)
    if not isinstance(response, dict):
        raise RuntimeError("raw client returned non-object response")
    return response


def benchmark_method(
    *,
    call: Callable[[], dict[str, Any]],
    iterations: int,
    warmup: int,
) -> tuple[float, float]:
    """Return (first_call_ms, mean_call_ms)."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        call()

    durations_ms: list[float] = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        call()
        durations_ms.append((time.perf_counter() - started_at) * 1000.0)

    return durations_ms[0], statistics.fmean(durations_ms)


def build_benchmark_rows(
    *,
    sdk_call: Callable[[], dict[str, Any]],
    raw_call: Callable[[], dict[str, Any]],
    iterations: int,
    warmup: int,
) -> list[BenchmarkRow]:
    """Benchmark both SDK and raw-client paths."""
    sdk_first, sdk_mean = benchmark_method(call=sdk_call, iterations=iterations, warmup=warmup)
    raw_first, raw_mean = benchmark_method(call=raw_call, iterations=iterations, warmup=warmup)
    return [
        BenchmarkRow(
            method="SDK (Tollama)",
            loc=sdk_example_loc(),
            first_ms=sdk_first,
            mean_ms=sdk_mean,
        ),
        BenchmarkRow(
            method="Raw Client",
            loc=raw_example_loc(),
            first_ms=raw_first,
            mean_ms=raw_mean,
        ),
    ]


def render_results_table(rows: list[BenchmarkRow]) -> str:
    """Render a simple fixed-width benchmark table."""
    headers = ["Method", "LOC", "First forecast (ms)", "Mean forecast (ms)"]
    method_width = max(len(headers[0]), *(len(row.method) for row in rows))
    loc_width = max(len(headers[1]), *(len(str(row.loc)) for row in rows))
    first_width = max(len(headers[2]), *(len(f"{row.first_ms:.2f}") for row in rows))
    mean_width = max(len(headers[3]), *(len(f"{row.mean_ms:.2f}") for row in rows))

    lines = [
        f"{headers[0]:<{method_width}}  {headers[1]:>{loc_width}}  "
        f"{headers[2]:>{first_width}}  {headers[3]:>{mean_width}}",
        f"{'-' * method_width}  {'-' * loc_width}  {'-' * first_width}  {'-' * mean_width}",
    ]
    lines.extend(
        f"{row.method:<{method_width}}  {row.loc:>{loc_width}}  "
        f"{row.first_ms:>{first_width}.2f}  {row.mean_ms:>{mean_width}.2f}"
        for row in rows
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Tollama SDK ergonomics and forecast latency vs raw client calls.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Daemon base URL")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout",
    )
    parser.add_argument("--model", default="mock", help="Model name to call")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon")
    parser.add_argument("--context-length", type=int, default=8, help="Input series length")
    parser.add_argument("--iterations", type=int, default=3, help="Timed iterations per method")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup calls per method")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    values = build_values(context_length=args.context_length)

    sdk = Tollama(base_url=args.base_url, timeout=float(args.timeout))
    client = TollamaClient(base_url=args.base_url, timeout=float(args.timeout))

    def sdk_call() -> dict[str, Any]:
        return run_sdk_call(
            sdk=sdk,
            model=args.model,
            horizon=int(args.horizon),
            values=values,
        )

    def raw_call() -> dict[str, Any]:
        return run_raw_call(
            client=client,
            model=args.model,
            horizon=int(args.horizon),
            values=values,
        )

    try:
        rows = build_benchmark_rows(
            sdk_call=sdk_call,
            raw_call=raw_call,
            iterations=int(args.iterations),
            warmup=int(args.warmup),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"benchmark failed: {exc}")
        return 1

    sdk_row = next(row for row in rows if row.method == "SDK (Tollama)")
    raw_row = next(row for row in rows if row.method == "Raw Client")
    payload = {
        "config": {
            "base_url": args.base_url,
            "timeout": float(args.timeout),
            "model": args.model,
            "horizon": int(args.horizon),
            "context_length": int(args.context_length),
            "iterations": int(args.iterations),
            "warmup": int(args.warmup),
        },
        "results": [
            {
                "method": row.method,
                "loc": row.loc,
                "first_forecast_ms": round(row.first_ms, 3),
                "mean_forecast_ms": round(row.mean_ms, 3),
            }
            for row in rows
        ],
        "ratios": {
            "loc_raw_over_sdk": round(raw_row.loc / max(sdk_row.loc, 1), 3),
            "first_forecast_raw_over_sdk": round(raw_row.first_ms / max(sdk_row.first_ms, 1e-9), 3),
            "mean_forecast_raw_over_sdk": round(raw_row.mean_ms / max(sdk_row.mean_ms, 1e-9), 3),
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return 0

    print(render_results_table(rows))
    print("")
    print(
        "LOC ratio (raw/sdk): "
        f"{payload['ratios']['loc_raw_over_sdk']:.2f}x",
    )
    print(
        "First forecast latency ratio (raw/sdk): "
        f"{payload['ratios']['first_forecast_raw_over_sdk']:.2f}x",
    )
    print(
        "Mean forecast latency ratio (raw/sdk): "
        f"{payload['ratios']['mean_forecast_raw_over_sdk']:.2f}x",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
