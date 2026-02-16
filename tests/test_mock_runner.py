"""Subprocess tests for the mock runner stdio protocol."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from tollama.core.protocol import decode_response_line


@contextmanager
def _runner_process() -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, "-m", "tollama.runners.mock.main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        yield process
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _send(process: subprocess.Popen[str], payload: dict[str, object]) -> dict[str, object]:
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(json.dumps(payload) + "\n")
    process.stdin.flush()

    line = process.stdout.readline()
    assert line, "runner exited before responding"
    response = decode_response_line(line)
    return response.model_dump(mode="json", exclude_none=True)


def test_mock_runner_hello_and_forecast_for_multiple_series() -> None:
    with _runner_process() as process:
        hello_response = _send(
            process,
            {"id": "req-1", "method": "hello", "params": {}},
        )
        assert hello_response["id"] == "req-1"
        assert hello_response["result"] == {
            "name": "tollama-mock",
            "version": "0.1.0",
            "capabilities": ["hello", "forecast"],
        }

        forecast_response = _send(
            process,
            {
                "id": "req-2",
                "method": "forecast",
                "params": {
                    "model": "mock-naive",
                    "horizon": 3,
                    "quantiles": [0.1, 0.9],
                    "series": [
                        {
                            "id": "series-a",
                            "freq": "D",
                            "timestamps": ["2025-01-01", "2025-01-02"],
                            "target": [10.0, 20.0],
                        },
                        {
                            "id": "series-b",
                            "freq": "D",
                            "timestamps": ["2025-01-01", "2025-01-02"],
                            "target": [5.0, 7.0],
                        },
                    ],
                    "options": {},
                },
            },
        )

    assert forecast_response["id"] == "req-2"
    result = forecast_response["result"]
    assert result["model"] == "mock-naive"
    assert len(result["forecasts"]) == 2

    first = result["forecasts"][0]
    assert first["id"] == "series-a"
    assert first["mean"] == [20.0, 20.0, 20.0]
    assert first["quantiles"] == {"0.1": [20.0, 20.0, 20.0], "0.9": [20.0, 20.0, 20.0]}

    second = result["forecasts"][1]
    assert second["id"] == "series-b"
    assert second["mean"] == [7.0, 7.0, 7.0]
    assert second["quantiles"] == {"0.1": [7.0, 7.0, 7.0], "0.9": [7.0, 7.0, 7.0]}


def test_mock_runner_returns_structured_error_for_invalid_requests() -> None:
    with _runner_process() as process:
        unknown_method = _send(
            process,
            {"id": "req-x", "method": "not-supported", "params": {}},
        )
        assert unknown_method["id"] == "req-x"
        assert unknown_method["error"]["code"] == -32601
        assert unknown_method["error"]["message"] == "method not found"

        invalid_params = _send(
            process,
            {
                "id": "req-y",
                "method": "forecast",
                "params": {"model": "bad", "horizon": "3", "series": []},
            },
        )
        assert invalid_params["id"] == "req-y"
        assert invalid_params["error"]["code"] == -32602
        assert invalid_params["error"]["message"] == "invalid params"

        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write("{bad-json}\n")
        process.stdin.flush()
        line = process.stdout.readline()
        decoded = decode_response_line(line).model_dump(mode="json", exclude_none=True)
        assert decoded["id"] == "unknown"
        assert decoded["error"]["code"] == -32600
        assert decoded["error"]["message"] == "invalid request"


def test_mock_runner_console_script_is_registered() -> None:
    import importlib.metadata

    entry_points = importlib.metadata.entry_points(group="console_scripts")
    assert any(ep.name == "tollama-runner-mock" for ep in entry_points)
