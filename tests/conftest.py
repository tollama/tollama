"""Shared pytest fixtures and marker registration for tollama tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tollama.core.protocol import ProtocolRequest
from tollama.core.storage import TollamaPaths


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests that take noticeably longer to run")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU-backed runtimes")
    config.addinivalue_line("markers", "network: marks tests that require live network access")


@pytest.fixture
def tmp_tollama_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TollamaPaths:
    """Create one isolated Tollama state directory for a test."""
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    monkeypatch.setenv("TOLLAMA_HOME", str(paths.base_dir))
    return paths


@pytest.fixture
def sample_series_input() -> dict[str, Any]:
    """Return one minimal valid series payload."""
    return {
        "id": "series-1",
        "freq": "D",
        "timestamps": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "target": [1.0, 2.0, 3.0],
    }


@pytest.fixture
def protocol_request_factory() -> Callable[..., ProtocolRequest]:
    """Build validated protocol request objects with sensible defaults."""

    def _factory(
        *,
        request_id: str = "req-1",
        method: str = "forecast",
        params: dict[str, Any] | None = None,
    ) -> ProtocolRequest:
        return ProtocolRequest(
            id=request_id,
            method=method,
            params=params or {},
        )

    return _factory
