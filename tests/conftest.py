"""Pytest configuration for tollama tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _disable_ansi_colors() -> None:
    """Disable ANSI colors in CLI output for consistent test assertions."""
    os.environ["TERM"] = "dumb"
