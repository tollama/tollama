"""Basic smoke tests for package wiring."""

import tollama


def test_version_exposed() -> None:
    assert isinstance(tollama.__version__, str)
    assert tollama.__version__
