"""Smoke checks for the concrete-solution demo script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_core_concrete_solution_demo_script_is_wired_to_realdata_flow() -> None:
    script_path = (
        Path(__file__).resolve().parents[1] / "examples" / "core_concrete_solution_demo.sh"
    )

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr

    content = script_path.read_text(encoding="utf-8")
    assert "export_core_solution_input.py" in content
    assert "USE_CHECKED_IN_INPUT" in content
    assert "core_solution_hourly_input.json" in content
    assert "tollama.cli.main benchmark" in content
    assert "routing apply" in content
    assert "chronos2,granite-ttm-r2,timesfm-2.5-200m,moirai-2.0-R-small" in content
