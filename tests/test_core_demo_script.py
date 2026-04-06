"""Smoke checks for the Core workflow demo script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_core_workflow_demo_script_is_wired_to_core_flow() -> None:
    script_path = Path(__file__).resolve().parents[1] / "examples" / "core_workflow_demo.sh"

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr

    content = script_path.read_text(encoding="utf-8")
    assert "run_pipeline" in content
    assert "quickstart" in content
    assert "benchmark" in content
    assert "routing apply" in content
