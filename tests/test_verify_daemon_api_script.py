"""Smoke checks for the daemon API verification script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_verify_daemon_api_script_supports_release_gate_artifacts() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "verify_daemon_api.sh"

    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert syntax.returncode == 0, syntax.stderr

    content = script_path.read_text(encoding="utf-8")
    assert "--output-dir" in content
    assert "result.json" in content
    assert "summary.json" in content
    assert "summary.md" in content
