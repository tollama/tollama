"""Tests for OpenClaw tollama-forecast helper scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "tollama-forecast"
HEALTH_SCRIPT = SKILL_DIR / "bin" / "tollama-health.sh"
MODELS_SCRIPT = SKILL_DIR / "bin" / "tollama-models.sh"
FORECAST_SCRIPT = SKILL_DIR / "bin" / "tollama-forecast.sh"

BASH_BIN = shutil.which("bash")
REQUIRED_RUNTIME_BINS = ("awk", "cat", "mktemp", "rm", "python3")
MISSING_RUNTIME_BINS = [name for name in REQUIRED_RUNTIME_BINS if shutil.which(name) is None]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run_script(
    script: Path,
    args: list[str],
    *,
    env: dict[str, str],
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    assert BASH_BIN is not None
    return subprocess.run(
        [BASH_BIN, str(script), *args],
        input=stdin_text,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def _base_env(path_prefix: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = path_prefix
    return env


def _make_runtime_bin(
    tmp_path: Path,
    *,
    tollama_script: str | None = None,
    curl_script: str | None = None,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    assert BASH_BIN is not None
    (bin_dir / "bash").symlink_to(BASH_BIN)

    for name in REQUIRED_RUNTIME_BINS:
        source = shutil.which(name)
        assert source is not None
        (bin_dir / name).symlink_to(source)

    if curl_script is None:
        curl_bin = shutil.which("curl")
        assert curl_bin is not None
        (bin_dir / "curl").symlink_to(curl_bin)
    else:
        _write_executable(bin_dir / "curl", curl_script)

    if tollama_script is not None:
        _write_executable(bin_dir / "tollama", tollama_script)

    return bin_dir


def _minimal_request_payload() -> dict[str, Any]:
    return {
        "horizon": 2,
        "series": [
            {
                "id": "s1",
                "freq": "D",
                "timestamps": ["2025-01-01"],
                "target": [1.0],
            }
        ],
        "options": {},
    }


def _write_minimal_request(path: Path) -> None:
    path.write_text(json.dumps(_minimal_request_payload()), encoding="utf-8")


def _build_fake_curl_script(route_logic: str, *, log_path: Path | None = None) -> str:
    log_stmt = ""
    if log_path is not None:
        log_stmt = f'printf "%s\\t%s\\t%s\\n" "$method" "$url" "$data" >> "{log_path}"\n'

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        method="GET"
        out_file=""
        data=""
        url=""

        while (($# > 0)); do
          case "$1" in
            -X)
              shift
              method="$1"
              ;;
            -o)
              shift
              out_file="$1"
              ;;
            --data)
              shift
              data="$1"
              ;;
            -H|--connect-timeout|--max-time|-w)
              shift
              ;;
            -s|-S|-sS)
              ;;
            *)
              url="$1"
              ;;
          esac
          shift
        done

        status="500"
        body='{{"detail":"unhandled"}}'

        {log_stmt}
        {route_logic}

        printf '%s' "$body" > "$out_file"
        printf '%s' "$status"
        """,
    )


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_health_success_returns_zero_and_json(tmp_path: Path) -> None:
    curl_script = _build_fake_curl_script(
        """
        if [[ "$url" == *"/v1/health" ]]; then
          status="200"
          body='{"status":"ok"}'
        elif [[ "$url" == *"/api/version" ]]; then
          status="200"
          body='{"version":"0.1.0"}'
        else
          status="404"
          body='{"detail":"not found"}'
        fi
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, curl_script=curl_script)

    result = _run_script(
        HEALTH_SCRIPT,
        ["--base-url", "http://daemon.test:11435", "--timeout", "2"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["base_url"] == "http://daemon.test:11435"
    assert payload["health"]["status"] == 200
    assert payload["version"]["status"] == 200


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_health_failure_returns_exit_five_and_hint(tmp_path: Path) -> None:
    curl_script = _build_fake_curl_script(
        """
        if [[ "$url" == *"/v1/health" ]]; then
          status="500"
          body='{"detail":"boom"}'
        elif [[ "$url" == *"/api/version" ]]; then
          status="200"
          body='{"version":"0.1.0"}'
        fi
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, curl_script=curl_script)

    result = _run_script(
        HEALTH_SCRIPT,
        ["--base-url", "http://daemon.test:11435", "--timeout", "2"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 5
    assert "exec host and --base-url may be mismatched" in result.stderr


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_models_available_remote_mode_does_not_local_fallback_when_cli_fails(
    tmp_path: Path,
) -> None:
    curl_log = tmp_path / "curl.log"

    tollama_script = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$1" == "info" ]]; then
          echo "daemon unreachable" >&2
          exit 1
        fi
        exit 1
        """,
    )
    curl_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        echo called >> "{curl_log}"
        exit 0
        """,
    )
    runtime_bin = _make_runtime_bin(
        tmp_path,
        tollama_script=tollama_script,
        curl_script=curl_script,
    )

    result = _run_script(
        MODELS_SCRIPT,
        ["available", "--base-url", "http://daemon.test:11435", "--timeout", "2"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 5
    assert "tollama info --json --remote failed" in result.stderr
    assert not curl_log.exists()


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_model_missing_without_pull_returns_exit_four(tmp_path: Path) -> None:
    tollama_log = tmp_path / "tollama.log"
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    tollama_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        echo "$*" >> "{tollama_log}"
        if [[ "$1" == "show" ]]; then
          echo "Error: show model failed with HTTP 404: missing" >&2
          exit 1
        fi
        if [[ "$1" == "pull" ]]; then
          echo "unexpected pull" >&2
          exit 1
        fi
        if [[ "$1" == "run" ]]; then
          echo "unexpected run" >&2
          exit 1
        fi
        exit 1
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, tollama_script=tollama_script)

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path)],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 4
    assert "Re-run with --pull" in result.stderr
    log_lines = tollama_log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("show ") for line in log_lines)
    assert not any(line.startswith("pull ") for line in log_lines)


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_missing_model_with_pull_and_accept_license(tmp_path: Path) -> None:
    tollama_log = tmp_path / "tollama.log"
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    tollama_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        echo "$*" >> "{tollama_log}"
        if [[ "$1" == "show" ]]; then
          echo "Error: show model failed with HTTP 404: missing" >&2
          exit 1
        fi
        if [[ "$1" == "pull" ]]; then
          echo '{{"status":"success","model":"mock"}}'
          exit 0
        fi
        if [[ "$1" == "run" ]]; then
          echo '{{"model":"mock","forecasts":[]}}'
          exit 0
        fi
        exit 1
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, tollama_script=tollama_script)

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path), "--pull", "--accept-license"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["model"] == "mock"
    log_lines = tollama_log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("pull ") for line in log_lines)
    assert any("--accept-license" in line for line in log_lines if line.startswith("pull "))
    assert any(line.startswith("run ") for line in log_lines)


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_cli_success_does_not_call_curl(tmp_path: Path) -> None:
    curl_log = tmp_path / "curl.log"
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    tollama_script = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$1" == "show" ]]; then
          echo '{"model":"mock"}'
          exit 0
        fi
        if [[ "$1" == "run" ]]; then
          echo '{"model":"mock","forecasts":[]}'
          exit 0
        fi
        exit 1
        """,
    )
    curl_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        echo called >> "{curl_log}"
        exit 1
        """,
    )
    runtime_bin = _make_runtime_bin(
        tmp_path,
        tollama_script=tollama_script,
        curl_script=curl_script,
    )

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path)],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["model"] == "mock"
    assert not curl_log.exists()


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_cli_non_zero_does_not_fallback_to_http(tmp_path: Path) -> None:
    curl_log = tmp_path / "curl.log"
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    tollama_script = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "$1" == "show" ]]; then
          echo '{"model":"mock"}'
          exit 0
        fi
        if [[ "$1" == "run" ]]; then
          echo 'runner failed' >&2
          exit 42
        fi
        exit 1
        """,
    )
    curl_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        echo called >> "{curl_log}"
        exit 0
        """,
    )
    runtime_bin = _make_runtime_bin(
        tmp_path,
        tollama_script=tollama_script,
        curl_script=curl_script,
    )

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path)],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 6
    assert "tollama run failed" in result.stderr
    assert not curl_log.exists()


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_cli_missing_fallbacks_api_then_v1_on_404(tmp_path: Path) -> None:
    curl_log = tmp_path / "curl.log"
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    curl_script = _build_fake_curl_script(
        """
        if [[ "$url" == *"/api/show" ]]; then
          status="200"
          body='{"name":"mock"}'
        elif [[ "$url" == *"/api/forecast" ]]; then
          status="404"
          body='{"detail":"not found"}'
        elif [[ "$url" == *"/v1/forecast" ]]; then
          status="200"
          body='{"model":"mock","forecasts":[]}'
        fi
        """,
        log_path=curl_log,
    )
    runtime_bin = _make_runtime_bin(tmp_path, curl_script=curl_script)

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path), "--base-url", "http://daemon.test"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["model"] == "mock"

    lines = curl_log.read_text(encoding="utf-8").splitlines()
    requests = [line.split("\t", 2) for line in lines]
    urls = [url for _method, url, _data in requests]
    assert urls == [
        "http://daemon.test/api/show",
        "http://daemon.test/api/forecast",
        "http://daemon.test/v1/forecast",
    ]

    _method, _url, data = requests[1]
    sent = json.loads(data)
    assert sent["model"] == "mock"
    assert sent["stream"] is False


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
def test_forecast_http_400_preserves_detail_and_returns_exit_six(tmp_path: Path) -> None:
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    curl_script = _build_fake_curl_script(
        """
        if [[ "$url" == *"/api/show" ]]; then
          status="200"
          body='{"name":"mock"}'
        elif [[ "$url" == *"/api/forecast" ]]; then
          status="400"
          body='{"detail":"future_covariates mismatch"}'
        fi
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, curl_script=curl_script)

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path), "--base-url", "http://daemon.test"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 6
    assert "future_covariates mismatch" in result.stderr


@pytest.mark.skipif(
    BASH_BIN is None or MISSING_RUNTIME_BINS,
    reason="bash and runtime binaries required",
)
@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (503, "runner unavailable"),
        (502, "runner protocol error"),
    ],
)
def test_forecast_http_5xx_preserves_status_and_detail(
    tmp_path: Path,
    status_code: int,
    detail: str,
) -> None:
    input_path = tmp_path / "req.json"
    _write_minimal_request(input_path)

    curl_script = _build_fake_curl_script(
        f"""
        if [[ "$url" == *"/api/show" ]]; then
          status="200"
          body='{{"name":"mock"}}'
        elif [[ "$url" == *"/api/forecast" ]]; then
          status="{status_code}"
          body='{{"detail":"{detail}"}}'
        fi
        """,
    )
    runtime_bin = _make_runtime_bin(tmp_path, curl_script=curl_script)

    result = _run_script(
        FORECAST_SCRIPT,
        ["--model", "mock", "--input", str(input_path), "--base-url", "http://daemon.test"],
        env=_base_env(str(runtime_bin)),
    )

    assert result.returncode == 6
    assert f"HTTP {status_code}" in result.stderr
    assert detail in result.stderr
