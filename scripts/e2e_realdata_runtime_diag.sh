#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT_DIR}/.venv/bin"

if [[ -d "${VENV_BIN}" ]]; then
  export PATH="${VENV_BIN}:$PATH"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="${PYTHON_BIN}"
elif [[ -x "${VENV_BIN}/python" ]]; then
  PYTHON_CMD="${VENV_BIN}/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="$(command -v python3)"
else
  PYTHON_CMD=""
fi

if [[ -z "${PYTHON_CMD}" ]]; then
  echo "python interpreter not found. Set PYTHON_BIN if needed."
  exit 1
fi

TIMEOUT_SECONDS="${IMPORT_TIMEOUT_SECONDS:-10}"
CAPTURE_STACK_SAMPLE_ON_TIMEOUT="${CAPTURE_STACK_SAMPLE_ON_TIMEOUT:-0}"
SAMPLE_DURATION_SECONDS="${SAMPLE_DURATION_SECONDS:-1}"
SAMPLE_DIR="${SAMPLE_DIR:-/tmp/tollama-e2e-runtime-samples}"
FAILURES=0

run_python_check() {
  local label="$1"
  shift
  local log_file
  local sample_file=""
  local status=0
  local timed_out=0

  log_file="$(mktemp -t tollama-e2e-runtime-check.XXXXXX)"
  "$PYTHON_CMD" "$@" >"${log_file}" 2>&1 &
  local worker_pid=$!
  if [[ "${CAPTURE_STACK_SAMPLE_ON_TIMEOUT}" == "1" ]] && command -v sample >/dev/null 2>&1; then
    mkdir -p "${SAMPLE_DIR}"
    sample_file="${SAMPLE_DIR}/${label}-${worker_pid}.sample.txt"
  fi
  (
    sleep "${TIMEOUT_SECONDS}"
    if kill -0 "${worker_pid}" 2>/dev/null; then
      if [[ -n "${sample_file}" ]]; then
        sample "${worker_pid}" "${SAMPLE_DURATION_SECONDS}" 1 >"${sample_file}" 2>&1 || true
      fi
      kill -TERM "${worker_pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${worker_pid}" 2>/dev/null || true
    fi
  ) &
  local watchdog_pid=$!

  if wait "${worker_pid}"; then
    status=0
  else
    status=$?
  fi

  if kill -0 "${watchdog_pid}" 2>/dev/null; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  else
    wait "${watchdog_pid}" 2>/dev/null || true
  fi

  if [[ "${status}" -eq 143 || "${status}" -eq 137 ]]; then
    timed_out=1
  fi

  if [[ "${status}" -eq 0 ]]; then
    cat "${log_file}"
    rm -f "${log_file}"
    return 0
  fi

  echo "${label}: failed"
  if [[ "${timed_out}" -eq 1 ]]; then
    echo "  reason: timed out after ${TIMEOUT_SECONDS}s"
    if [[ -n "${sample_file}" ]]; then
      echo "  sample: ${sample_file}"
    fi
  fi
  cat "${log_file}" 2>/dev/null || true
  rm -f "${log_file}"
  FAILURES=$((FAILURES + 1))
}

echo "python=${PYTHON_CMD}"
echo "timeout_seconds=${TIMEOUT_SECONDS}"
echo "capture_stack_sample_on_timeout=${CAPTURE_STACK_SAMPLE_ON_TIMEOUT}"
if [[ "${CAPTURE_STACK_SAMPLE_ON_TIMEOUT}" == "1" ]]; then
  echo "sample_duration_seconds=${SAMPLE_DURATION_SECONDS}"
  echo "sample_dir=${SAMPLE_DIR}"
fi

run_python_check "startup" -V
run_python_check "math" -c "import math; print('math: ok')"
run_python_check "ssl" -c "import ssl; print('ssl: ok')"
run_python_check "yaml" -c "import yaml; print('yaml: ok')"
run_python_check "httpx" -c "import httpx; print('httpx: ok')"
run_python_check "pytest" -c "import pytest; print('pytest: ok')"

if [[ "${FAILURES}" -gt 0 ]]; then
  echo "runtime diagnostics failed (${FAILURES})"
  exit 1
fi

echo "runtime diagnostics passed"
