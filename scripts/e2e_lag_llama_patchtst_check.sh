#!/usr/bin/env bash
set -euo pipefail

WITH_SUCCESS=0
if [[ "${1:-}" == "--with-success" ]]; then
  WITH_SUCCESS=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${ROOT_DIR}/.venv/bin"
if [[ -d "${VENV_BIN}" ]]; then
  export PATH="${VENV_BIN}:$PATH"
fi

if [[ -x "${VENV_BIN}/python" ]]; then
  PYTHON_BIN="${VENV_BIN}/python"
else
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -x "${VENV_BIN}/pytest" ]]; then
  PYTEST_BIN="${VENV_BIN}/pytest"
else
  PYTEST_BIN="$(command -v pytest || true)"
fi

if [[ -x "${VENV_BIN}/tollama" ]]; then
  TOLLAMA_BIN="${VENV_BIN}/tollama"
else
  TOLLAMA_BIN="$(command -v tollama || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python not found"
  exit 1
fi

if [[ -z "${PYTEST_BIN}" ]]; then
  echo "pytest not found"
  exit 1
fi

echo "[1/2] Running targeted regression tests for Lag-Llama + PatchTST runners"
PYTHONPATH=src "${PYTEST_BIN}" -q \
  tests/test_lag_llama_runner.py \
  tests/test_patchtst_runner.py \
  tests/test_runner_manager.py

if [[ "$WITH_SUCCESS" -eq 1 ]]; then
  if [[ -z "${TOLLAMA_BIN}" ]]; then
    echo "tollama not found; skipping success-path smoke"
    exit 0
  fi

  echo "[2/2] Running optional local success-path smoke"
  "${TOLLAMA_BIN}" run lag-llama --input examples/lag_llama_request.json --no-stream >/dev/null
  "${TOLLAMA_BIN}" run patchtst --input examples/request.json --no-stream >/dev/null
  echo "Optional success-path smoke completed"
else
  echo "[2/2] Skipped success-path smoke (pass --with-success to enable)"
fi
