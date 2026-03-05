#!/usr/bin/env bash
set -euo pipefail

WITH_SUCCESS=0
if [[ "${1:-}" == "--with-success" ]]; then
  WITH_SUCCESS=1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found"
  exit 1
fi

if ! command -v pytest >/dev/null 2>&1; then
  echo "pytest not found"
  exit 1
fi

echo "[1/2] Running targeted regression tests for Lag-Llama + PatchTST runners"
PYTHONPATH=src pytest -q \
  tests/test_lag_llama_runner.py \
  tests/test_patchtst_runner.py \
  tests/test_runner_manager.py

if [[ "$WITH_SUCCESS" -eq 1 ]]; then
  if ! command -v tollama >/dev/null 2>&1; then
    echo "tollama not found; skipping success-path smoke"
    exit 0
  fi

  echo "[2/2] Running optional local success-path smoke"
  tollama run lag-llama --input examples/lag_llama_request.json --no-stream >/dev/null
  tollama run patchtst --input examples/request.json --no-stream >/dev/null
  echo "Optional success-path smoke completed"
else
  echo "[2/2] Skipped success-path smoke (pass --with-success to enable)"
fi
