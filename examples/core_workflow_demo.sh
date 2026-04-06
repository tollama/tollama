#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

BASE_URL="${TOLLAMA_BASE_URL:-http://127.0.0.1:11435}"
MODEL="${MODEL:-mock}"
QUICKSTART_HORIZON="${QUICKSTART_HORIZON:-3}"
BENCHMARK_HORIZON="${BENCHMARK_HORIZON:-4}"
FOLDS="${FOLDS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/benchmarks/core-demo}"
BENCHMARK_DATA="${BENCHMARK_DATA:-examples/benchmark_data.json}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN_ALT:-python}"
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Python 3.11+ is required to run the Tollama Core demo." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "== Tollama Core demo =="
echo "Repo root: ${PROJECT_ROOT}"
echo "Base URL: ${BASE_URL}"
echo "Model: ${MODEL}"
echo ""

echo "[1/4] preprocess irregular data"
"${PYTHON_BIN}" - <<'PY'
import numpy as np

from tollama.preprocess import PreprocessConfig, run_pipeline

x = np.arange(48, dtype=float)
y = np.sin(x * 0.15) * 10
y[[7, 19, 33]] = np.nan

result = run_pipeline(x, y, config=PreprocessConfig(lookback=12, horizon=4))
print(f"preprocess windows: X={result.X.shape}, y={result.y.shape}")
PY
echo ""

echo "[2/4] forecast via quickstart"
"${PYTHON_BIN}" -m tollama.cli.main quickstart \
  --model "${MODEL}" \
  --horizon "${QUICKSTART_HORIZON}" \
  --base-url "${BASE_URL}"
echo ""

echo "[3/4] benchmark and write Core artifacts"
"${PYTHON_BIN}" -m tollama.cli.main benchmark \
  "${BENCHMARK_DATA}" \
  --models "${MODEL}" \
  --horizon "${BENCHMARK_HORIZON}" \
  --folds "${FOLDS}" \
  --output "${OUTPUT_DIR}"
echo ""

echo "[4/4] apply routing evidence"
"${PYTHON_BIN}" -m tollama.cli.main routing apply "${OUTPUT_DIR}/result.json"
"${PYTHON_BIN}" -m tollama.cli.main routing show
echo ""

echo "Core demo complete."
echo "Artifacts:"
echo "  - ${OUTPUT_DIR}/result.json"
echo "  - ${OUTPUT_DIR}/routing.json"
echo "  - ${OUTPUT_DIR}/leaderboard.csv"
