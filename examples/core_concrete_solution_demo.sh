#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN_ALT:-python}"
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
  echo "Python 3.11+ is required to run the Tollama Core concrete-solution demo." >&2
  exit 1
fi

BASE_URL="${TOLLAMA_BASE_URL:-http://127.0.0.1:11435}"
MODE="${MODE:-local}"
DATASET="${DATASET:-pjm_hourly_energy}"
FALLBACK_DATASET="${FALLBACK_DATASET:-m4_daily}"
MODELS="${MODELS:-chronos2,granite-ttm-r2,timesfm-2.5-200m,moirai-2.0-R-small}"
FOLDS="${FOLDS:-2}"
INPUT_PATH="${INPUT_PATH:-artifacts/core-solution/benchmark_input.json}"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-artifacts/core-solution/benchmark}"

ALLOW_KAGGLE_FALLBACK_FLAG=()
if [[ "${ALLOW_KAGGLE_FALLBACK:-1}" != "0" ]]; then
  ALLOW_KAGGLE_FALLBACK_FLAG+=(--allow-kaggle-fallback)
fi

echo "== Tollama Core concrete-solution demo =="
echo "Repo root: ${PROJECT_ROOT}"
echo "Base URL: ${BASE_URL}"
echo "Mode: ${MODE}"
echo "Preferred dataset: ${DATASET}"
echo "Fallback dataset: ${FALLBACK_DATASET}"
echo "Models: ${MODELS}"
echo ""

echo "[1/3] export prepared benchmark input"
"${PYTHON_BIN}" scripts/e2e_realdata/export_core_solution_input.py \
  --mode "${MODE}" \
  --dataset "${DATASET}" \
  --fallback-dataset "${FALLBACK_DATASET}" \
  "${ALLOW_KAGGLE_FALLBACK_FLAG[@]}" \
  --output "${INPUT_PATH}"
echo ""

BENCHMARK_HORIZON="${BENCHMARK_HORIZON:-$("${PYTHON_BIN}" -c 'import json, sys; print(json.loads(open(sys.argv[1], encoding="utf-8").read())["recommended_horizon"])' "${INPUT_PATH}")}"

echo "[2/3] benchmark curated models"
"${PYTHON_BIN}" -m tollama.cli.main benchmark \
  "${INPUT_PATH}" \
  --models "${MODELS}" \
  --horizon "${BENCHMARK_HORIZON}" \
  --folds "${FOLDS}" \
  --base-url "${BASE_URL}" \
  --output "${BENCHMARK_OUTPUT_DIR}"
echo ""

echo "[3/3] apply routing evidence"
"${PYTHON_BIN}" -m tollama.cli.main routing apply "${BENCHMARK_OUTPUT_DIR}/result.json"
"${PYTHON_BIN}" -m tollama.cli.main routing show
echo ""

echo "Concrete solution demo complete."
echo "Artifacts:"
echo "  - ${INPUT_PATH}"
echo "  - ${BENCHMARK_OUTPUT_DIR}/result.json"
echo "  - ${BENCHMARK_OUTPUT_DIR}/routing.json"
echo "  - ${BENCHMARK_OUTPUT_DIR}/leaderboard.csv"
