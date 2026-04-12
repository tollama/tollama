#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pr}"
MODEL="${2:-all}"
BASE_URL="${3:-http://127.0.0.1:11435}"
OUTPUT_DIR="${4:-$ROOT_DIR/artifacts/e2e_realdata}"
ALLOW_KAGGLE_FALLBACK="${5:-false}"
GATE_PROFILE="${6:-strict}"
EXTRA_ARGS=()
VENV_BIN="${ROOT_DIR}/.venv/bin"
PYTHON_PROBE_TIMEOUT="${PYTHON_PROBE_TIMEOUT:-5}"
PYTHON_PROBE_CODE='import ssl, yaml, httpx; print("python-runtime-ok")'
PYTHON_PROBE_OUTPUT=""
PYTHON_PROBE_REASON=""

if [[ -d "${VENV_BIN}" ]]; then
  export PATH="${VENV_BIN}:$PATH"
fi

probe_python_cmd() {
  local python_cmd="$1"
  shift
  local log_file
  local worker_pid
  local watchdog_pid
  local status

  PYTHON_PROBE_OUTPUT=""
  PYTHON_PROBE_REASON=""
  log_file="$(mktemp -t tollama-python-probe.XXXXXX)"

  "$python_cmd" "$@" >"${log_file}" 2>&1 &
  worker_pid=$!
  (
    sleep "${PYTHON_PROBE_TIMEOUT}"
    if kill -0 "${worker_pid}" 2>/dev/null; then
      kill -TERM "${worker_pid}" 2>/dev/null || true
      sleep 1
      kill -KILL "${worker_pid}" 2>/dev/null || true
    fi
  ) &
  watchdog_pid=$!

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

  PYTHON_PROBE_OUTPUT="$(cat "${log_file}" 2>/dev/null || true)"
  rm -f "${log_file}"

  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi

  if [[ "${status}" -eq 143 || "${status}" -eq 137 ]]; then
    PYTHON_PROBE_REASON="timed out after ${PYTHON_PROBE_TIMEOUT}s"
  elif [[ -n "${PYTHON_PROBE_OUTPUT}" ]]; then
    PYTHON_PROBE_REASON="${PYTHON_PROBE_OUTPUT}"
  else
    PYTHON_PROBE_REASON="command exited with status ${status}"
  fi
  return 1
}

select_python_cmd() {
  local raw_candidates=()
  local seen=""
  local candidate
  local candidate_key
  local failures=()

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    raw_candidates+=("${PYTHON_BIN}")
  else
    raw_candidates+=("${VENV_BIN}/python")
    if command -v python >/dev/null 2>&1; then
      raw_candidates+=("$(command -v python)")
    fi
    if command -v python3 >/dev/null 2>&1; then
      raw_candidates+=("$(command -v python3)")
    fi
    raw_candidates+=("/usr/bin/python3")
  fi

  for candidate in "${raw_candidates[@]}"; do
    if [[ -z "${candidate}" || ! -x "${candidate}" ]]; then
      continue
    fi
    candidate_key="$(stat -Lf '%d:%i' "${candidate}" 2>/dev/null || printf '%s' "${candidate}")"
    case " ${seen} " in
      *" ${candidate_key} "*) continue ;;
    esac
    seen="${seen} ${candidate_key}"

    if [[ "${PYTHON_SKIP_PROBE:-0}" == "1" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi

    if ! probe_python_cmd "${candidate}" -V; then
      failures+=("${candidate}: startup probe failed (${PYTHON_PROBE_REASON})")
      continue
    fi
    if ! probe_python_cmd "${candidate}" -c "${PYTHON_PROBE_CODE}"; then
      failures+=("${candidate}: runtime probe failed (${PYTHON_PROBE_REASON})")
      continue
    fi

    printf '%s\n' "${candidate}"
    return 0
  done

  printf '%s\n' "${failures[@]}" >&2
  return 1
}

if ! PYTHON_CMD="$(select_python_cmd)"; then
  echo -e "${RED}Error: no healthy python interpreter found for the real-data wrapper.${NC}"
  echo "Try: bash scripts/e2e_realdata_runtime_diag.sh"
  echo "Or force a specific interpreter: PYTHON_BIN=/path/to/python bash scripts/e2e_realdata_tsfm.sh ..."
  echo "Set PYTHON_SKIP_PROBE=1 only if you intentionally want to bypass startup/runtime preflight."
  exit 1
fi

echo -e "${YELLOW}Running real-data TSFM E2E suite...${NC}"
echo "mode=$MODE model=$MODEL base_url=$BASE_URL"
echo "output_dir=$OUTPUT_DIR"
echo "allow_kaggle_fallback=$ALLOW_KAGGLE_FALLBACK"
echo "gate_profile=$GATE_PROFILE"
echo "python=$PYTHON_CMD"

if [[ "$ALLOW_KAGGLE_FALLBACK" == "true" ]]; then
  EXTRA_ARGS+=(--allow-kaggle-fallback)
fi

"$PYTHON_CMD" "$ROOT_DIR/scripts/e2e_realdata/run_tsfm_realdata.py" \
  --mode "$MODE" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --gate-profile "$GATE_PROFILE" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

status=$?
if [[ $status -eq 0 ]]; then
  echo -e "${GREEN}Real-data TSFM E2E passed.${NC}"
elif [[ $status -eq 2 ]]; then
  echo -e "${RED}Real-data TSFM E2E failed due to infra/preflight issues.${NC}"
else
  echo -e "${RED}Real-data TSFM E2E gate failed.${NC}"
fi

exit $status
