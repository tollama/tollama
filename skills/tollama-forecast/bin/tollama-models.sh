#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://localhost:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DEP_MISSING=3
EXIT_DAEMON_UNREACHABLE=5
EXIT_REQUEST_FAILED=6

COMMAND=""
MODEL=""
BASE_URL_ARG=""
TIMEOUT_ARG=""

usage() {
  cat <<'USAGE' >&2
Usage:
  tollama-models.sh installed [--base-url URL] [--timeout SEC]
  tollama-models.sh loaded [--base-url URL] [--timeout SEC]
  tollama-models.sh show <model> [--base-url URL] [--timeout SEC]
  tollama-models.sh available [--base-url URL] [--timeout SEC]
USAGE
}

is_positive_number() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v v="$value" 'BEGIN { exit !(v > 0) }'
}

print_host_mismatch_hint() {
  echo "Hint: exec host and --base-url may be mismatched. 127.0.0.1 inside sandbox/container is not host localhost." >&2
}

is_daemon_error() {
  local text="$1"
  [[ "$text" == *"Connection refused"* ]] || \
  [[ "$text" == *"Failed to connect"* ]] || \
  [[ "$text" == *"timed out"* ]] || \
  [[ "$text" == *"Name or service not known"* ]] || \
  [[ "$text" == *"Temporary failure in name resolution"* ]] || \
  [[ "$text" == *"No route to host"* ]]
}

normalize_detail() {
  local raw="$1"
  local flattened
  flattened="${raw//$'\n'/ }"
  printf '%s' "${flattened:0:500}"
}

http_request() {
  local method="$1"
  local path="$2"
  local payload="${3:-}"
  local body_file err_file

  body_file="$(mktemp)"
  err_file="$(mktemp)"

  if [[ -n "$payload" ]]; then
    if ! HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -X "$method" -H 'content-type: application/json' --data "$payload" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
      HTTP_ERROR="$(cat "$err_file")"
      rm -f "$body_file" "$err_file"
      return 1
    fi
  else
    if ! HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -X "$method" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
      HTTP_ERROR="$(cat "$err_file")"
      rm -f "$body_file" "$err_file"
      return 1
    fi
  fi

  HTTP_BODY="$(cat "$body_file")"
  HTTP_ERROR=""
  rm -f "$body_file" "$err_file"
  return 0
}

extract_available_models() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required to extract models.available from /api/info." >&2
    exit "$EXIT_DEP_MISSING"
  fi

  python3 - <<'PY'
import json
import sys

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError as exc:
    print(f"Error: invalid JSON payload: {exc}", file=sys.stderr)
    sys.exit(2)

models = payload.get("models") if isinstance(payload, dict) else None
available = []
if isinstance(models, dict):
    available = models.get("available")
if not isinstance(available, list):
    available = []

json.dump({"models": available}, sys.stdout, sort_keys=True)
sys.stdout.write("\n")
PY
}

run_tollama_cmd() {
  local out_file err_file
  out_file="$(mktemp)"
  err_file="$(mktemp)"

  if "$@" >"$out_file" 2>"$err_file"; then
    cat "$out_file"
    rm -f "$out_file" "$err_file"
    return 0
  fi

  local err_text
  err_text="$(cat "$err_file")"
  rm -f "$out_file" "$err_file"

  echo "$err_text" >&2
  if is_daemon_error "$err_text"; then
    print_host_mismatch_hint
    return "$EXIT_DAEMON_UNREACHABLE"
  fi
  return "$EXIT_REQUEST_FAILED"
}

if [[ $# -lt 1 ]]; then
  usage
  exit "$EXIT_USAGE"
fi

COMMAND="$1"
shift

case "$COMMAND" in
  installed|loaded|available)
    ;;
  show)
    if [[ $# -lt 1 ]] || [[ "$1" == --* ]]; then
      usage
      exit "$EXIT_USAGE"
    fi
    MODEL="$1"
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage
    exit "$EXIT_USAGE"
    ;;
esac

while (($# > 0)); do
  case "$1" in
    --base-url)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      BASE_URL_ARG="$1"
      ;;
    --timeout)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      TIMEOUT_ARG="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit "$EXIT_USAGE"
      ;;
  esac
  shift
done

BASE_URL="${BASE_URL_ARG:-${TOLLAMA_BASE_URL:-$DEFAULT_BASE_URL}}"
TIMEOUT="${TIMEOUT_ARG:-${TOLLAMA_FORECAST_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT}}"

if ! is_positive_number "$TIMEOUT"; then
  echo "Error: timeout must be a positive number, got '$TIMEOUT'." >&2
  exit "$EXIT_USAGE"
fi

HAS_TOLLAMA=0
if command -v tollama >/dev/null 2>&1; then
  HAS_TOLLAMA=1
fi

case "$COMMAND" in
  installed)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if ! run_tollama_cmd tollama list --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_DEP_MISSING"
    fi

    if ! http_request GET "/api/tags"; then
      echo "Error: GET ${BASE_URL}/api/tags failed: ${HTTP_ERROR:-unknown error}" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    echo "Error: GET ${BASE_URL}/api/tags returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$EXIT_REQUEST_FAILED"
    ;;

  loaded)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if ! run_tollama_cmd tollama ps --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_DEP_MISSING"
    fi

    if ! http_request GET "/api/ps"; then
      echo "Error: GET ${BASE_URL}/api/ps failed: ${HTTP_ERROR:-unknown error}" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    echo "Error: GET ${BASE_URL}/api/ps returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$EXIT_REQUEST_FAILED"
    ;;

  show)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if ! run_tollama_cmd tollama show "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_DEP_MISSING"
    fi

    if ! http_request POST "/api/show" "{\"model\":\"$MODEL\"}"; then
      echo "Error: POST ${BASE_URL}/api/show failed: ${HTTP_ERROR:-unknown error}" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    echo "Error: POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$EXIT_REQUEST_FAILED"
    ;;

  available)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      tmp_out="$(mktemp)"
      tmp_err="$(mktemp)"
      if tollama info --json --remote --base-url "$BASE_URL" --timeout "$TIMEOUT" >"$tmp_out" 2>"$tmp_err"; then
        if ! extract_available_models <"$tmp_out"; then
          rm -f "$tmp_out" "$tmp_err"
          exit "$EXIT_REQUEST_FAILED"
        fi
        rm -f "$tmp_out" "$tmp_err"
        exit 0
      fi

      err_text="$(cat "$tmp_err")"
      rm -f "$tmp_out" "$tmp_err"
      echo "Error: tollama info --json --remote failed: $err_text" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_DEP_MISSING"
    fi

    if ! http_request GET "/api/info"; then
      echo "Error: GET ${BASE_URL}/api/info failed: ${HTTP_ERROR:-unknown error}" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
      echo "Error: GET ${BASE_URL}/api/info returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
      print_host_mismatch_hint
      exit "$EXIT_DAEMON_UNREACHABLE"
    fi

    if ! extract_available_models <<<"$HTTP_BODY"; then
      exit "$EXIT_REQUEST_FAILED"
    fi
    exit 0
    ;;
esac
