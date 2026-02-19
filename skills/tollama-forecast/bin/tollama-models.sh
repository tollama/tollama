#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://127.0.0.1:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DAEMON_UNREACHABLE=3
EXIT_MODEL_MISSING=4
EXIT_PERMISSION=5
EXIT_TIMEOUT=6
EXIT_INTERNAL=10

COMMAND=""
MODEL=""
BASE_URL_ARG=""
TIMEOUT_ARG=""
HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

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

normalize_detail() {
  local raw="$1"
  local flattened
  flattened="${raw//$'\n'/ }"
  printf '%s' "${flattened:0:500}"
}

is_timeout_error() {
  local text="$1"
  [[ "$text" == *"timed out"* ]] ||
  [[ "$text" == *"timeout"* ]] ||
  [[ "$text" == *"operation timeout"* ]]
}

is_daemon_error() {
  local text="$1"
  [[ "$text" == *"Connection refused"* ]] ||
  [[ "$text" == *"Failed to connect"* ]] ||
  [[ "$text" == *"No route to host"* ]] ||
  [[ "$text" == *"Could not resolve host"* ]] ||
  [[ "$text" == *"Name or service not known"* ]] ||
  [[ "$text" == *"Temporary failure in name resolution"* ]]
}

is_permission_error() {
  local text="$1"
  local lower
  lower="$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')"
  [[ "$lower" == *"permission"* ]] ||
  [[ "$lower" == *"forbidden"* ]] ||
  [[ "$lower" == *"unauthorized"* ]] ||
  [[ "$lower" == *"access denied"* ]]
}

is_license_error() {
  local text="$1"
  local lower
  lower="$(printf '%s' "$text" | tr '[:upper:]' '[:lower:]')"
  [[ "$lower" == *"license"* ]] || [[ "$lower" == *"accept-license"* ]]
}

extract_http_status_from_text() {
  local text="$1"
  local status
  status="$(printf '%s' "$text" | grep -Eo 'HTTP [0-9]{3}' | awk '{print $2}' | tail -n1 || true)"
  if [[ "$status" =~ ^[0-9]{3}$ ]]; then
    printf '%s' "$status"
    return 0
  fi
  return 1
}

classify_curl_error() {
  local curl_exit="$1"
  local err_text="$2"

  if [[ "$curl_exit" -eq 28 ]] || is_timeout_error "$err_text"; then
    echo "$EXIT_TIMEOUT"
    return
  fi

  if is_daemon_error "$err_text"; then
    echo "$EXIT_DAEMON_UNREACHABLE"
    return
  fi

  echo "$EXIT_INTERNAL"
}

classify_http_status() {
  local context="$1"
  local status="$2"

  case "$status" in
    400)
      echo "$EXIT_USAGE"
      return
      ;;
    401|403)
      echo "$EXIT_PERMISSION"
      return
      ;;
    404)
      if [[ "$context" == "show" ]]; then
        echo "$EXIT_MODEL_MISSING"
      else
        echo "$EXIT_INTERNAL"
      fi
      return
      ;;
    408|504)
      echo "$EXIT_TIMEOUT"
      return
      ;;
    409)
      echo "$EXIT_PERMISSION"
      return
      ;;
    *)
      echo "$EXIT_INTERNAL"
      return
      ;;
  esac
}

classify_text_error() {
  local context="$1"
  local err_text="$2"

  if is_timeout_error "$err_text"; then
    echo "$EXIT_TIMEOUT"
    return
  fi

  if is_daemon_error "$err_text"; then
    echo "$EXIT_DAEMON_UNREACHABLE"
    return
  fi

  if is_permission_error "$err_text" || is_license_error "$err_text"; then
    echo "$EXIT_PERMISSION"
    return
  fi

  status="$(extract_http_status_from_text "$err_text" || true)"
  if [[ -n "$status" ]]; then
    classify_http_status "$context" "$status"
    return
  fi

  echo "$EXIT_INTERNAL"
}

http_request() {
  local method="$1"
  local path="$2"
  local payload="${3:-}"
  local body_file err_file
  local curl_exit

  body_file="$(mktemp)"
  err_file="$(mktemp)"

  if [[ -n "$payload" ]]; then
    if HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -X "$method" -H 'content-type: application/json' --data "$payload" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
      :
    else
      curl_exit=$?
      HTTP_ERROR="$(cat "$err_file")"
      rm -f "$body_file" "$err_file"
      return "$curl_exit"
    fi
  else
    if HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -X "$method" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
      :
    else
      curl_exit=$?
      HTTP_ERROR="$(cat "$err_file")"
      rm -f "$body_file" "$err_file"
      return "$curl_exit"
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
    return "$EXIT_INTERNAL"
  fi

  if ! python3 - <<'PY'; then
import json
import sys

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError as exc:
    print(f"Error: invalid JSON payload: {exc}", file=sys.stderr)
    raise SystemExit(10)

models = payload.get("models") if isinstance(payload, dict) else None
available = []
if isinstance(models, dict):
    available = models.get("available")
if not isinstance(available, list):
    available = []

json.dump({"models": available}, sys.stdout, sort_keys=True)
sys.stdout.write("\n")
PY
    return "$EXIT_INTERNAL"
  fi

  return 0
}

run_tollama_cmd() {
  local context="$1"
  shift

  local out_file err_file
  out_file="$(mktemp)"
  err_file="$(mktemp)"

  if "$@" >"$out_file" 2>"$err_file"; then
    cat "$out_file"
    rm -f "$out_file" "$err_file"
    return 0
  fi

  local err_text
  local exit_code
  err_text="$(cat "$err_file")"
  rm -f "$out_file" "$err_file"

  echo "$err_text" >&2
  exit_code="$(classify_text_error "$context" "$err_text")"
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    print_host_mismatch_hint
  fi
  return "$exit_code"
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
      if run_tollama_cmd installed tollama list --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_INTERNAL"
    fi

    if http_request GET "/api/tags"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      echo "Error: GET ${BASE_URL}/api/tags failed: ${HTTP_ERROR:-unknown error}" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status installed "$HTTP_STATUS")"
    echo "Error: GET ${BASE_URL}/api/tags returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$exit_code"
    ;;

  loaded)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd loaded tollama ps --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_INTERNAL"
    fi

    if http_request GET "/api/ps"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      echo "Error: GET ${BASE_URL}/api/ps failed: ${HTTP_ERROR:-unknown error}" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status loaded "$HTTP_STATUS")"
    echo "Error: GET ${BASE_URL}/api/ps returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$exit_code"
    ;;

  show)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd show tollama show "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        rc=$?
        exit "$rc"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_INTERNAL"
    fi

    if http_request POST "/api/show" "{\"model\":\"$MODEL\"}"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      echo "Error: POST ${BASE_URL}/api/show failed: ${HTTP_ERROR:-unknown error}" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status show "$HTTP_STATUS")"
    echo "Error: POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$exit_code"
    ;;

  available)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      tmp_out="$(mktemp)"
      tmp_err="$(mktemp)"

      if tollama info --json --remote --base-url "$BASE_URL" --timeout "$TIMEOUT" >"$tmp_out" 2>"$tmp_err"; then
        if extract_available_models <"$tmp_out"; then
          :
        else
          rc=$?
          rm -f "$tmp_out" "$tmp_err"
          exit "$rc"
        fi
        rm -f "$tmp_out" "$tmp_err"
        exit 0
      fi

      err_text="$(cat "$tmp_err")"
      rm -f "$tmp_out" "$tmp_err"

      exit_code="$(classify_text_error available "$err_text")"
      echo "Error: tollama info --json --remote failed: $err_text" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: neither tollama nor curl is available in PATH." >&2
      exit "$EXIT_INTERNAL"
    fi

    if http_request GET "/api/info"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      echo "Error: GET ${BASE_URL}/api/info failed: ${HTTP_ERROR:-unknown error}" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
      exit_code="$(classify_http_status available "$HTTP_STATUS")"
      echo "Error: GET ${BASE_URL}/api/info returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        print_host_mismatch_hint
      fi
      exit "$exit_code"
    fi

    if extract_available_models <<<"$HTTP_BODY"; then
      :
    else
      rc=$?
      exit "$rc"
    fi
    exit 0
    ;;
esac
