#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://127.0.0.1:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DAEMON_UNREACHABLE=3
EXIT_PERMISSION=5
EXIT_TIMEOUT=6
EXIT_INTERNAL=10

BASE_URL_ARG=""
TIMEOUT_ARG=""
HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

usage() {
  cat <<'USAGE' >&2
Usage: tollama-health.sh [--base-url URL] [--timeout SEC]
USAGE
}

is_positive_number() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v v="$value" 'BEGIN { exit !(v > 0) }'
}

normalize_detail() {
  local raw="$1"
  local flattened
  flattened="${raw//$'\n'/ }"
  printf '%s' "${flattened:0:500}"
}

print_host_mismatch_hint() {
  echo "Hint: exec host and --base-url may be mismatched. 127.0.0.1 inside sandbox/container is not host localhost." >&2
}

is_timeout_error() {
  local text="$1"
  [[ "$text" == *"timed out"* ]] ||
  [[ "$text" == *"timeout"* ]] ||
  [[ "$text" == *"operation timeout"* ]]
}

is_connection_error() {
  local text="$1"
  [[ "$text" == *"Connection refused"* ]] ||
  [[ "$text" == *"Failed to connect"* ]] ||
  [[ "$text" == *"No route to host"* ]] ||
  [[ "$text" == *"Could not resolve host"* ]] ||
  [[ "$text" == *"Name or service not known"* ]] ||
  [[ "$text" == *"Temporary failure in name resolution"* ]]
}

classify_curl_error() {
  local curl_exit="$1"
  local err_text="$2"

  if [[ "$curl_exit" -eq 28 ]] || is_timeout_error "$err_text"; then
    echo "$EXIT_TIMEOUT"
    return
  fi

  if is_connection_error "$err_text"; then
    echo "$EXIT_DAEMON_UNREACHABLE"
    return
  fi

  echo "$EXIT_INTERNAL"
}

classify_http_status() {
  local status="$1"

  case "$status" in
    401|403)
      echo "$EXIT_PERMISSION"
      ;;
    408|504)
      echo "$EXIT_TIMEOUT"
      ;;
    *)
      echo "$EXIT_DAEMON_UNREACHABLE"
      ;;
  esac
}

http_get() {
  local path="$1"
  local body_file err_file
  local curl_exit

  body_file="$(mktemp)"
  err_file="$(mktemp)"

  if HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
    :
  else
    curl_exit=$?
    HTTP_ERROR="$(cat "$err_file")"
    rm -f "$body_file" "$err_file"
    return "$curl_exit"
  fi

  HTTP_BODY="$(cat "$body_file")"
  HTTP_ERROR=""
  rm -f "$body_file" "$err_file"
  return 0
}

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

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl not found in PATH." >&2
  exit "$EXIT_INTERNAL"
fi

if http_get "/v1/health"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  echo "Error: GET ${BASE_URL}/v1/health failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$exit_code"
fi

HEALTH_STATUS="$HTTP_STATUS"
HEALTH_BODY="$HTTP_BODY"

if [[ ! "$HEALTH_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status "$HEALTH_STATUS")"
  echo "Error: GET ${BASE_URL}/v1/health returned HTTP $HEALTH_STATUS: $(normalize_detail "$HEALTH_BODY")" >&2
  print_host_mismatch_hint
  exit "$exit_code"
fi

if http_get "/api/version"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  echo "Error: GET ${BASE_URL}/api/version failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$exit_code"
fi

VERSION_STATUS="$HTTP_STATUS"
VERSION_BODY="$HTTP_BODY"

if [[ ! "$VERSION_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status "$VERSION_STATUS")"
  echo "Error: GET ${BASE_URL}/api/version returned HTTP $VERSION_STATUS: $(normalize_detail "$VERSION_BODY")" >&2
  print_host_mismatch_hint
  exit "$exit_code"
fi

printf '{"base_url":"%s","timeout_seconds":%s,"health":{"status":%s},"version":{"status":%s}}\n' \
  "$BASE_URL" "$TIMEOUT" "$HEALTH_STATUS" "$VERSION_STATUS"
