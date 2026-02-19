#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://localhost:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DEP_MISSING=3
EXIT_DAEMON_UNREACHABLE=5

BASE_URL_ARG=""
TIMEOUT_ARG=""

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

http_get() {
  local path="$1"
  local body_file err_file

  body_file="$(mktemp)"
  err_file="$(mktemp)"

  if ! HTTP_STATUS="$(curl -sS --connect-timeout "$TIMEOUT" --max-time "$TIMEOUT" -o "$body_file" -w '%{http_code}' "${BASE_URL}${path}" 2>"$err_file")"; then
    HTTP_ERROR="$(cat "$err_file")"
    rm -f "$body_file" "$err_file"
    return 1
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
  exit "$EXIT_DEP_MISSING"
fi

if ! http_get "/v1/health"; then
  echo "Error: GET ${BASE_URL}/v1/health failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

HEALTH_STATUS="$HTTP_STATUS"
HEALTH_BODY="$HTTP_BODY"

if [[ ! "$HEALTH_STATUS" =~ ^2 ]]; then
  echo "Error: GET ${BASE_URL}/v1/health returned HTTP $HEALTH_STATUS: $(normalize_detail "$HEALTH_BODY")" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

if ! http_get "/api/version"; then
  echo "Error: GET ${BASE_URL}/api/version failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

VERSION_STATUS="$HTTP_STATUS"
VERSION_BODY="$HTTP_BODY"

if [[ ! "$VERSION_STATUS" =~ ^2 ]]; then
  echo "Error: GET ${BASE_URL}/api/version returned HTTP $VERSION_STATUS: $(normalize_detail "$VERSION_BODY")" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

printf '{"base_url":"%s","timeout_seconds":%s,"health":{"status":%s},"version":{"status":%s}}\n' \
  "$BASE_URL" "$TIMEOUT" "$HEALTH_STATUS" "$VERSION_STATUS"
