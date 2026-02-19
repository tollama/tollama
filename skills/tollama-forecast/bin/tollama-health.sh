#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://127.0.0.1:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DAEMON_UNREACHABLE=3
EXIT_PERMISSION=5
EXIT_TIMEOUT=6
EXIT_INTERNAL=10

SKILL_BIN_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
# shellcheck source=skills/tollama-forecast/bin/_tollama_lib.sh
source "$SKILL_BIN_DIR/_tollama_lib.sh"

BASE_URL_ARG=""
TIMEOUT_ARG=""
INCLUDE_RUNTIMES=0
HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

usage() {
  cat <<'USAGE' >&2
Usage: tollama-health.sh [--base-url URL] [--timeout SEC] [--runtimes]
USAGE
}

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    emit_error "$EXIT_INTERNAL" "python3 is required for JSON parsing in tollama-health.sh."
    exit "$EXIT_INTERNAL"
  fi
}

extract_version_name() {
  local payload="$1"
  require_python3

  if ! python3 - "$payload" <<'PY'; then
import json
import sys

version = ""
try:
    body = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)

if isinstance(body, dict):
    raw = body.get("version")
    if isinstance(raw, str):
        version = raw

print(version)
PY
    return "$EXIT_INTERNAL"
  fi

  return 0
}

extract_runtimes_json() {
  local payload="$1"
  require_python3

  if ! python3 - "$payload" <<'PY'; then
import json
import sys

try:
    body = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print("[]")
    raise SystemExit(0)

runners = body.get("runners") if isinstance(body, dict) else None
if not isinstance(runners, list):
    runners = []

normalized = []
for runner in runners:
    if not isinstance(runner, dict):
        continue
    normalized.append(
        {
            "family": runner.get("family"),
            "installed": bool(runner.get("installed")),
            "running": bool(runner.get("running")),
        }
    )

print(json.dumps(normalized, separators=(",", ":"), sort_keys=True))
PY
    return "$EXIT_INTERNAL"
  fi

  return 0
}

emit_success_payload() {
  local version_name="$1"
  local runtimes_json="$2"

  require_python3

  if ! python3 - "$BASE_URL" "$TIMEOUT" "$HEALTH_STATUS" "$VERSION_STATUS" "$version_name" "$INCLUDE_RUNTIMES" "$runtimes_json" <<'PY'; then
import json
import sys

base_url = sys.argv[1]
timeout_raw = sys.argv[2]
health_status = int(sys.argv[3])
version_status = int(sys.argv[4])
version_name = sys.argv[5]
include_runtimes = sys.argv[6] == "1"
runtimes_json = sys.argv[7]

try:
    timeout_value = int(timeout_raw)
except ValueError:
    timeout_value = float(timeout_raw)

payload = {
    "healthy": True,
    "base_url": base_url,
    "timeout_seconds": timeout_value,
    "health": {"status": health_status},
    "version": {"status": version_status},
    "version_name": version_name,
}

if include_runtimes:
    try:
        payload["runtimes"] = json.loads(runtimes_json)
    except json.JSONDecodeError:
        payload["runtimes"] = []

print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
PY
    emit_error "$EXIT_INTERNAL" "failed to format tollama-health response JSON."
    exit "$EXIT_INTERNAL"
  fi
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
    --runtimes)
      INCLUDE_RUNTIMES=1
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
  emit_error "$EXIT_USAGE" "timeout must be a positive number, got '$TIMEOUT'."
  exit "$EXIT_USAGE"
fi

if ! command -v curl >/dev/null 2>&1; then
  emit_error "$EXIT_INTERNAL" "curl not found in PATH."
  exit "$EXIT_INTERNAL"
fi

if http_request GET "/v1/health"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "GET ${BASE_URL}/v1/health failed: ${HTTP_ERROR:-unknown error}" "$hint"
  exit "$exit_code"
fi

HEALTH_STATUS="$HTTP_STATUS"
HEALTH_BODY="$HTTP_BODY"

if [[ ! "$HEALTH_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status health "$HEALTH_STATUS")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "GET ${BASE_URL}/v1/health returned HTTP $HEALTH_STATUS: $(normalize_detail "$HEALTH_BODY")" "$hint"
  exit "$exit_code"
fi

if http_request GET "/api/version"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "GET ${BASE_URL}/api/version failed: ${HTTP_ERROR:-unknown error}" "$hint"
  exit "$exit_code"
fi

VERSION_STATUS="$HTTP_STATUS"
VERSION_BODY="$HTTP_BODY"

if [[ ! "$VERSION_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status version "$VERSION_STATUS")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "GET ${BASE_URL}/api/version returned HTTP $VERSION_STATUS: $(normalize_detail "$VERSION_BODY")" "$hint"
  exit "$exit_code"
fi

if ! VERSION_NAME="$(extract_version_name "$VERSION_BODY")"; then
  exit "$?"
fi

RUNTIMES_JSON="[]"
if [[ "$INCLUDE_RUNTIMES" -eq 1 ]]; then
  if http_request GET "/api/info"; then
    :
  else
    rc=$?
    exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
    hint=""
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      hint="$_TOLLAMA_HOST_MISMATCH_HINT"
    fi
    emit_error "$exit_code" "GET ${BASE_URL}/api/info failed: ${HTTP_ERROR:-unknown error}" "$hint"
    exit "$exit_code"
  fi

  INFO_STATUS="$HTTP_STATUS"
  INFO_BODY="$HTTP_BODY"

  if [[ ! "$INFO_STATUS" =~ ^2 ]]; then
    exit_code="$(classify_http_status info "$INFO_STATUS")"
    hint=""
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      hint="$_TOLLAMA_HOST_MISMATCH_HINT"
    fi
    emit_error "$exit_code" "GET ${BASE_URL}/api/info returned HTTP $INFO_STATUS: $(normalize_detail "$INFO_BODY")" "$hint"
    exit "$exit_code"
  fi

  if ! RUNTIMES_JSON="$(extract_runtimes_json "$INFO_BODY")"; then
    emit_error "$EXIT_INTERNAL" "failed to parse /api/info runtimes payload."
    exit "$EXIT_INTERNAL"
  fi
fi

emit_success_payload "$VERSION_NAME" "$RUNTIMES_JSON"
