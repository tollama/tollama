#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://localhost:11435"
DEFAULT_TIMEOUT="300"

EXIT_USAGE=2
EXIT_DEP_MISSING=3
EXIT_MODEL_MISSING=4
EXIT_DAEMON_UNREACHABLE=5
EXIT_FORECAST_FAILED=6

MODEL=""
INPUT_PATH=""
BASE_URL_ARG=""
TIMEOUT_ARG=""
ALLOW_PULL=0
ACCEPT_LICENSE=0

PAYLOAD_RAW_FILE=""
PAYLOAD_JSON_FILE=""
HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

usage() {
  cat <<'USAGE' >&2
Usage:
  tollama-forecast.sh --model NAME [--input FILE] [--base-url URL] [--timeout SEC] [--pull] [--accept-license]
USAGE
}

cleanup() {
  [[ -n "$PAYLOAD_RAW_FILE" ]] && rm -f "$PAYLOAD_RAW_FILE"
  [[ -n "$PAYLOAD_JSON_FILE" ]] && rm -f "$PAYLOAD_JSON_FILE"
}

trap cleanup EXIT

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
  printf '%s' "${flattened:0:700}"
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

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required for JSON request normalization." >&2
    exit "$EXIT_DEP_MISSING"
  fi
}

build_payload_json() {
  require_python3

  local raw_file="$1"
  local out_file="$2"
  if ! python3 - "$raw_file" "$out_file" "$MODEL" "$TIMEOUT" <<'PY'; then
import json
import sys

raw_path, out_path, model, timeout_s = sys.argv[1:5]
with open(raw_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if not isinstance(payload, dict):
    raise ValueError("forecast request payload must be a JSON object")

payload["model"] = model
payload["stream"] = False
payload["timeout"] = float(timeout_s)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True)
PY
    echo "Error: invalid request JSON. Provide a valid object payload." >&2
    exit "$EXIT_USAGE"
  fi
}

while (($# > 0)); do
  case "$1" in
    --model)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      MODEL="$1"
      ;;
    --input)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      INPUT_PATH="$1"
      ;;
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
    --pull)
      ALLOW_PULL=1
      ;;
    --accept-license)
      ACCEPT_LICENSE=1
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

if [[ -z "$MODEL" ]]; then
  usage
  exit "$EXIT_USAGE"
fi

BASE_URL="${BASE_URL_ARG:-${TOLLAMA_BASE_URL:-$DEFAULT_BASE_URL}}"
TIMEOUT="${TIMEOUT_ARG:-${TOLLAMA_FORECAST_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT}}"

if ! is_positive_number "$TIMEOUT"; then
  echo "Error: timeout must be a positive number, got '$TIMEOUT'." >&2
  exit "$EXIT_USAGE"
fi

PAYLOAD_RAW_FILE="$(mktemp)"
PAYLOAD_JSON_FILE="$(mktemp)"

if [[ -n "$INPUT_PATH" ]]; then
  if [[ ! -f "$INPUT_PATH" ]]; then
    echo "Error: input file not found: $INPUT_PATH" >&2
    exit "$EXIT_USAGE"
  fi
  cat "$INPUT_PATH" >"$PAYLOAD_RAW_FILE"
else
  if [[ -t 0 ]]; then
    echo "Error: missing request JSON. Provide --input FILE or pipe JSON via stdin." >&2
    exit "$EXIT_USAGE"
  fi
  cat >"$PAYLOAD_RAW_FILE"
fi

build_payload_json "$PAYLOAD_RAW_FILE" "$PAYLOAD_JSON_FILE"
REQUEST_PAYLOAD="$(cat "$PAYLOAD_JSON_FILE")"

if command -v tollama >/dev/null 2>&1; then
  show_out="$(mktemp)"
  show_err="$(mktemp)"
  if ! tollama show "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT" >"$show_out" 2>"$show_err"; then
    show_text="$(cat "$show_err")"
    rm -f "$show_out" "$show_err"

    if [[ "$show_text" == *"HTTP 404"* ]]; then
      if [[ "$ALLOW_PULL" -eq 0 ]]; then
        echo "Error: model '$MODEL' is not installed. Re-run with --pull to allow installation." >&2
        exit "$EXIT_MODEL_MISSING"
      fi

      pull_cmd=(tollama pull "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT" --no-stream)
      if [[ "$ACCEPT_LICENSE" -eq 1 ]]; then
        pull_cmd+=(--accept-license)
      fi
      pull_out="$(mktemp)"
      pull_err="$(mktemp)"
      if ! "${pull_cmd[@]}" >"$pull_out" 2>"$pull_err"; then
        pull_text="$(cat "$pull_err")"
        rm -f "$pull_out" "$pull_err"
        echo "Error: pull failed for model '$MODEL': $pull_text" >&2
        if is_daemon_error "$pull_text"; then
          print_host_mismatch_hint
          exit "$EXIT_DAEMON_UNREACHABLE"
        fi
        exit "$EXIT_FORECAST_FAILED"
      fi
      rm -f "$pull_out" "$pull_err"
    else
      echo "Error: tollama show failed: $show_text" >&2
      if is_daemon_error "$show_text"; then
        print_host_mismatch_hint
        exit "$EXIT_DAEMON_UNREACHABLE"
      fi
      exit "$EXIT_FORECAST_FAILED"
    fi
  else
    rm -f "$show_out" "$show_err"
  fi

  run_out="$(mktemp)"
  run_err="$(mktemp)"
  if tollama run "$MODEL" --input "$PAYLOAD_JSON_FILE" --no-stream --timeout "$TIMEOUT" --base-url "$BASE_URL" >"$run_out" 2>"$run_err"; then
    cat "$run_out"
    rm -f "$run_out" "$run_err"
    exit 0
  fi

  run_text="$(cat "$run_err")"
  rm -f "$run_out" "$run_err"
  echo "$run_text" >&2
  if is_daemon_error "$run_text"; then
    print_host_mismatch_hint
    exit "$EXIT_DAEMON_UNREACHABLE"
  fi
  echo "Error: tollama run failed." >&2
  exit "$EXIT_FORECAST_FAILED"
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: tollama is unavailable and curl is missing; cannot perform HTTP fallback." >&2
  exit "$EXIT_DEP_MISSING"
fi

if ! http_request POST "/api/show" "{\"model\":\"$MODEL\"}"; then
  echo "Error: POST ${BASE_URL}/api/show failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

if [[ "$HTTP_STATUS" == "404" ]]; then
  if [[ "$ALLOW_PULL" -eq 0 ]]; then
    echo "Error: model '$MODEL' is not installed. Re-run with --pull to allow installation." >&2
    exit "$EXIT_MODEL_MISSING"
  fi

  accept_license_json="false"
  if [[ "$ACCEPT_LICENSE" -eq 1 ]]; then
    accept_license_json="true"
  fi

  pull_payload="{\"model\":\"$MODEL\",\"stream\":false,\"accept_license\":$accept_license_json}"
  if ! http_request POST "/api/pull" "$pull_payload"; then
    echo "Error: POST ${BASE_URL}/api/pull failed: ${HTTP_ERROR:-unknown error}" >&2
    print_host_mismatch_hint
    exit "$EXIT_DAEMON_UNREACHABLE"
  fi

  if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
    echo "Error: POST ${BASE_URL}/api/pull returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$EXIT_FORECAST_FAILED"
  fi
elif [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
  echo "Error: POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

if ! http_request POST "/api/forecast" "$REQUEST_PAYLOAD"; then
  echo "Error: POST ${BASE_URL}/api/forecast failed: ${HTTP_ERROR:-unknown error}" >&2
  print_host_mismatch_hint
  exit "$EXIT_DAEMON_UNREACHABLE"
fi

if [[ "$HTTP_STATUS" =~ ^2 ]]; then
  echo "$HTTP_BODY"
  exit 0
fi

if [[ "$HTTP_STATUS" == "404" ]]; then
  if ! http_request POST "/v1/forecast" "$REQUEST_PAYLOAD"; then
    echo "Error: POST ${BASE_URL}/v1/forecast failed: ${HTTP_ERROR:-unknown error}" >&2
    print_host_mismatch_hint
    exit "$EXIT_DAEMON_UNREACHABLE"
  fi

  if [[ "$HTTP_STATUS" =~ ^2 ]]; then
    echo "$HTTP_BODY"
    exit 0
  fi
fi

echo "Error: forecast request failed with HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
exit "$EXIT_FORECAST_FAILED"
