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

MODEL=""
INPUT_PATH=""
BASE_URL_ARG=""
TIMEOUT_ARG=""
ALLOW_PULL=0
ACCEPT_LICENSE=0
METRICS_ARG=""
MASE_SEASONALITY_ARG=""

PAYLOAD_RAW_FILE=""
PAYLOAD_JSON_FILE=""
HTTP_STATUS=""
HTTP_BODY=""
HTTP_ERROR=""

usage() {
  cat <<'USAGE' >&2
Usage:
  tollama-forecast.sh --model NAME [--input FILE] [--base-url URL] [--timeout SEC]
    [--metrics CSV] [--mase-seasonality INT] [--pull] [--accept-license]
USAGE
}

cleanup() {
  if [[ -n "$PAYLOAD_RAW_FILE" ]]; then
    rm -f "$PAYLOAD_RAW_FILE"
  fi
  if [[ -n "$PAYLOAD_JSON_FILE" ]]; then
    rm -f "$PAYLOAD_JSON_FILE"
  fi
}

trap cleanup EXIT

is_positive_number() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v v="$value" 'BEGIN { exit !(v > 0) }'
}

is_positive_integer() {
  local value="$1"
  [[ "$value" =~ ^[1-9][0-9]*$ ]]
}

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

validate_metrics_csv() {
  local raw="$1"
  [[ -n "$raw" ]] || return 1

  local IFS=","
  local parts=()
  read -r -a parts <<< "$raw"
  if [[ ${#parts[@]} -eq 0 ]]; then
    return 1
  fi

  local part trimmed
  for part in "${parts[@]}"; do
    trimmed="$(trim_spaces "$part")"
    [[ -n "$trimmed" ]] || return 1
  done
  return 0
}

print_host_mismatch_hint() {
  echo "Hint: exec host and --base-url may be mismatched. 127.0.0.1 inside sandbox/container is not host localhost." >&2
}

normalize_detail() {
  local raw="$1"
  local flattened
  flattened="${raw//$'\n'/ }"
  printf '%s' "${flattened:0:700}"
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

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required for JSON request normalization." >&2
    exit "$EXIT_INTERNAL"
  fi
}

build_payload_json() {
  require_python3

  local raw_file="$1"
  local out_file="$2"
  if ! python3 - "$raw_file" "$out_file" "$MODEL" "$TIMEOUT" "$METRICS_ARG" "$MASE_SEASONALITY_ARG" <<'PY'; then
import json
import sys

raw_path, out_path, model, timeout_s, metrics_csv, mase_seasonality = sys.argv[1:7]
with open(raw_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

if not isinstance(payload, dict):
    raise ValueError("forecast request payload must be a JSON object")

payload["model"] = model
payload["stream"] = False
payload["timeout"] = float(timeout_s)

parameters = payload.get("parameters")
if parameters is None:
    parameters = {}
elif not isinstance(parameters, dict):
    raise ValueError("parameters must be a JSON object when provided")

metrics_payload = parameters.get("metrics")
if metrics_payload is None:
    metrics_payload = {}
elif not isinstance(metrics_payload, dict):
    raise ValueError("parameters.metrics must be a JSON object when provided")

if metrics_csv:
    names: list[str] = []
    for token in metrics_csv.split(","):
        normalized = token.strip()
        if not normalized:
            raise ValueError("metrics CSV contains empty names")
        names.append(normalized)
    metrics_payload["names"] = names

if mase_seasonality:
    metrics_payload["mase_seasonality"] = int(mase_seasonality)
    existing_names = metrics_payload.get("names")
    if not isinstance(existing_names, list) or len(existing_names) == 0:
        metrics_payload["names"] = ["mase"]

if metrics_payload:
    parameters["metrics"] = metrics_payload
if parameters:
    payload["parameters"] = parameters

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, sort_keys=True)
PY
    echo "Error: invalid request JSON. Provide a valid object payload." >&2
    exit "$EXIT_USAGE"
  fi
}

run_tollama_command() {
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
    --metrics)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      METRICS_ARG="$1"
      ;;
    --mase-seasonality)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      MASE_SEASONALITY_ARG="$1"
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

if [[ -n "$METRICS_ARG" ]] && ! validate_metrics_csv "$METRICS_ARG"; then
  echo "Error: --metrics must be comma-separated non-empty names (example: mape,mase)." >&2
  exit "$EXIT_USAGE"
fi

if [[ -n "$MASE_SEASONALITY_ARG" ]] && ! is_positive_integer "$MASE_SEASONALITY_ARG"; then
  echo "Error: --mase-seasonality must be a positive integer." >&2
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
  if run_tollama_command show tollama show "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT" >/dev/null; then
    :
  else
    rc=$?
    if [[ "$rc" -eq "$EXIT_MODEL_MISSING" ]]; then
      if [[ "$ALLOW_PULL" -eq 0 ]]; then
        echo "Error: model '$MODEL' is not installed. Re-run with --pull to allow installation." >&2
        exit "$EXIT_MODEL_MISSING"
      fi

      pull_cmd=(tollama pull "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT" --no-stream)
      if [[ "$ACCEPT_LICENSE" -eq 1 ]]; then
        pull_cmd+=(--accept-license)
      fi
      if run_tollama_command pull "${pull_cmd[@]}" >/dev/null; then
        :
      else
        pull_rc=$?
        exit "$pull_rc"
      fi
    else
      exit "$rc"
    fi
  fi

  if run_tollama_command forecast tollama run "$MODEL" --input "$PAYLOAD_JSON_FILE" --no-stream --timeout "$TIMEOUT" --base-url "$BASE_URL"; then
    :
  else
    rc=$?
    if [[ "$rc" -eq "$EXIT_INTERNAL" ]]; then
      echo "Error: tollama run failed." >&2
    fi
    exit "$rc"
  fi
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: tollama is unavailable and curl is missing; cannot perform HTTP fallback." >&2
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
  if http_request POST "/api/pull" "$pull_payload"; then
    :
  else
    rc=$?
    exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
    echo "Error: POST ${BASE_URL}/api/pull failed: ${HTTP_ERROR:-unknown error}" >&2
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      print_host_mismatch_hint
    fi
    exit "$exit_code"
  fi

  if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
    exit_code="$(classify_http_status pull "$HTTP_STATUS")"
    echo "Error: POST ${BASE_URL}/api/pull returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
    exit "$exit_code"
  fi
elif [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status show "$HTTP_STATUS")"
  echo "Error: POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    print_host_mismatch_hint
  fi
  exit "$exit_code"
fi

if http_request POST "/api/forecast" "$REQUEST_PAYLOAD"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  echo "Error: POST ${BASE_URL}/api/forecast failed: ${HTTP_ERROR:-unknown error}" >&2
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    print_host_mismatch_hint
  fi
  exit "$exit_code"
fi

if [[ "$HTTP_STATUS" =~ ^2 ]]; then
  echo "$HTTP_BODY"
  exit 0
fi

if [[ "$HTTP_STATUS" == "404" ]]; then
  if http_request POST "/v1/forecast" "$REQUEST_PAYLOAD"; then
    :
  else
    rc=$?
    exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
    echo "Error: POST ${BASE_URL}/v1/forecast failed: ${HTTP_ERROR:-unknown error}" >&2
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      print_host_mismatch_hint
    fi
    exit "$exit_code"
  fi

  if [[ "$HTTP_STATUS" =~ ^2 ]]; then
    echo "$HTTP_BODY"
    exit 0
  fi
fi

exit_code="$(classify_http_status forecast "$HTTP_STATUS")"
echo "Error: forecast request failed with HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" >&2
exit "$exit_code"
