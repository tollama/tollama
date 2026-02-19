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

SKILL_BIN_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
# shellcheck source=skills/tollama-forecast/bin/_tollama_lib.sh
source "$SKILL_BIN_DIR/_tollama_lib.sh"

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

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    emit_error "$EXIT_INTERNAL" "python3 is required for JSON request normalization."
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
    emit_error "$EXIT_USAGE" "invalid request JSON. Provide a valid object payload."
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

  local err_text exit_code hint
  err_text="$(cat "$err_file")"
  rm -f "$out_file" "$err_file"

  exit_code="$(classify_text_error "$context" "$err_text")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi

  emit_error "$exit_code" "$err_text" "$hint"
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
  emit_error "$EXIT_USAGE" "timeout must be a positive number, got '$TIMEOUT'."
  exit "$EXIT_USAGE"
fi

if [[ -n "$METRICS_ARG" ]] && ! validate_metrics_csv "$METRICS_ARG"; then
  emit_error "$EXIT_USAGE" "--metrics must be comma-separated non-empty names (example: mape,mase)."
  exit "$EXIT_USAGE"
fi

if [[ -n "$MASE_SEASONALITY_ARG" ]] && ! is_positive_integer "$MASE_SEASONALITY_ARG"; then
  emit_error "$EXIT_USAGE" "--mase-seasonality must be a positive integer."
  exit "$EXIT_USAGE"
fi

PAYLOAD_RAW_FILE="$(mktemp)"
PAYLOAD_JSON_FILE="$(mktemp)"

if [[ -n "$INPUT_PATH" ]]; then
  if [[ ! -f "$INPUT_PATH" ]]; then
    emit_error "$EXIT_USAGE" "input file not found: $INPUT_PATH"
    exit "$EXIT_USAGE"
  fi
  cat "$INPUT_PATH" >"$PAYLOAD_RAW_FILE"
else
  if [[ -t 0 ]]; then
    emit_error "$EXIT_USAGE" "missing request JSON. Provide --input FILE or pipe JSON via stdin."
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
        emit_error "$EXIT_MODEL_MISSING" "model '$MODEL' is not installed" "Re-run with --pull to allow installation"
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
    if [[ "$rc" -eq "$EXIT_INTERNAL" ]] && [[ "${TOLLAMA_JSON_STDERR:-0}" != "1" ]]; then
      emit_error "$EXIT_INTERNAL" "tollama run failed."
    fi
    exit "$rc"
  fi
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  emit_error "$EXIT_INTERNAL" "tollama is unavailable and curl is missing; cannot perform HTTP fallback."
  exit "$EXIT_INTERNAL"
fi

if http_request POST "/api/show" "{\"model\":\"$MODEL\"}"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "POST ${BASE_URL}/api/show failed: ${HTTP_ERROR:-unknown error}" "$hint"
  exit "$exit_code"
fi

if [[ "$HTTP_STATUS" == "404" ]]; then
  if [[ "$ALLOW_PULL" -eq 0 ]]; then
    emit_error "$EXIT_MODEL_MISSING" "model '$MODEL' is not installed" "Re-run with --pull to allow installation"
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
    hint=""
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      hint="$_TOLLAMA_HOST_MISMATCH_HINT"
    fi
    emit_error "$exit_code" "POST ${BASE_URL}/api/pull failed: ${HTTP_ERROR:-unknown error}" "$hint"
    exit "$exit_code"
  fi

  if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
    exit_code="$(classify_http_status pull "$HTTP_STATUS")"
    emit_error "$exit_code" "POST ${BASE_URL}/api/pull returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
  fi
elif [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
  exit_code="$(classify_http_status show "$HTTP_STATUS")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" "$hint"
  exit "$exit_code"
fi

if http_request POST "/api/forecast" "$REQUEST_PAYLOAD"; then
  :
else
  rc=$?
  exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
  hint=""
  if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
    hint="$_TOLLAMA_HOST_MISMATCH_HINT"
  fi
  emit_error "$exit_code" "POST ${BASE_URL}/api/forecast failed: ${HTTP_ERROR:-unknown error}" "$hint"
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
    hint=""
    if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
      hint="$_TOLLAMA_HOST_MISMATCH_HINT"
    fi
    emit_error "$exit_code" "POST ${BASE_URL}/v1/forecast failed: ${HTTP_ERROR:-unknown error}" "$hint"
    exit "$exit_code"
  fi

  if [[ "$HTTP_STATUS" =~ ^2 ]]; then
    echo "$HTTP_BODY"
    exit 0
  fi
fi

exit_code="$(classify_http_status forecast "$HTTP_STATUS")"
emit_error "$exit_code" "forecast request failed with HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
exit "$exit_code"
