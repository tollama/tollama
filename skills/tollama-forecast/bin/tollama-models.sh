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

COMMAND=""
MODEL=""
BASE_URL_ARG=""
TIMEOUT_ARG=""
SECTION="all"
ACCEPT_LICENSE=0
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
  tollama-models.sh pull <model> [--base-url URL] [--timeout SEC] [--accept-license]
  tollama-models.sh rm <model> [--base-url URL] [--timeout SEC]
  tollama-models.sh info [--base-url URL] [--timeout SEC] [--section daemon|models|runners|env|all]
USAGE
}

require_python3() {
  if ! command -v python3 >/dev/null 2>&1; then
    emit_error "$EXIT_INTERNAL" "python3 is required for JSON extraction."
    exit "$EXIT_INTERNAL"
  fi
}

extract_available_models() {
  local payload="$1"
  require_python3

  if ! python3 - "$payload" <<'PY'; then
import json
import sys

try:
    payload = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    print(f"invalid JSON payload: {exc}")
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

extract_info_section() {
  local section="$1"
  local payload="$2"
  require_python3

  if ! python3 - "$section" "$payload" <<'PY'; then
import json
import sys

section = sys.argv[1]
try:
    payload = json.loads(sys.argv[2])
except json.JSONDecodeError as exc:
    print(f"invalid JSON payload: {exc}")
    raise SystemExit(10)

if not isinstance(payload, dict):
    raise SystemExit(10)

if section == "all":
    result = payload
else:
    result = payload.get(section)

if result is None:
    result = {}

json.dump(result, sys.stdout, sort_keys=True)
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

if [[ $# -lt 1 ]]; then
  usage
  exit "$EXIT_USAGE"
fi

COMMAND="$1"
shift

case "$COMMAND" in
  installed|loaded|available|info)
    ;;
  show|pull|rm)
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
    --section)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      SECTION="$1"
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

if [[ "$SECTION" != "all" && "$SECTION" != "daemon" && "$SECTION" != "models" && "$SECTION" != "runners" && "$SECTION" != "env" ]]; then
  emit_error "$EXIT_USAGE" "--section must be one of: daemon, models, runners, env, all"
  exit "$EXIT_USAGE"
fi

if [[ "$COMMAND" != "info" && "$SECTION" != "all" ]]; then
  emit_error "$EXIT_USAGE" "--section is only valid with the info subcommand"
  exit "$EXIT_USAGE"
fi

if [[ "$COMMAND" != "pull" && "$ACCEPT_LICENSE" -eq 1 ]]; then
  emit_error "$EXIT_USAGE" "--accept-license is only valid with the pull subcommand"
  exit "$EXIT_USAGE"
fi

BASE_URL="${BASE_URL_ARG:-${TOLLAMA_BASE_URL:-$DEFAULT_BASE_URL}}"
TIMEOUT="${TIMEOUT_ARG:-${TOLLAMA_FORECAST_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT}}"

if ! is_positive_number "$TIMEOUT"; then
  emit_error "$EXIT_USAGE" "timeout must be a positive number, got '$TIMEOUT'."
  exit "$EXIT_USAGE"
fi

HAS_TOLLAMA=0
if command -v tollama >/dev/null 2>&1; then
  HAS_TOLLAMA=1
fi

case "$COMMAND" in
  installed)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd installed tollama list --json --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        exit "$?"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
    fi

    if http_request GET "/api/tags"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "GET ${BASE_URL}/api/tags failed: ${HTTP_ERROR:-unknown error}" "$hint"
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status installed "$HTTP_STATUS")"
    emit_error "$exit_code" "GET ${BASE_URL}/api/tags returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
    ;;

  loaded)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd loaded tollama ps --json --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        exit "$?"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
    fi

    if http_request GET "/api/ps"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "GET ${BASE_URL}/api/ps failed: ${HTTP_ERROR:-unknown error}" "$hint"
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status loaded "$HTTP_STATUS")"
    emit_error "$exit_code" "GET ${BASE_URL}/api/ps returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
    ;;

  show)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd show tollama show "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        exit "$?"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
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

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status show "$HTTP_STATUS")"
    emit_error "$exit_code" "POST ${BASE_URL}/api/show returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
    ;;

  pull)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      pull_cmd=(tollama pull "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT" --no-stream)
      if [[ "$ACCEPT_LICENSE" -eq 1 ]]; then
        pull_cmd+=(--accept-license)
      fi
      if run_tollama_cmd pull "${pull_cmd[@]}"; then
        :
      else
        exit "$?"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
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

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status pull "$HTTP_STATUS")"
    emit_error "$exit_code" "POST ${BASE_URL}/api/pull returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
    ;;

  rm)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      if run_tollama_cmd rm tollama rm "$MODEL" --base-url "$BASE_URL" --timeout "$TIMEOUT"; then
        :
      else
        exit "$?"
      fi
      exit 0
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
    fi

    if http_request DELETE "/api/delete" "{\"model\":\"$MODEL\"}"; then
      :
    else
      rc=$?
      exit_code="$(classify_curl_error "$rc" "${HTTP_ERROR:-}")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "DELETE ${BASE_URL}/api/delete failed: ${HTTP_ERROR:-unknown error}" "$hint"
      exit "$exit_code"
    fi

    if [[ "$HTTP_STATUS" =~ ^2 ]]; then
      echo "$HTTP_BODY"
      exit 0
    fi

    exit_code="$(classify_http_status rm "$HTTP_STATUS")"
    emit_error "$exit_code" "DELETE ${BASE_URL}/api/delete returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")"
    exit "$exit_code"
    ;;

  available)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      tmp_out="$(mktemp)"
      tmp_err="$(mktemp)"

      if tollama info --json --remote --base-url "$BASE_URL" --timeout "$TIMEOUT" >"$tmp_out" 2>"$tmp_err"; then
        if extract_available_models "$(cat "$tmp_out")"; then
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
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "tollama info --json --remote failed: $err_text" "$hint"
      exit "$exit_code"
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
    fi

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

    if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
      exit_code="$(classify_http_status available "$HTTP_STATUS")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "GET ${BASE_URL}/api/info returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" "$hint"
      exit "$exit_code"
    fi

    if extract_available_models "$HTTP_BODY"; then
      :
    else
      exit "$?"
    fi
    exit 0
    ;;

  info)
    if [[ "$HAS_TOLLAMA" -eq 1 ]]; then
      tmp_out="$(mktemp)"
      tmp_err="$(mktemp)"

      if tollama info --json --remote --base-url "$BASE_URL" --timeout "$TIMEOUT" >"$tmp_out" 2>"$tmp_err"; then
        if extract_info_section "$SECTION" "$(cat "$tmp_out")"; then
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

      exit_code="$(classify_text_error info "$err_text")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "tollama info --json --remote failed: $err_text" "$hint"
      exit "$exit_code"
    fi

    if ! command -v curl >/dev/null 2>&1; then
      emit_error "$EXIT_INTERNAL" "neither tollama nor curl is available in PATH."
      exit "$EXIT_INTERNAL"
    fi

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

    if [[ ! "$HTTP_STATUS" =~ ^2 ]]; then
      exit_code="$(classify_http_status info "$HTTP_STATUS")"
      hint=""
      if [[ "$exit_code" == "$EXIT_DAEMON_UNREACHABLE" ]]; then
        hint="$_TOLLAMA_HOST_MISMATCH_HINT"
      fi
      emit_error "$exit_code" "GET ${BASE_URL}/api/info returned HTTP $HTTP_STATUS: $(normalize_detail "$HTTP_BODY")" "$hint"
      exit "$exit_code"
    fi

    if extract_info_section "$SECTION" "$HTTP_BODY"; then
      :
    else
      exit "$?"
    fi
    exit 0
    ;;
esac
