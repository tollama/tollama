#!/usr/bin/env bash

# Shared helpers for tollama OpenClaw skill scripts.

: "${EXIT_USAGE:=2}"
: "${EXIT_DAEMON_UNREACHABLE:=3}"
: "${EXIT_MODEL_MISSING:=4}"
: "${EXIT_PERMISSION:=5}"
: "${EXIT_TIMEOUT:=6}"
: "${EXIT_INTERNAL:=10}"

_TOLLAMA_HOST_MISMATCH_HINT="Hint: exec host and --base-url may be mismatched. 127.0.0.1 inside sandbox/container is not host localhost."

is_positive_number() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
  awk -v v="$value" 'BEGIN { exit !(v > 0) }'
}

normalize_detail() {
  local raw="$1"
  local flattened
  flattened="${raw//$'\n'/ }"
  printf '%s' "${flattened:0:700}"
}

print_host_mismatch_hint() {
  if [[ "${TOLLAMA_JSON_STDERR:-0}" == "1" ]]; then
    return
  fi
  echo "$_TOLLAMA_HOST_MISMATCH_HINT" >&2
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

_error_code_for_exit() {
  local exit_code="$1"
  case "$exit_code" in
    "$EXIT_USAGE") printf '%s' "INVALID_REQUEST" ;;
    "$EXIT_DAEMON_UNREACHABLE") printf '%s' "DAEMON_UNREACHABLE" ;;
    "$EXIT_MODEL_MISSING") printf '%s' "MODEL_MISSING" ;;
    "$EXIT_PERMISSION") printf '%s' "PERMISSION_DENIED" ;;
    "$EXIT_TIMEOUT") printf '%s' "TIMEOUT" ;;
    "$EXIT_INTERNAL") printf '%s' "INTERNAL_ERROR" ;;
    *) printf '%s' "INTERNAL_ERROR" ;;
  esac
}

_json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  value="${value//$'\t'/\\t}"
  printf '%s' "$value"
}

emit_error() {
  local exit_code="$1"
  local message="$2"
  local hint="${3:-}"
  local subcode="${4:-}"
  local code

  code="$(_error_code_for_exit "$exit_code")"
  if [[ "$exit_code" == "$EXIT_PERMISSION" && -z "$subcode" ]]; then
    if is_license_error "$message"; then
      subcode="LICENSE_REQUIRED"
    else
      subcode="PERMISSION_DENIED"
    fi
  fi

  if [[ "${TOLLAMA_JSON_STDERR:-0}" == "1" ]]; then
    local code_json subcode_json message_json hint_json
    code_json="$(_json_escape "$code")"
    subcode_json="$(_json_escape "$subcode")"
    message_json="$(_json_escape "$message")"
    hint_json="$(_json_escape "$hint")"

    if [[ -n "$subcode" && -n "$hint" ]]; then
      printf '{"error":{"code":"%s","subcode":"%s","exit_code":%s,"message":"%s","hint":"%s"}}\n' \
        "$code_json" "$subcode_json" "$exit_code" "$message_json" "$hint_json" >&2
    elif [[ -n "$subcode" ]]; then
      printf '{"error":{"code":"%s","subcode":"%s","exit_code":%s,"message":"%s"}}\n' \
        "$code_json" "$subcode_json" "$exit_code" "$message_json" >&2
    elif [[ -n "$hint" ]]; then
      printf '{"error":{"code":"%s","exit_code":%s,"message":"%s","hint":"%s"}}\n' \
        "$code_json" "$exit_code" "$message_json" "$hint_json" >&2
    else
      printf '{"error":{"code":"%s","exit_code":%s,"message":"%s"}}\n' \
        "$code_json" "$exit_code" "$message_json" >&2
    fi
    return
  fi

  echo "Error: $message" >&2
  if [[ -n "$hint" ]]; then
    echo "$hint" >&2
  fi
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

  case "$context" in
    health|version|info)
      case "$status" in
        401|403) echo "$EXIT_PERMISSION" ;;
        408|504) echo "$EXIT_TIMEOUT" ;;
        *) echo "$EXIT_DAEMON_UNREACHABLE" ;;
      esac
      return
      ;;
  esac

  case "$status" in
    400)
      echo "$EXIT_USAGE"
      ;;
    401|403)
      echo "$EXIT_PERMISSION"
      ;;
    404)
      if [[ "$context" == "show" ]]; then
        echo "$EXIT_MODEL_MISSING"
      else
        echo "$EXIT_INTERNAL"
      fi
      ;;
    408|504)
      echo "$EXIT_TIMEOUT"
      ;;
    409)
      echo "$EXIT_PERMISSION"
      ;;
    *)
      echo "$EXIT_INTERNAL"
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

  local status
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
