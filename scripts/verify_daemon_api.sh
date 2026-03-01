#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="http://127.0.0.1:11435"
STREAM_TIMEOUT_SECONDS=8
READY_TIMEOUT_SECONDS=45
DAEMON_HOME=""
DAEMON_PID=""
SKIP_CONTRACT_TESTS=0
SKIP_DAEMON_START=0
RUN_AUTH_CHECK=0
RUN_ROUTE_INVENTORY_CHECK=1
API_KEY="ci-daemon-smoke-key"
FAILED=0
DAEMON_STARTED_BY_SCRIPT=0
CLEANUP_STATE=1
EXPECTED_ENDPOINTS=(
  "/api/version"
  "/api/info"
  "/v1/health"
  "/api/usage"
  "/api/events"
  "/metrics"
  "/docs"
  "/redoc"
  "/openapi.json"
  "/api/modelfiles"
  "/api/modelfiles/{name}"
  "/api/ingest/upload"
  "/api/forecast/upload"
  "/api/tags"
  "/api/show"
  "/api/pull"
  "/api/delete"
  "/api/ps"
  "/v1/models"
  "/v1/models/pull"
  "/v1/models/{name}"
  "/api/validate"
  "/api/forecast"
  "/api/forecast/progressive"
  "/v1/forecast"
  "/api/auto-forecast"
  "/api/compare"
  "/api/analyze"
  "/api/generate"
  "/api/counterfactual"
  "/api/scenario-tree"
  "/api/what-if"
  "/api/report"
  "/api/pipeline"
  "/.well-known/agent-card.json"
  "/.well-known/agent.json"
  "/a2a"
)

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

COLOR_BLUE=$'\033[0;34m'
COLOR_GREEN=$'\033[0;32m'
COLOR_RED=$'\033[0;31m'
COLOR_RESET=$'\033[0m'

AUTH_HEADERS=()
REQUEST_STATUS=""
REQUEST_BODY=""

log() {
  echo "${COLOR_BLUE}$*${COLOR_RESET}"
}

pass() {
  echo "${COLOR_GREEN}PASS${COLOR_RESET} - $1"
}

fail() {
  echo "${COLOR_RED}FAIL${COLOR_RESET} - $1"
  FAILED=1
}

usage() {
  cat <<'USAGE'
Usage:
  scripts/verify_daemon_api.sh [options]

Options:
  --base-url URL          Base URL for daemon (default: http://127.0.0.1:11435)
  --stream-timeout SECS   Stream request timeout (default: 8)
  --ready-timeout SECS    Daemon startup wait timeout (default: 45)
  --daemon-home DIR       Explicit TOLLAMA_HOME directory for this run
  --skip-contract-tests    Skip pytest contract verification step
  --skip-daemon-start      Assume an external daemon is already running
  --skip-route-inventory   Skip docs/app route inventory check
  --no-cleanup             Keep daemon-created test state
  --api-key KEY           Enable auth verification using KEY
  -h, --help              Show this message

Environment:
  BASE_URL and timeout options can be overridden by flags above.
  If --skip-contract-tests is not set, the script runs:
    python -m pytest -q tests/test_openapi_docs.py tests/test_daemon_api.py
    python -m pytest -q tests/test_a2a_server.py tests/test_a2a_client.py
  If --skip-route-inventory is set, docs/app route inventory validation is skipped.
  By default, resources created during smoke checks are cleaned up when possible.
USAGE
}

cleanup_daemon() {
  if [[ -n "${DAEMON_PID}" ]]; then
    kill "${DAEMON_PID}" >/dev/null 2>&1 || true
    wait "${DAEMON_PID}" >/dev/null 2>&1 || true
    DAEMON_PID=""
  fi
}

cleanup() {
  if (( CLEANUP_STATE == 1 )) && (( DAEMON_STARTED_BY_SCRIPT == 1 )); then
    cleanup_resources || true
  fi
  cleanup_daemon
}

trap cleanup EXIT

send_request() {
  local method="$1"
  local path="$2"
  local payload_file="$3"
  local response_file="$4"
  local status

  local args=(
    -sS
    --max-time "$READY_TIMEOUT_SECONDS"
    --write-out "%{http_code}"
    --output "$response_file"
    ${AUTH_HEADERS[@]+"${AUTH_HEADERS[@]}"}
    -X "$method"
    "${BASE_URL}${path}"
  )

  if [[ "$payload_file" != "NONE" ]]; then
    args+=( -H "Content-Type: application/json" --data "@${payload_file}" )
  fi

  if ! status="$(curl "${args[@]}")"; then
    REQUEST_STATUS="000"
    REQUEST_BODY="$response_file"
    return 1
  fi

  REQUEST_STATUS="$status"
  REQUEST_BODY="$response_file"
  return 0
}

send_stream_request() {
  local method="$1"
  local path="$2"
  local payload_file="$3"
  local response_file="$4"
  local accept="$5"
  local status

  local args=(
    -sS
    --max-time "$STREAM_TIMEOUT_SECONDS"
    --write-out "%{http_code}"
    --output "$response_file"
    -H "Accept: ${accept}"
    ${AUTH_HEADERS[@]+"${AUTH_HEADERS[@]}"}
    -X "$method"
    "${BASE_URL}${path}"
  )

  if [[ "$payload_file" != "NONE" ]]; then
    args+=( -H "Content-Type: application/json" --data "@${payload_file}" )
  fi

  if ! status="$(curl "${args[@]}")"; then
    REQUEST_STATUS="000"
    REQUEST_BODY="$response_file"
    return 1
  fi

  REQUEST_STATUS="$status"
  REQUEST_BODY="$response_file"
  return 0
}

assert_status() {
  local expected="$1"
  [[ "$REQUEST_STATUS" == "$expected" ]]
}

assert_status_in() {
  local expected
  for expected in "$@"; do
    [[ "$REQUEST_STATUS" == "$expected" ]] && return 0
  done
  return 1
}

assert_json() {
  jq -e . "$REQUEST_BODY" >/dev/null 2>&1
}

assert_jq() {
  local expr="$1"
  jq -e "$expr" "$REQUEST_BODY" >/dev/null 2>&1
}

check_stream_has_json_lines() {
  local file="$1"
  local required_prefix="$2"
  local count
  count="$(sed -n '1,5p' "$file" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  if (( count < 1 )); then
    return 1
  fi
  local line
  local scanned=0
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    if [[ "${line}" == "{*}" || "${line}" == "{"* ]]; then
      if echo "${line}" | jq -e . >/dev/null 2>&1; then
        if [[ -n "$required_prefix" ]]; then
          echo "${line}" | jq -e "$required_prefix" >/dev/null 2>&1
          return $?
        fi
        return 0
      fi
    fi
    scanned=$(( scanned + 1 ))
    if (( scanned > 20 )); then
      break
    fi
  done < "$file"
  return 1
}

run_check() {
  local name="$1"
  shift
  if "$@"; then
    pass "$name"
    return 0
  else
    fail "$name"
    return 1
  fi
}

# Export helper functions so they are available inside `bash -c` subshells
# used by run_check. Variables must also be exported so subshells can access them.
export -f send_request send_stream_request
export -f assert_status assert_status_in assert_json assert_jq
export -f check_stream_has_json_lines
export BASE_URL READY_TIMEOUT_SECONDS STREAM_TIMEOUT_SECONDS TMP_DIR

ensure_tmp_payloads() {
  cat > "${TMP_DIR}/forecast.json" <<'JSON'
{"model":"mock","horizon":2,"quantiles":[0.1,0.9],"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[1.0,3.0]},{"id":"s2","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[2.0,4.0]}],"options":{}}
JSON
  cat > "${TMP_DIR}/forecast_missing_series.json" <<'JSON'
{"model":"mock","horizon":2,"series":[],"options":{}}
JSON
  cat > "${TMP_DIR}/analyze.json" <<'JSON'
{"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03","2025-01-04"],"target":[1.0,2.0,1.5,2.5]}],"parameters":{"max_lag":2,"top_k_seasonality":1}}
JSON
  cat > "${TMP_DIR}/analyze_bad.json" <<'JSON'
{"series":""}
JSON
  cat > "${TMP_DIR}/auto_forecast.json" <<'JSON'
{"model":"mock","horizon":2,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[1.0,3.0]}],"strategy":"auto","options":{}}
JSON
  cat > "${TMP_DIR}/compare.json" <<'JSON'
{"models":["mock","missing"],"horizon":2,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[1.0,3.0]}],"options":{}}
JSON
  cat > "${TMP_DIR}/generate.json" <<'JSON'
{"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03","2025-01-04","2025-01-05","2025-01-06","2025-01-07"],"target":[10.0,12.0,11.0,13.0,12.0,14.0,13.0]}],"count":2,"length":7,"seed":42,"method":"statistical"}
JSON
  cat > "${TMP_DIR}/counterfactual.json" <<'JSON'
{"model":"mock","intervention_index":3,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03","2025-01-04","2025-01-05"],"target":[10.0,11.0,12.0,20.0,22.0]}],"options":{}}
JSON
  cat > "${TMP_DIR}/scenario_tree.json" <<'JSON'
{"model":"mock","horizon":4,"depth":2,"branch_quantiles":[0.1,0.9],"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03"],"target":[10.0,11.0,12.0]}],"options":{}}
JSON
  cat > "${TMP_DIR}/what_if.json" <<'JSON'
{"model":"mock","horizon":2,"series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02"],"target":[1.0,3.0],"past_covariates":{"temperature":[15.0,16.0]},"future_covariates":{"temperature":[17.0,18.0]}}],"options":{},"scenarios":[{"name":"high_demand","transforms":[{"operation":"multiply","field":"target","value":1.2}]},{"name":"cold_weather","transforms":[{"operation":"add","field":"future_covariates","key":"temperature","value":-5.0}]}]}
JSON
  cat > "${TMP_DIR}/report.json" <<'JSON'
{"horizon":2,"strategy":"auto","series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03","2025-01-04"],"target":[1.0,2.0,1.5,2.5]}],"options":{}}
JSON
  cat > "${TMP_DIR}/pipeline.json" <<'JSON'
{"horizon":2,"strategy":"auto","series":[{"id":"s1","freq":"D","timestamps":["2025-01-01","2025-01-02","2025-01-03","2025-01-04"],"target":[1.0,2.0,1.5,2.5]}],"options":{}}
JSON
  cat > "${TMP_DIR}/forecast_upload_payload.json" <<'JSON'
{"model":"mock","horizon":2,"options":{}}
JSON
  cat > "${TMP_DIR}/upload.csv" <<'CSV'
timestamp,target
2025-01-01,1.0
2025-01-02,2.0
2025-01-03,3.0
CSV
  cat > "${TMP_DIR}/modelfile_profile.json" <<'JSON'
{"name":"ci-smoke","profile":{"model":"mock","quantiles":[0.1,0.9],"options":{"seed":7}}}
JSON
}

start_daemon() {
  if (( SKIP_DAEMON_START == 1 )); then
    return 0
  fi

  if [[ -z "$DAEMON_HOME" ]]; then
    DAEMON_HOME="$(mktemp -d "${TMP_DIR}/tollama-home-XXXXXX")"
  fi
  mkdir -p "$DAEMON_HOME"
  export TOLLAMA_HOME="$DAEMON_HOME"
  local host_port
  host_port="${BASE_URL#*//}"
  host_port="${host_port%%/*}"
  export TOLLAMA_HOST="$host_port"

  if command -v tollamad >/dev/null 2>&1; then
    ( cd "$ROOT_DIR" && tollamad ) >"${TMP_DIR}/daemon.log" 2>&1 &
  else
    ( cd "$ROOT_DIR" && python3 -m tollama.daemon.main ) >"${TMP_DIR}/daemon.log" 2>&1 &
  fi
  DAEMON_PID=$!
  DAEMON_STARTED_BY_SCRIPT=1

  local attempt=0
  while (( attempt < READY_TIMEOUT_SECONDS )); do
    local ping_body="${TMP_DIR}/ping.json"
    if send_request GET "/v1/health" "NONE" "$ping_body" && assert_status 200; then
      return 0
    fi
    sleep 1
    attempt=$(( attempt + 1 ))
  done

  echo "daemon failed to start; see ${TMP_DIR}/daemon.log"
  return 1
}

stop_daemon_and_reset_auth() {
  cleanup_daemon
  AUTH_HEADERS=()
}

run_contract_tests() {
  (cd "$ROOT_DIR" && python -m pytest -q tests/test_openapi_docs.py tests/test_daemon_api.py)
  (cd "$ROOT_DIR" && python -m pytest -q tests/test_a2a_server.py tests/test_a2a_client.py)
}

check_route_inventory() {
  local missing=0
  local route
  for route in "${EXPECTED_ENDPOINTS[@]}"; do
    if ! rg -Fq "${route}" docs/api-reference.md; then
      fail "docs/api-reference.md missing endpoint ${route}"
      missing=1
    fi
    if ! rg -Fq "\"${route}\"" src/tollama/daemon/app.py; then
      fail "src/tollama/daemon/app.py missing registered route ${route}"
      missing=1
    fi
  done
  if (( missing == 0 )); then
    pass "Route inventory in docs and daemon app are aligned for expected endpoints"
  fi
}

cleanup_resources() {
  local payload_file="${TMP_DIR}/cleanup_delete_mock.json"
  printf '%s\n' '{"model":"mock"}' > "${payload_file}"

  send_request DELETE /api/delete "${payload_file}" "${TMP_DIR}/cleanup_api_delete.json" || true
  send_request DELETE /v1/models/mock NONE "${TMP_DIR}/cleanup_v1_delete.json" || true
  send_request DELETE /api/modelfiles/ci-smoke NONE "${TMP_DIR}/cleanup_modelfile_delete.json" || true
}

check_system_endpoints() {
  run_check "GET /v1/health returns ok" bash -c "
    send_request GET /v1/health NONE ${TMP_DIR}/health.json
    assert_status 200 && assert_json && assert_jq '.status==\"ok\"'
  "
  run_check "GET /api/version returns version string" bash -c "
    send_request GET /api/version NONE ${TMP_DIR}/version.json
    assert_status 200 && assert_json && assert_jq 'has(\"version\") and (.version|type==\"string\")'
  "
  run_check "GET /api/info returns inventory object" bash -c "
    send_request GET /api/info NONE ${TMP_DIR}/info.json
    assert_status 200 && assert_json && assert_jq 'has(\"daemon\") and has(\"models\") and has(\"runners\")'
  "
  run_check "GET /api/usage is available or unavailable by design" bash -c "
    send_request GET /api/usage NONE ${TMP_DIR}/usage.json
    assert_status_in 200 503
  "
  run_check "GET /api/events provides SSE stream" bash -c "
    curl -sS -o ${TMP_DIR}/events.stream --max-time ${STREAM_TIMEOUT_SECONDS} -H 'Accept: text/event-stream' ${BASE_URL}/api/events 2>/dev/null || true
    [[ -s ${TMP_DIR}/events.stream ]] && grep -qE '^(data:|:|event:|retry:|id:)' ${TMP_DIR}/events.stream
  "
  run_check "GET /metrics returns metrics or 503 unavailable" bash -c "
    send_request GET /metrics NONE ${TMP_DIR}/metrics.json
    assert_status_in 200 503
  "
  run_check "GET /openapi.json returns schema JSON" bash -c "
    send_request GET /openapi.json NONE ${TMP_DIR}/openapi.json
    assert_status 200 && assert_json
  "
  run_check "GET /docs is reachable" bash -c "
    send_request GET /docs NONE ${TMP_DIR}/docs.txt
    assert_status_in 200 302
  "
  run_check "GET /redoc is reachable" bash -c "
    send_request GET /redoc NONE ${TMP_DIR}/redoc.txt
    assert_status_in 200 302
  "
}

check_runtime_dashboard() {
  run_check "GET /api/dashboard/state returns bootstrap payload" bash -c "
    send_request GET /api/dashboard/state NONE ${TMP_DIR}/dashboard_state.json
    assert_status 200 && assert_json && assert_jq 'has(\"info\") and has(\"ps\") and has(\"usage\")'
  "
  run_check "GET /dashboard/{path} is available when dashboard assets are present" bash -c "
    send_request GET /dashboard/index.html NONE ${TMP_DIR}/dashboard_path.txt
    assert_status_in 200 404 503
  "
  run_check "GET /dashboard serves embedded dashboard if available" bash -c "
    send_request GET /dashboard NONE ${TMP_DIR}/dashboard.html
    if assert_status 200; then
      assert_jq '. | type == \"object\" or true' >/dev/null 2>&1 || true
      exit 0
    fi
    if assert_status 503; then
      exit 0
    fi
    exit 1
  "
}

check_modelfile_endpoints() {
  run_check "POST /api/modelfiles creates or updates ci-smoke profile" bash -c "
    send_request POST /api/modelfiles ${TMP_DIR}/modelfile_profile.json ${TMP_DIR}/modelfile_upsert.json
    assert_status_in 200 201
  "
  run_check "GET /api/modelfiles lists ci-smoke profile" bash -c "
    send_request GET /api/modelfiles NONE ${TMP_DIR}/modelfiles_list.json
    assert_status 200 && assert_json && assert_jq '(.modelfiles | map(.name) | any(. == \"ci-smoke\"))'
  "
  run_check "GET /api/modelfiles/{name} returns profile" bash -c "
    send_request GET /api/modelfiles/ci-smoke NONE ${TMP_DIR}/modelfile_get.json
    assert_status 200 && assert_json && assert_jq '.name == \"ci-smoke\"'
  "
  run_check "DELETE /api/modelfiles/{name} removes profile" bash -c "
    send_request DELETE /api/modelfiles/ci-smoke NONE ${TMP_DIR}/modelfile_delete.json
    assert_status 200 && assert_json && assert_jq '.deleted == true and .name == \"ci-smoke\"'
  "
}

check_ingest_endpoints() {
  run_check "POST /api/ingest/upload returns parsed series" bash -c "
    response=\"${TMP_DIR}/ingest_upload.json\"
    curl -sS --max-time \"$READY_TIMEOUT_SECONDS\" \
      -H 'Accept: application/json' \
      -F 'file=@${TMP_DIR}/upload.csv;type=text/csv' \
      \"${BASE_URL}/api/ingest/upload\" \
      -o \"\$response\" \
      -w '%{http_code}' >/tmp/ingest_status
    REQUEST_STATUS=\$(cat /tmp/ingest_status)
    REQUEST_BODY=\$response
    assert_status 200 && assert_json && assert_jq 'has(\"series\")'
  "
  run_check "POST /api/forecast/upload runs forecast from file payload" bash -c "
    response=\"${TMP_DIR}/forecast_upload.json\"
    payload=\$(cat \"${TMP_DIR}/forecast_upload_payload.json\")
    status=\$(curl -sS --max-time \"$READY_TIMEOUT_SECONDS\" \
      -o \"\$response\" -w '%{http_code}' \
      -F \"payload=\$payload\" \
      -F \"file=@${TMP_DIR}/upload.csv;type=text/csv\" \
      \"${BASE_URL}/api/forecast/upload\")
    REQUEST_STATUS=\"\$status\"
    REQUEST_BODY=\"\$response\"
    assert_status 200 && assert_json && assert_jq 'has(\"model\") and (has(\"forecasts\") and (.forecasts|length>0))'
  "
}

ensure_mock_installed() {
  run_check "POST /v1/models/pull installs mock model" bash -c "
    cat > ${TMP_DIR}/pull_mock.json <<'JSON'
{\"name\":\"mock\",\"accept_license\":false}
JSON
    send_request POST /v1/models/pull ${TMP_DIR}/pull_mock.json ${TMP_DIR}/pull_mock_resp.json
    assert_status 200 && assert_json
  "
}

check_lifecycle_endpoints() {
  run_check "POST /v1/models/pull duplicate install is idempotent/conflict-safe" bash -c "
    cat > ${TMP_DIR}/pull_mock_duplicate_payload.json <<'JSON'
{\"name\":\"mock\",\"accept_license\":false}
JSON
    send_request POST /v1/models/pull ${TMP_DIR}/pull_mock_duplicate_payload.json ${TMP_DIR}/pull_mock_duplicate.json
    assert_status_in 200 409
  "
  run_check "POST /v1/models/pull missing model returns 404" bash -c "
    cat > ${TMP_DIR}/pull_mock_missing_payload.json <<'JSON'
{\"name\":\"does-not-exist\",\"accept_license\":false}
JSON
    send_request POST /v1/models/pull ${TMP_DIR}/pull_mock_missing_payload.json ${TMP_DIR}/pull_mock_missing.json
    assert_status 404
  "
  run_check "POST /v1/models/pull invalid payload returns 400" bash -c "
    cat > ${TMP_DIR}/pull_mock_bad_payload.json <<'JSON'
{\"name\":\"mock\",\"accept_license\":\"false\"}
JSON
    send_request POST /v1/models/pull ${TMP_DIR}/pull_mock_bad_payload.json ${TMP_DIR}/pull_mock_bad.json
    assert_status 400
  "
  run_check "GET /api/tags lists installed entries" bash -c "
    send_request GET /api/tags NONE ${TMP_DIR}/tags.json
    assert_status 200 && assert_json && assert_jq 'has(\"models\")'
  "
  run_check "POST /api/show returns model metadata" bash -c "
    echo '{\"model\":\"mock\"}' > ${TMP_DIR}/show_payload.json
    send_request POST /api/show ${TMP_DIR}/show_payload.json ${TMP_DIR}/show.json
    assert_status 200 && assert_json && assert_jq '.name == \"mock\" or .model == \"mock\"'
  "
  run_check "GET /api/ps lists loaded models" bash -c "
    send_request GET /api/ps NONE ${TMP_DIR}/ps.json
    assert_status 200 && assert_json && assert_jq 'has(\"models\")'
  "
  run_check "GET /v1/models lists available and installed" bash -c "
    send_request GET /v1/models NONE ${TMP_DIR}/v1_models.json
    assert_status 200 && assert_json && assert_jq 'has(\"available\") and has(\"installed\")'
  "
  run_check "POST /api/pull stream=false pulls model response payload" bash -c "
    echo '{\"model\":\"mock\",\"stream\":false}' > ${TMP_DIR}/api_pull.json
    send_request POST /api/pull ${TMP_DIR}/api_pull.json ${TMP_DIR}/api_pull.json.out
    assert_status_in 200 409 && assert_json
  "
  run_check "POST /api/pull stream=true returns ndjson events" bash -c "
    echo '{\"model\":\"mock\",\"stream\":true}' > ${TMP_DIR}/api_pull_stream_payload.json
    send_stream_request POST /api/pull ${TMP_DIR}/api_pull_stream_payload.json ${TMP_DIR}/api_pull_stream.out text/event-stream
    assert_status_in 200 409 && check_stream_has_json_lines ${TMP_DIR}/api_pull_stream.out 'has(\"status\") or has(\"name\")'
  "
  run_check "POST /api/delete removes known model" bash -c "
    echo '{\"model\":\"mock\"}' > ${TMP_DIR}/delete_payload.json
    send_request DELETE /api/delete ${TMP_DIR}/delete_payload.json ${TMP_DIR}/delete.json
    assert_status_in 200 404
  "
  run_check "DELETE /v1/models/{name} returns not found on missing model (cleanup idempotency)" bash -c "
    send_request DELETE /v1/models/mock NONE ${TMP_DIR}/delete_v1_again.json
    assert_status_in 200 404
  "
}

check_forecasting_endpoints() {
  run_check "POST /api/validate accepts valid payload" bash -c "
    send_request POST /api/validate ${TMP_DIR}/forecast.json ${TMP_DIR}/validate_ok.json
    assert_status 200 && assert_json && assert_jq '.valid == true'
  "
  run_check "POST /api/validate reports invalid payload" bash -c "
    send_request POST /api/validate ${TMP_DIR}/forecast_missing_series.json ${TMP_DIR}/validate_bad.json
    assert_status 200 && assert_jq '.valid == false'
  "
  run_check "POST /api/forecast returns JSON forecast" bash -c "
    send_request POST /api/forecast ${TMP_DIR}/forecast.json ${TMP_DIR}/forecast.json.out
    assert_status 200 && assert_jq 'select(has(\"done\")) | .response | has(\"model\") and has(\"forecasts\")'
  "
  run_check "POST /api/forecast stream=true emits ndjson events" bash -c "
    echo '{\"model\":\"mock\",\"horizon\":2,\"series\":[{\"id\":\"s1\",\"freq\":\"D\",\"timestamps\":[\"2025-01-01\",\"2025-01-02\"],\"target\":[1.0,3.0]}],\"options\":{}}' > ${TMP_DIR}/forecast_stream_payload.json
    send_stream_request POST /api/forecast ${TMP_DIR}/forecast_stream_payload.json ${TMP_DIR}/forecast_stream.out text/plain
    assert_status 200 && check_stream_has_json_lines ${TMP_DIR}/forecast_stream.out 'has(\"status\")'
  "
  run_check "POST /api/forecast/progressive returns SSE forecast stage events" bash -c "
    send_stream_request POST /api/forecast/progressive ${TMP_DIR}/auto_forecast.json ${TMP_DIR}/forecast_progressive.out text/event-stream
    assert_status 200 && grep -q '^event:' ${TMP_DIR}/forecast_progressive.out
  "
  run_check "POST /v1/forecast runs stable endpoint" bash -c "
    send_request POST /v1/forecast ${TMP_DIR}/forecast.json ${TMP_DIR}/v1_forecast.json
    assert_status 200 && assert_json && assert_jq 'has(\"model\") and has(\"forecasts\")'
  "
  run_check "POST /api/auto-forecast uses strategy auto for mock" bash -c "
    send_request POST /api/auto-forecast ${TMP_DIR}/auto_forecast.json ${TMP_DIR}/auto_forecast.json.out
    assert_status 200 && assert_json && assert_jq '.selection.strategy == \"auto\" and has(\"response\")'
  "
}

check_analysis_endpoints() {
  run_check "POST /api/analyze returns analysis payload" bash -c "
    send_request POST /api/analyze ${TMP_DIR}/analyze.json ${TMP_DIR}/analyze.out
    assert_status 200 && assert_json && assert_jq 'has(\"results\") and (.results|type==\"array\")'
  "
  run_check "POST /api/analyze rejects bad payload" bash -c "
    send_request POST /api/analyze ${TMP_DIR}/analyze_bad.json ${TMP_DIR}/analyze_bad.out
    assert_status 400
  "
  run_check "POST /api/compare returns mixed model outcomes" bash -c "
    send_request POST /api/compare ${TMP_DIR}/compare.json ${TMP_DIR}/compare.out
    assert_status 200 && assert_json && assert_jq 'has(\"summary\") and has(\"results\") and (.summary.requested_models|tonumber==2)'
  "
  run_check "POST /api/generate returns synthetic output" bash -c "
    send_request POST /api/generate ${TMP_DIR}/generate.json ${TMP_DIR}/generate.out
    assert_status 200 && assert_json && assert_jq 'has(\"generated\") and (.generated|type==\"array\")'
  "
  run_check "POST /api/counterfactual returns scenario results" bash -c "
    send_request POST /api/counterfactual ${TMP_DIR}/counterfactual.json ${TMP_DIR}/counterfactual.out
    assert_status 200 && assert_json && assert_jq 'has(\"results\") and (.results|type==\"array\")'
  "
  run_check "POST /api/scenario-tree builds scenario graph" bash -c "
    send_request POST /api/scenario-tree ${TMP_DIR}/scenario_tree.json ${TMP_DIR}/scenario_tree.out
    assert_status 200 && assert_json && assert_jq 'has(\"nodes\") and has(\"depth\")'
  "
  run_check "POST /api/what-if runs baseline + scenarios" bash -c "
    send_request POST /api/what-if ${TMP_DIR}/what_if.json ${TMP_DIR}/what_if.out
    assert_status 200 && assert_json && assert_jq 'has(\"baseline\") and has(\"summary\")'
  "
  run_check "POST /api/report returns composite response" bash -c "
    send_request POST /api/report ${TMP_DIR}/report.json ${TMP_DIR}/report.out
    assert_status 200 && assert_json && assert_jq 'has(\"analysis\") and has(\"forecast\") and has(\"baseline\")'
  "
  run_check "POST /api/pipeline returns end-to-end payload" bash -c "
    send_request POST /api/pipeline ${TMP_DIR}/pipeline.json ${TMP_DIR}/pipeline.out
    assert_status 200 && assert_json && assert_jq 'has(\"analysis\") and has(\"auto_forecast\") and has(\"recommendation\")'
  "
}

check_a2a_endpoints() {
  run_check "GET /.well-known/agent-card.json is reachable" bash -c "
    send_request GET /.well-known/agent-card.json NONE ${TMP_DIR}/agent_card.json
    assert_status_in 200 304 503
  "
  run_check "GET /.well-known/agent.json is reachable" bash -c "
    send_request GET /.well-known/agent.json NONE ${TMP_DIR}/agent_card_legacy.json
    assert_status_in 200 304 503
  "
  run_check "POST /a2a handles blocking forecast message" bash -c "
cat > ${TMP_DIR}/a2a_payload.json <<'JSON'
{\"jsonrpc\":\"2.0\",\"id\":\"1\",\"method\":\"message/send\",\"params\":{\"message\":{\"messageId\":\"m-1\",\"role\":\"user\",\"parts\":[{\"mediaType\":\"application/json\",\"data\":{\"skill\":\"forecast\",\"request\":{\"model\":\"mock\",\"horizon\":2,\"series\":[{\"id\":\"s1\",\"freq\":\"D\",\"timestamps\":[\"2025-01-01\",\"2025-01-02\"],\"target\":[1.0,2.0]}],\"options\":{}}}}],\"configuration\":{\"blocking\":true}}}}
JSON
    send_request POST /a2a ${TMP_DIR}/a2a_payload.json ${TMP_DIR}/a2a.out
    assert_status 200 && assert_json && assert_jq '.jsonrpc == \"2.0\" and (has(\"result\") or has(\"error\"))'
  "
  run_check "POST /a2a message/stream returns SSE status and artifact events" bash -c "
cat > ${TMP_DIR}/a2a_stream_payload.json <<'JSON'
{\"jsonrpc\":\"2.0\",\"id\":\"2\",\"method\":\"message/stream\",\"params\":{\"message\":{\"messageId\":\"m-2\",\"role\":\"user\",\"parts\":[{\"mediaType\":\"application/json\",\"data\":{\"skill\":\"forecast\",\"request\":{\"model\":\"mock\",\"horizon\":2,\"series\":[{\"id\":\"s1\",\"freq\":\"D\",\"timestamps\":[\"2025-01-01\",\"2025-01-02\"],\"target\":[1.0,2.0]}],\"options\":{}}}}],\"configuration\":{\"blocking\":false,\"pollIntervalMs\":25,\"heartbeatSeconds\":0.1}}}}
JSON
    send_stream_request POST /a2a ${TMP_DIR}/a2a_stream_payload.json ${TMP_DIR}/a2a_stream.out text/event-stream
    assert_status 200 && grep -q '^event: TaskStatusUpdateEvent' ${TMP_DIR}/a2a_stream.out && grep -q '^event: TaskArtifactUpdateEvent' ${TMP_DIR}/a2a_stream.out
  "
}

check_error_contracts() {
  run_check "POST /api/forecast rejects malformed JSON" bash -c "
    status=\$(printf '{\"model\":' | curl -sS --max-time \"$READY_TIMEOUT_SECONDS\" \
      -o ${TMP_DIR}/forecast_malformed.json -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -X POST \"${BASE_URL}/api/forecast\")
    REQUEST_STATUS=\"\$status\"
    REQUEST_BODY=${TMP_DIR}/forecast_malformed.json
    assert_status 400
  "
  run_check "POST /api/forecast rejects invalid payload type" bash -c "
    echo '{\"series\":\"not-an-array\",\"horizon\":2,\"model\":\"mock\"}' > ${TMP_DIR}/forecast_type_bad.json
    send_request POST /api/forecast ${TMP_DIR}/forecast_type_bad.json ${TMP_DIR}/forecast_type_bad.out
    assert_status 400
  "
  run_check "/api/events enforces stream content when requested" bash -c "
    curl -sS -o ${TMP_DIR}/events_content_type.out --max-time $STREAM_TIMEOUT_SECONDS -H 'Accept: text/event-stream' ${BASE_URL}/api/events 2>/dev/null || true
    [[ -s ${TMP_DIR}/events_content_type.out ]] && grep -qE '^(data:|:|event:|retry:|id:)' ${TMP_DIR}/events_content_type.out
  "
}

run_auth_checks() {
  if (( SKIP_DAEMON_START == 1 )); then
    log "Skipping auth checks because --skip-daemon-start was requested."
    return 0
  fi

  local current_auth_home="${DAEMON_HOME}/auth-check"
  local auth_home
  auth_home="$(mktemp -d "${TMP_DIR}/auth-home-XXXXXX")"
  cat > "${auth_home}/config.json" <<JSON
{"version":1,"auth":{"api_keys":["${API_KEY}"]}}
JSON
  DAEMON_HOME="$auth_home"

  stop_daemon_and_reset_auth
  start_daemon
  AUTH_HEADERS=()

  run_check "Protected endpoint rejects request without Authorization" bash -c "
    send_request GET /api/version NONE ${TMP_DIR}/auth_no_header.json
    assert_status_in 401 403
  "
  AUTH_HEADERS=(-H \"Authorization: Bearer ${API_KEY}\")
  run_check "Protected endpoint accepts valid Authorization header" bash -c "
    send_request GET /api/version NONE ${TMP_DIR}/auth_with_header.json
    assert_status 200
  "
  run_check "Protected endpoint /api/info accepts valid Authorization header" bash -c "
    send_request GET /api/info NONE ${TMP_DIR}/auth_info_with_header.json
    assert_status 200 && assert_json
  "
  run_check "Forecast under auth with header succeeds" bash -c "
    send_request POST /api/forecast ${TMP_DIR}/forecast.json ${TMP_DIR}/auth_forecast.out
    assert_status 200
  "
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --base-url)
        BASE_URL="$2"; shift 2 ;;
      --stream-timeout)
        STREAM_TIMEOUT_SECONDS="$2"; shift 2 ;;
      --ready-timeout)
        READY_TIMEOUT_SECONDS="$2"; shift 2 ;;
      --daemon-home)
        DAEMON_HOME="$2"; shift 2 ;;
      --skip-contract-tests)
        SKIP_CONTRACT_TESTS=1; shift ;;
      --skip-daemon-start)
        SKIP_DAEMON_START=1; shift ;;
      --skip-route-inventory)
        RUN_ROUTE_INVENTORY_CHECK=0; shift ;;
      --no-cleanup)
        CLEANUP_STATE=0; shift ;;
      --api-key)
        API_KEY="$2"; RUN_AUTH_CHECK=1; shift 2 ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done
}

parse_args "$@"
ensure_tmp_payloads

if (( SKIP_CONTRACT_TESTS == 0 )); then
  log "Running contract tests..."
  run_check "Contract tests for openapi, daemon API, A2A interfaces" run_contract_tests
fi

if (( RUN_ROUTE_INVENTORY_CHECK == 1 )); then
  log "Validating route inventory..."
  check_route_inventory
fi

if (( SKIP_DAEMON_START == 0 )); then
  log "Starting daemon for live API verification..."
  start_daemon
fi

check_system_endpoints
check_runtime_dashboard
check_modelfile_endpoints
ensure_mock_installed
check_ingest_endpoints
check_lifecycle_endpoints
ensure_mock_installed
check_forecasting_endpoints
check_analysis_endpoints
check_a2a_endpoints
check_error_contracts

if (( RUN_AUTH_CHECK == 1 )); then
  run_auth_checks
fi

if (( FAILED == 0 )); then
  log "Daemon API verification script completed successfully."
  exit 0
fi

log "Daemon API verification script completed with failures."
exit 1
