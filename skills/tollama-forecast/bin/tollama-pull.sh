#!/usr/bin/env bash
set -euo pipefail

EXIT_USAGE=2

SKILL_BIN_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
MODELS_SCRIPT="$SKILL_BIN_DIR/tollama-models.sh"

MODEL=""
PASS_ARGS=()

usage() {
  cat <<'USAGE' >&2
Usage: tollama-pull.sh --model NAME [--base-url URL] [--timeout SEC] [--accept-license]
USAGE
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
    --base-url|--timeout)
      key="$1"
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      PASS_ARGS+=("$key" "$1")
      ;;
    --accept-license)
      PASS_ARGS+=("$1")
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

exec "$MODELS_SCRIPT" pull "$MODEL" "${PASS_ARGS[@]}"
