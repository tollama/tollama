#!/usr/bin/env bash
set -euo pipefail

EXIT_USAGE=2

SKILL_BIN_DIR="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
MODELS_SCRIPT="$SKILL_BIN_DIR/tollama-models.sh"

PASS_ARGS=()

usage() {
  cat <<'USAGE' >&2
Usage: tollama-info.sh [--base-url URL] [--timeout SEC] [--section daemon|models|runners|env|all]
USAGE
}

while (($# > 0)); do
  case "$1" in
    --base-url|--timeout|--section)
      key="$1"
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit "$EXIT_USAGE"
      }
      PASS_ARGS+=("$key" "$1")
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

exec "$MODELS_SCRIPT" info "${PASS_ARGS[@]}"
