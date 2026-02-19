#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_URL="http://127.0.0.1:11435"
SERVER_NAME="tollama"
COMMAND_NAME="tollama-mcp"
BASE_URL="${TOLLAMA_BASE_URL:-$DEFAULT_BASE_URL}"
CONFIG_PATH_ARG=""
DRY_RUN=0

usage() {
  cat <<'USAGE' >&2
Usage: install_mcp.sh [--config PATH] [--server-name NAME] [--command CMD] [--base-url URL] [--dry-run]
USAGE
}

while (($# > 0)); do
  case "$1" in
    --config)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit 2
      }
      CONFIG_PATH_ARG="$1"
      ;;
    --server-name)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit 2
      }
      SERVER_NAME="$1"
      ;;
    --command)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit 2
      }
      COMMAND_NAME="$1"
      ;;
    --base-url)
      shift
      [[ $# -gt 0 ]] || {
        usage
        exit 2
      }
      BASE_URL="$1"
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required." >&2
  exit 1
fi

if [[ -n "$CONFIG_PATH_ARG" ]]; then
  CONFIG_PATH="$CONFIG_PATH_ARG"
else
  case "$(uname -s)" in
    Darwin)
      CONFIG_PATH="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
      ;;
    Linux)
      CONFIG_PATH="$HOME/.config/Claude/claude_desktop_config.json"
      ;;
    *)
      echo "Error: unsupported OS. Use --config to set Claude config path explicitly." >&2
      exit 1
      ;;
  esac
fi

CONFIG_DIR="$(cd "${CONFIG_PATH%/*}" && pwd 2>/dev/null || true)"
if [[ -z "$CONFIG_DIR" ]]; then
  CONFIG_DIR="${CONFIG_PATH%/*}"
fi
mkdir -p "$CONFIG_DIR"

BACKUP_PATH=""
if [[ -f "$CONFIG_PATH" ]]; then
  BACKUP_PATH="${CONFIG_PATH}.bak.$(date +%Y%m%d%H%M%S)"
  cp "$CONFIG_PATH" "$BACKUP_PATH"
fi

python3 - "$CONFIG_PATH" "$SERVER_NAME" "$COMMAND_NAME" "$BASE_URL" "$DRY_RUN" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
server_name = sys.argv[2]
command_name = sys.argv[3]
base_url = sys.argv[4]
dry_run = sys.argv[5] == "1"

if config_path.exists():
    with config_path.open("r", encoding="utf-8") as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error: invalid JSON in {config_path}: {exc}")
else:
    payload = {}

if not isinstance(payload, dict):
    raise SystemExit(f"Error: top-level config must be an object in {config_path}")

mcp_servers = payload.get("mcpServers")
if not isinstance(mcp_servers, dict):
    mcp_servers = {}

existing = mcp_servers.get(server_name)
server_payload: dict[str, object]
if isinstance(existing, dict):
    server_payload = dict(existing)
else:
    server_payload = {}

server_payload["command"] = command_name
existing_env = server_payload.get("env")
env_payload: dict[str, object]
if isinstance(existing_env, dict):
    env_payload = dict(existing_env)
else:
    env_payload = {}
env_payload["TOLLAMA_BASE_URL"] = base_url
server_payload["env"] = env_payload

mcp_servers[server_name] = server_payload
payload["mcpServers"] = mcp_servers

if dry_run:
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0)

config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")

print(f"Updated Claude Desktop MCP config: {config_path}")
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run complete. No file changes applied." >&2
else
  if [[ -n "$BACKUP_PATH" ]]; then
    echo "Backup created: $BACKUP_PATH" >&2
  fi
  echo "Installed MCP server '$SERVER_NAME' with command '$COMMAND_NAME'." >&2
fi
