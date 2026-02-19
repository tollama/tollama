#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_DIR="$ROOT_DIR/skills/tollama-forecast"
SKILL_FILE="$SKILL_DIR/SKILL.md"

required_files=(
  "$SKILL_FILE"
  "$SKILL_DIR/bin/_tollama_lib.sh"
  "$SKILL_DIR/bin/tollama-health.sh"
  "$SKILL_DIR/bin/tollama-models.sh"
  "$SKILL_DIR/bin/tollama-forecast.sh"
  "$SKILL_DIR/bin/tollama-pull.sh"
  "$SKILL_DIR/bin/tollama-rm.sh"
  "$SKILL_DIR/bin/tollama-info.sh"
  "$SKILL_DIR/openai-tools.json"
  "$SKILL_DIR/examples/simple_forecast.json"
  "$SKILL_DIR/examples/multi_series.json"
  "$SKILL_DIR/examples/covariates_forecast.json"
  "$SKILL_DIR/examples/metrics_forecast.json"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

frontmatter="$(awk '
  NR==1 && $0=="---" { in_fm=1; next }
  in_fm && $0=="---" { exit }
  in_fm { print }
' "$SKILL_FILE")"

if [[ -z "$frontmatter" ]]; then
  echo "Invalid SKILL.md: missing YAML frontmatter" >&2
  exit 1
fi

for key in name description homepage user-invocable metadata; do
  if ! grep -qE "^${key}:" <<<"$frontmatter"; then
    echo "Invalid SKILL.md: missing frontmatter key '$key'" >&2
    exit 1
  fi
done

metadata_line="$(grep -E '^metadata:' <<<"$frontmatter" || true)"
if [[ -z "$metadata_line" ]]; then
  echo "Invalid SKILL.md: metadata line not found" >&2
  exit 1
fi

metadata_json="${metadata_line#metadata: }"
if ! python3 - <<'PY' "$metadata_json"; then
import json
import sys

raw = sys.argv[1]
obj = json.loads(raw)
if not isinstance(obj, dict):
    raise SystemExit("metadata must be a JSON object")

openclaw = obj.get("openclaw")
if not isinstance(openclaw, dict):
    raise SystemExit("metadata.openclaw must be an object")

requires = openclaw.get("requires")
if not isinstance(requires, dict):
    raise SystemExit("metadata.openclaw.requires must be an object")

bins = requires.get("bins")
if not isinstance(bins, list) or "bash" not in bins:
    raise SystemExit("metadata.openclaw.requires.bins must include 'bash'")

any_bins = requires.get("anyBins")
if not isinstance(any_bins, list):
    raise SystemExit("metadata.openclaw.requires.anyBins must be a list")

for required in ("tollama", "curl"):
    if required not in any_bins:
        raise SystemExit(
            f"metadata.openclaw.requires.anyBins must include {required!r}",
        )
PY
  echo "Invalid SKILL.md: metadata must be a single-line JSON object" >&2
  exit 1
fi

for script in "$SKILL_DIR"/bin/*.sh; do
  bash -n "$script"
  if [[ ! -x "$script" ]]; then
    echo "Script is not executable: $script" >&2
    exit 1
  fi
done

for json_file in "$SKILL_DIR"/examples/*.json; do
  python3 -m json.tool "$json_file" >/dev/null
done

python3 -m json.tool "$SKILL_DIR/openai-tools.json" >/dev/null
python3 - <<'PY' "$SKILL_DIR/openai-tools.json"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)

tools = payload.get("tools")
if not isinstance(tools, list) or not tools:
    raise SystemExit("openai-tools.json must include a non-empty tools array")

names = set()
for item in tools:
    if not isinstance(item, dict) or item.get("type") != "function":
        raise SystemExit("all entries in openai-tools.json must be function tools")
    function = item.get("function")
    if not isinstance(function, dict):
        raise SystemExit("tool.function must be an object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise SystemExit("tool.function.name must be a non-empty string")
    names.add(name)

required = {"tollama_forecast", "tollama_health", "tollama_models"}
missing = required.difference(names)
if missing:
    raise SystemExit(f"openai-tools.json missing required tool names: {sorted(missing)!r}")
PY

echo "OpenClaw tollama-forecast skill validation: OK"
