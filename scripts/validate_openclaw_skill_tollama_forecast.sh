#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_DIR="$ROOT_DIR/skills/tollama-forecast"
SKILL_FILE="$SKILL_DIR/SKILL.md"

required_files=(
  "$SKILL_FILE"
  "$SKILL_DIR/bin/tollama-health.sh"
  "$SKILL_DIR/bin/tollama-models.sh"
  "$SKILL_DIR/bin/tollama-forecast.sh"
  "$SKILL_DIR/examples/simple_forecast.json"
  "$SKILL_DIR/examples/multi_series.json"
  "$SKILL_DIR/examples/covariates_forecast.json"
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

echo "OpenClaw tollama-forecast skill validation: OK"
