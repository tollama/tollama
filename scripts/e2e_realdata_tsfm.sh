#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-pr}"
MODEL="${2:-all}"
BASE_URL="${3:-http://127.0.0.1:11435}"
OUTPUT_DIR="${4:-$ROOT_DIR/artifacts/e2e_realdata}"

if ! command -v python >/dev/null 2>&1; then
  echo -e "${RED}Error: python not found in PATH.${NC}"
  exit 1
fi

echo -e "${YELLOW}Running real-data TSFM E2E suite...${NC}"
echo "mode=$MODE model=$MODEL base_url=$BASE_URL"
echo "output_dir=$OUTPUT_DIR"

python "$ROOT_DIR/scripts/e2e_realdata/run_tsfm_realdata.py" \
  --mode "$MODE" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --output-dir "$OUTPUT_DIR"

status=$?
if [[ $status -eq 0 ]]; then
  echo -e "${GREEN}Real-data TSFM E2E passed.${NC}"
elif [[ $status -eq 2 ]]; then
  echo -e "${RED}Real-data TSFM E2E failed due to infra/preflight issues.${NC}"
else
  echo -e "${RED}Real-data TSFM E2E gate failed.${NC}"
fi

exit $status
