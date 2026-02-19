#!/bin/bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DAEMON_PORT=11435
DAEMON_HOST="127.0.0.1"
DAEMON_PID=""

echo -e "${YELLOW}Starting Skills E2E Test...${NC}"

# Cleanup function
cleanup() {
    if [[ -n "$DAEMON_PID" ]]; then
        echo "Stopping tollama daemon (PID: $DAEMON_PID)..."
        kill "$DAEMON_PID" || true
    fi
}
trap cleanup EXIT

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# Check if port is already in use
if lsof -i :$DAEMON_PORT >/dev/null; then
    echo -e "${RED}Error: Port $DAEMON_PORT is already in use. Please stop any running instances of tollama.${NC}"
    exit 1
fi

# Start tollama daemon
echo "Starting tollama daemon..."
# We use nohup to detach it, but keep track of PID
# Assuming 'tollamad' is in PATH or accessible. If not, might need 'python -m tollama.daemon.main'
if command -v tollamad &> /dev/null; then
    tollamad &
    DAEMON_PID=$!
else
    echo "tollamad not found in PATH, trying python module..."
    python3 -m tollama.daemon.main &
    DAEMON_PID=$!
fi

echo "Daemon PID: $DAEMON_PID"

# Wait for daemon to be ready
echo "Waiting for daemon to be ready on port $DAEMON_PORT..."
MAX_RETRIES=30
for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s "http://$DAEMON_HOST:$DAEMON_PORT/api/health" >/dev/null; then
        echo -e "${GREEN}Daemon is ready!${NC}"
        break
    fi
    if [[ $i -eq $MAX_RETRIES ]]; then
        echo -e "${RED}Timeout waiting for daemon.${NC}"
        exit 1
    fi
    sleep 1
done

# Step 1: Validate Skill Structure
echo ""
echo -e "${YELLOW}Step 1: Validating Skill Structure...${NC}"
if "$SCRIPT_DIR/validate_openclaw_skill_tollama_forecast.sh"; then
    echo -e "${GREEN}Skill validation passed.${NC}"
else
    echo -e "${RED}Skill validation failed.${NC}"
    exit 1
fi

# Step 2: Run Skill Execution
echo ""
echo -e "${YELLOW}Step 2: Running Skill Execution (Mock Model)...${NC}"
SKILL_BIN="$ROOT_DIR/skills/tollama-forecast/bin/tollama-forecast.sh"
SKILL_INPUT="$ROOT_DIR/skills/tollama-forecast/examples/simple_forecast.json"

if [[ ! -f "$SKILL_BIN" ]]; then
    echo -e "${RED}Error: Skill binary not found at $SKILL_BIN${NC}"
    exit 1
fi

if [[ ! -f "$SKILL_INPUT" ]]; then
    echo -e "${RED}Error: Skill input example not found at $SKILL_INPUT${NC}"
    exit 1
fi

# Run the forecast using the mock model (which doesn't require pulling heavy weights)
# We override the model in the input json via command line arg to be sure, or just rely on 'mock' being in the json.
# The `simple_forecast.json` already uses "model": "mock".

if "$SKILL_BIN" --model mock --input "$SKILL_INPUT" --base-url "http://$DAEMON_HOST:$DAEMON_PORT"; then
    echo -e "${GREEN}Skill execution passed.${NC}"
else
    echo -e "${RED}Skill execution failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Skill E2E Test Completed Successfully!${NC}"
exit 0
