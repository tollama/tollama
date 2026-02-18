#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting Uni2ts (Moirai) Family E2E Test..."

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# 1. Pull the model
echo "Step 1: Pulling moirai-2.0-R-small..."
tollama pull moirai-2.0-R-small --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled moirai-2.0-R-small.${NC}"
else
    echo -e "${RED}Failed to pull moirai-2.0-R-small.${NC}"
    exit 1
fi

# 2. Run the model
echo "Step 2: Running prediction..."
INPUT_FILE="examples/moirai_2p0_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run moirai-2.0-R-small --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran moirai-2.0-R-small prediction.${NC}"
else
    echo -e "${RED}Failed to run moirai-2.0-R-small prediction.${NC}"
    echo "Note: Ensure you have 'uni2ts' runner installed via 'pip install .[runner_uni2ts]'"
    exit 1
fi

echo -e "${GREEN}Uni2ts (Moirai) Family E2E Test Completed Successfully!${NC}"
