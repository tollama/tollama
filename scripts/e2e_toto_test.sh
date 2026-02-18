#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting Toto Family E2E Test..."

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# 1. Pull the model
echo "Step 1: Pulling toto-open-base-1.0..."
tollama pull toto-open-base-1.0 --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled toto-open-base-1.0.${NC}"
else
    echo -e "${RED}Failed to pull toto-open-base-1.0.${NC}"
    exit 1
fi

# 2. Run the model
echo "Step 2: Running prediction..."
INPUT_FILE="examples/toto_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run toto-open-base-1.0 --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran toto-open-base-1.0 prediction.${NC}"
else
    echo -e "${RED}Failed to run toto-open-base-1.0 prediction.${NC}"
    echo "Note: Ensure you have 'toto' runner installed via 'pip install .[runner_toto]'"
    exit 1
fi

echo -e "${GREEN}Toto Family E2E Test Completed Successfully!${NC}"
