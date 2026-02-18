#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting Sundial Family E2E Test..."

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# 1. Pull the model
echo "Step 1: Pulling sundial-base-128m..."
tollama pull sundial-base-128m --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled sundial-base-128m.${NC}"
else
    echo -e "${RED}Failed to pull sundial-base-128m.${NC}"
    exit 1
fi

# 2. Run the model
echo "Step 2: Running prediction..."
INPUT_FILE="examples/sundial_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run sundial-base-128m --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran sundial-base-128m prediction.${NC}"
else
    echo -e "${RED}Failed to run sundial-base-128m prediction.${NC}"
    echo "Note: Ensure you have 'sundial' runner installed via 'pip install .[runner_sundial]'"
    exit 1
fi

echo -e "${GREEN}Sundial Family E2E Test Completed Successfully!${NC}"
