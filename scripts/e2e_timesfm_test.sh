#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting TimesFM Family E2E Test..."

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# 1. Pull the model
echo "Step 1: Pulling timesfm-2.5-200m..."
tollama pull timesfm-2.5-200m --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled timesfm-2.5-200m.${NC}"
else
    echo -e "${RED}Failed to pull timesfm-2.5-200m.${NC}"
    exit 1
fi

# 2. Run the model
echo "Step 2: Running prediction..."
INPUT_FILE="examples/timesfm_2p5_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run timesfm-2.5-200m --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran timesfm-2.5-200m prediction.${NC}"
else
    echo -e "${RED}Failed to run timesfm-2.5-200m prediction.${NC}"
    echo "Note: Ensure you have 'timesfm' runner installed via 'pip install .[runner_timesfm]'"
    exit 1
fi

echo -e "${GREEN}TimesFM Family E2E Test Completed Successfully!${NC}"
