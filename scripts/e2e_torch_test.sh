#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Starting Torch Family E2E Test (Chronos2 + Granite TTM)..."

# Check if tollama is installed
if ! command -v tollama &> /dev/null; then
    echo -e "${RED}Error: tollama is not installed or not in PATH.${NC}"
    exit 1
fi

# Test Chronos2
echo ""
echo -e "${YELLOW}Testing Chronos2 Model...${NC}"
echo "Step 1: Pulling chronos2..."
tollama pull chronos2 --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled chronos2.${NC}"
else
    echo -e "${RED}Failed to pull chronos2.${NC}"
    exit 1
fi

echo "Step 2: Running chronos2 prediction..."
INPUT_FILE="examples/chronos2_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run chronos2 --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran chronos2 prediction.${NC}"
else
    echo -e "${RED}Failed to run chronos2 prediction.${NC}"
    echo "Note: Ensure you have 'torch' runner installed via 'pip install .[runner_torch]'"
    exit 1
fi

# Test Granite TTM
echo ""
echo -e "${YELLOW}Testing Granite TTM Model...${NC}"
echo "Step 1: Pulling granite-ttm-r2..."
tollama pull granite-ttm-r2 --accept-license
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully pulled granite-ttm-r2.${NC}"
else
    echo -e "${RED}Failed to pull granite-ttm-r2.${NC}"
    exit 1
fi

echo "Step 2: Running granite-ttm-r2 prediction..."
INPUT_FILE="examples/granite_ttm_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: $INPUT_FILE not found.${NC}"
    exit 1
fi

tollama run granite-ttm-r2 --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran granite-ttm-r2 prediction.${NC}"
else
    echo -e "${RED}Failed to run granite-ttm-r2 prediction.${NC}"
    echo "Note: Ensure you have 'torch' runner installed via 'pip install .[runner_torch]'"
    exit 1
fi

echo ""
echo -e "${GREEN}Torch Family E2E Test Completed Successfully!${NC}"
