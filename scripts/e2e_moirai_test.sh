#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting Moirai 2.0 E2E Test..."

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
# Create a temporary input file if examples/moirai_request.json doesn't exist
INPUT_FILE="examples/moirai_request.json"
if [ ! -f "$INPUT_FILE" ]; then
    echo "Creating temporary input file..."
    cat <<EOF > moirai_request_tmp.json
{
  "model": "moirai-2.0-R-small",
  "horizon": 10,
  "quantiles": [0.1, 0.5, 0.9],
  "series": [
    {
      "id": "s1",
      "freq": "H",
      "timestamps": ["2023-01-01T00:00:00Z", "2023-01-01T01:00:00Z", "2023-01-01T02:00:00Z"],
      "target": [1.0, 2.0, 3.0]
    }
  ]
}
EOF
    INPUT_FILE="moirai_request_tmp.json"
fi

tollama run moirai-2.0-R-small --input "$INPUT_FILE" --no-stream
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully ran moirai-2.0-R-small prediction.${NC}"
else
    echo -e "${RED}Failed to run moirai-2.0-R-small prediction.${NC}"
    echo "Note: Ensure you have 'uni2ts' installed via 'pip install .[runner_uni2ts]'"
    exit 1
fi

# Cleanup
if [ "$INPUT_FILE" == "moirai_request_tmp.json" ]; then
    rm moirai_request_tmp.json
fi

echo -e "${GREEN}Moirai 2.0 E2E Test Completed Successfully!${NC}"
