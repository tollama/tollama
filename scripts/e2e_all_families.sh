#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Tollama E2E Tests - Per-Family Suite  ${NC}"
echo -e "${BLUE}========================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS=()
FAILED=0

# Array of test scripts to run
declare -a TESTS=(
    "e2e_torch_test.sh"
    "e2e_timesfm_test.sh"
    "e2e_uni2ts_test.sh"
    "e2e_sundial_test.sh"
    "e2e_toto_test.sh"
)

# Run each test
for test in "${TESTS[@]}"; do
    echo ""
    echo -e "${YELLOW}Running: $test${NC}"
    echo "---"
    
    if bash "$SCRIPT_DIR/$test"; then
        RESULTS+=("${GREEN}✓${NC} $test")
    else
        RESULTS+=("${RED}✗${NC} $test")
        ((FAILED++))
    fi
done

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Results Summary                  ${NC}"
echo -e "${BLUE}========================================${NC}"
for result in "${RESULTS[@]}"; do
    echo -e "$result"
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED test(s) failed.${NC}"
    exit 1
fi
