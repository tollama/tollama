#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Starting tollama serve in background..."
tollama serve > .test_daemon.log 2>&1 &
DAEMON_PID=$!

echo "Waiting for daemon to be ready..."
sleep 5
for i in {1..10}; do
  if curl -s http://127.0.0.1:11435/api/version > /dev/null; then
    echo -e "${GREEN}Daemon is ready.${NC}"
    break
  fi
  sleep 2
done

if ! curl -s http://127.0.0.1:11435/api/version > /dev/null; then
  echo -e "${RED}Daemon failed to start.${NC}"
  cat .test_daemon.log
  kill $DAEMON_PID || true
  exit 1
fi

echo "Running HuggingFace datasets E2E evaluation for all 6 models..."
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model all \
  --catalog-path scripts/e2e_realdata/hf_dataset_catalog.yaml \
  --output-dir artifacts/realdata/hf-local \
  --allow-kaggle-fallback
STATUS=$?

echo "Stopping daemon..."
kill $DAEMON_PID || true
rm .test_daemon.log

if [ $STATUS -eq 0 ]; then
  echo -e "${GREEN}Evaluation completed successfully.${NC}"
else
  echo -e "${RED}Evaluation failed with exit code $STATUS.${NC}"
fi

exit $STATUS
