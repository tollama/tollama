#!/usr/bin/env bash
set +e

cd /Users/yongchoelchoi/Documents/GitHub/tollama || exit 1
export PATH="$PWD/.venv_langchain_e2e/bin:$PATH"

# Build a long-context payload (>=512 points) for models that require
# substantial history windows.
"$PWD/.venv_langchain_e2e/bin/python" - <<'PY'
import json
from datetime import date, timedelta

start = date(2024, 1, 1)
n = 520
timestamps = [(start + timedelta(days=i)).isoformat() for i in range(n)]
target = [100.0 + 0.1 * i + ((i % 7) - 3) * 0.2 for i in range(n)]
payload = {
  "model": "granite-ttm-r2",
  "horizon": 12,
  "quantiles": [],
  "series": [
    {
      "id": "series-001",
      "freq": "D",
      "timestamps": timestamps,
      "target": target,
    }
  ],
  "options": {},
}

with open("/tmp/tollama_long_context_request.json", "w", encoding="utf-8") as f:
  json.dump(payload, f)

patchtst_payload = dict(payload)
patchtst_payload["model"] = "patchtst"
patchtst_payload["horizon"] = 7
with open("/tmp/tollama_patchtst_request.json", "w", encoding="utf-8") as f:
  json.dump(patchtst_payload, f)
PY

models=(
  "mock examples/request.json"
  "chronos2 examples/chronos2_request.json"
  "granite-ttm-r2 /tmp/tollama_long_context_request.json"
  "timesfm-2.5-200m examples/timesfm_2p5_request.json"
  "moirai-2.0-R-small examples/moirai_2p0_request.json"
  "sundial-base-128m examples/sundial_request.json"
  "toto-open-base-1.0 examples/toto_request.json"
  "lag-llama examples/lag_llama_request.json"
  "patchtst /tmp/tollama_patchtst_request.json"
  "tide /tmp/tollama_long_context_request.json"
  "nhits /tmp/tollama_long_context_request.json"
  "nbeatsx /tmp/tollama_long_context_request.json"
)

summary=/tmp/tollama_all_e2e_summary.txt
: > "$summary"

for item in "${models[@]}"; do
  model="${item%% *}"
  input="${item#* }"
  echo "=== $model ==="

  tollama pull "$model" --accept-license >/tmp/pull_${model}.log 2>&1 || \
    tollama pull "$model" >/tmp/pull_${model}.log 2>&1

  tollama run "$model" --input "$input" --no-stream >/tmp/run_${model}.out 2>/tmp/run_${model}.err
  code=$?
  if [ $code -eq 0 ]; then
    echo "PASS $model" | tee -a "$summary"
  else
    echo "FAIL $model" | tee -a "$summary"
    tail -n 6 /tmp/run_${model}.err
  fi

done

echo "--- SUMMARY ---"
cat "$summary"
