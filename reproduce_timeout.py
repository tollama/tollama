
import os
import time
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from tollama.daemon.app import create_app


def main():
    print("Starting reproduction script...")
    start_time = time.time()

    # Set up environment
    # Increase timeout for reproduction script to see how long it actually takes
    os.environ["TOLLAMA_FORECAST_TIMEOUT_SECONDS"] = "300"

    print("Creating app...")
    app = create_app()
    client = TestClient(app)

    model_name = "timesfm-2.5-200m"

    # Pull model
    print(f"Pulling model {model_name}...")
    pull_start = time.time()
    response = client.post("/api/pull", json={"model": model_name, "stream": False})
    pull_duration = time.time() - pull_start
    print(f"Pull finished in {pull_duration:.2f}s. Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Pull failed: {response.text}")
        return

    # Prepare forecast payload
    start = datetime(2025, 1, 1, tzinfo=UTC)
    payload = {
        "model": model_name,
        "horizon": 24,
        "series": [
            {
                "id": "s1",
                "freq": "H",
                "timestamps": [
                    (start + timedelta(hours=i)).isoformat()
                    for i in range(168) # 1 week of history
                ],
                "target": [float(i) for i in range(168)],
            }
        ],
    }

    # Run forecast
    print("Running forecast...")
    forecast_start = time.time()
    try:
        response = client.post("/v1/forecast", json=payload)
        forecast_duration = time.time() - forecast_start
        print(f"Forecast finished in {forecast_duration:.2f}s. Status: {response.status_code}")
        if response.status_code == 200:
            print("Forecast successful.")
            msg = response.json()
            warnings = msg.get("warnings")
            if warnings:
                print(f"Warnings: {warnings}")
        else:
            print(f"Forecast failed: {response.text}")
    except Exception as e:
        forecast_duration = time.time() - forecast_start
        print(f"Forecast crashed after {forecast_duration:.2f}s: {e}")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
