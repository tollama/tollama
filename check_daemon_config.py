
import requests
import json
import os

def check_daemon_info():
    try:
        response = requests.get("http://localhost:11435/api/info", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        env = data.get("env", {})
        print(f"Daemon TOLLAMA_FORECAST_TIMEOUT_SECONDS: {env.get('TOLLAMA_FORECAST_TIMEOUT_SECONDS')}")
        
        # Also check if we can infer the default from code (not directly possible via API unless exposed)
        # But we can check if the env var was picked up.
        
    except Exception as e:
        print(f"Error checking daemon: {e}")

if __name__ == "__main__":
    check_daemon_info()
