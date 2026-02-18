
import os
import subprocess
import sys
import time

import requests

DAEMON_PORT = 11436
BASE_URL = f"http://localhost:{DAEMON_PORT}"

def wait_for_daemon():
    for _ in range(30):
        try:
            requests.get(f"{BASE_URL}/v1/health", timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False

def run_reproduction():
    print("Starting isolated reproduction script...")
    
    # Start Daemon with custom timeout
    env = os.environ.copy()
    env["TOLLAMA_FORECAST_TIMEOUT_SECONDS"] = "600"
    env["TOLLAMA_HOST"] = f"127.0.0.1:{DAEMON_PORT}"
    
    print(f"Launching daemon on port {DAEMON_PORT} with timeout 600s...")
    # Use sys.executable to ensure we use the same environment
    daemon_proc = subprocess.Popen(
        [sys.executable, "-m", "tollama.cli.main", "serve", "--port", str(DAEMON_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        if not wait_for_daemon():
            print("Failed to start daemon")
            # Force print logs
            if daemon_proc.poll() is not None:
                print(f"Daemon exited with code {daemon_proc.returncode}")
            return
            
        print("Daemon ready. Pulling model...")
        subprocess.run(
            [
                sys.executable, "-m", "tollama.cli.main", "pull",
                "timesfm-2.5-200m", "--base-url", BASE_URL
            ],
            check=True
        )
        
        print("Running forecast...")
        start_time = time.time()
        
        # Use a distinctive timeout for CLI to verify
        result = subprocess.run(
            [
                sys.executable, "-m", "tollama.cli.main", "run", "timesfm-2.5-200m",
                "--input", "examples/timesfm_2p5_request.json",
                "--no-stream",
                "--timeout", "600",
                "--base-url", BASE_URL
            ],
            capture_output=True,
            text=True
        )
        
        duration = time.time() - start_time
        print(f"Forecast finished in {duration:.2f}s")
        
        if result.returncode != 0:
            print("Forecast FAILED!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        else:
            print("Forecast SUCCESS!")
            
    finally:
        print("Stopping daemon...")
        if daemon_proc.poll() is None:
            daemon_proc.terminate()
            try:
                daemon_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                daemon_proc.kill()
            
        print("\n--- DAEMON STDOUT ---")
        print(daemon_proc.stdout.read())
        print("\n--- DAEMON STDERR ---")
        print(daemon_proc.stderr.read())

if __name__ == "__main__":
    run_reproduction()
