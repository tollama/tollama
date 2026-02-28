import json
from pathlib import Path

def compute_mape(actuals, preds):
    if not actuals or len(actuals) != len(preds): return None
    errors = []
    for a, p in zip(actuals, preds):
        if a != 0:
            errors.append(abs((a - p) / a))
    if not errors: return None
    return (sum(errors) / len(errors)) * 100

def compute_mase(actuals, preds, history):
    if not actuals or len(actuals) != len(preds): return None
    if len(history) < 2: return None
    naive_diffs = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
    mean_naive = sum(naive_diffs) / len(naive_diffs)
    if mean_naive == 0: return None
    mae = sum(abs(a - p) for a, p in zip(actuals, preds)) / len(actuals)
    return mae / mean_naive

result_path = Path("artifacts/realdata/hf-local/result.json")
data = json.loads(result_path.read_text())

for model_dir in Path("artifacts/realdata/hf-local/raw").iterdir():
    if not model_dir.is_dir(): continue
    for dataset_dir in model_dir.iterdir():
        if not dataset_dir.is_dir(): continue
        for entry_file in (dataset_dir / "benchmark_target_only").glob("*.json"):
            try:
                entry_data = json.loads(entry_file.read_text())
                entry = entry_data["entry"]
                if entry.get("status") == "pass":
                    actuals = entry_data["request"]["series"][0]["actuals"]
                    preds = entry_data["response"]["forecasts"][0]["mean"]
                    history = entry_data["request"]["series"][0]["target"]
                    mape = compute_mape(actuals, preds)
                    mase = compute_mase(actuals, preds, history)
                    
                    found = False
                    for i, r_entry in enumerate(data["entries"]):
                        if (r_entry.get("model") == entry.get("model") and 
                            r_entry.get("dataset") == entry.get("dataset") and 
                            r_entry.get("scenario") == "benchmark_target_only"):
                            # There could be multiple series per dataset, we need to match the specific series.
                            # The entry in result.json doesn't easily store series ID, but wait, the entry 
                            # in raw/ is 1:1 with the result.json entries if we match in order, or we can just calculate
                            # it directly over the raw entries and then rewrite result.json!
                            pass
            except Exception:
                pass

# Let's just completely rebuild data["entries"] metrics by patching them directly.
# Wait, result.json has a list of entries. I can just iterate over data["entries"].
for r_entry in data["entries"]:
    if r_entry.get("status") == "pass" and r_entry.get("scenario") == "benchmark_target_only":
        # Find corresponding raw file
        model = r_entry["model"]
        dataset = r_entry["dataset"]
        # The raw file is named after series ID. result.json doesn't have series ID.
        # But there's a 1-to-1 mapping. Let's just find the corresponding file.
        # Actually, in result.json, there's no series id. The summary is an average of the dataset.
        pass

# Better approach: Read ALL raw files, calculate mape/mase, update the raw file's entry, then reconstruct result.json entries entirely from raw files!
entries = []
for model_dir in Path("artifacts/realdata/hf-local/raw").iterdir():
    if not model_dir.is_dir(): continue
    for dataset_dir in model_dir.iterdir():
        if not dataset_dir.is_dir(): continue
        for scenario_dir in dataset_dir.iterdir():
            if not scenario_dir.is_dir(): continue
            for entry_file in scenario_dir.glob("*.json"):
                entry_data = json.loads(entry_file.read_text())
                entry = entry_data["entry"]
                if entry.get("status") == "pass" and entry.get("scenario") == "benchmark_target_only":
                    actuals = entry_data["request"]["series"][0]["actuals"]
                    preds = entry_data["response"]["forecasts"][0]["mean"]
                    history = entry_data["request"]["series"][0]["target"]
                    mape = compute_mape(actuals, preds)
                    mase = compute_mase(actuals, preds, history)
                    if "metrics" not in entry: entry["metrics"] = {}
                    if mape is not None: entry["metrics"]["mape"] = mape
                    if mase is not None: entry["metrics"]["mase"] = mase
                entries.append(entry)

data["entries"] = entries
result_path.write_text(json.dumps(data, indent=2))
