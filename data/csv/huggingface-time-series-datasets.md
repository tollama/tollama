# Hugging Face Time-Series Forecasting Datasets (Download Guide)

This document lists practical Hugging Face datasets for time-series forecasting and shows how to download them directly.

Reference page:
- https://huggingface.co/datasets?task_categories=task_categories%3Atime-series-forecasting

## 1) Quick Start

### Option A: Download full dataset repository files (raw files)

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

repo_id = "pkr7098/time-series-forecasting-datasets"
local_dir = "./data/csv/hf_time_series/pkr7098_time-series-forecasting-datasets"
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
)
print(f"Downloaded to: {local_dir}")
PY
```

### Option B: Load by split using Hugging Face Datasets

```bash
python -m pip install -U datasets
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("monash_tsf", "nn5_daily")
print(ds)
print(ds["train"][0])
PY
```

## 2) Recommended Datasets

## A. pkr7098/time-series-forecasting-datasets

- Dataset: https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets
- Typical files in this repo include benchmark CSVs used in long-horizon forecasting:
  - `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
  - `electricity.csv`, `traffic.csv`, `weather.csv`
  - `exchange_rate.csv`, `national_illness.csv`

Direct file-style URLs (examples):
- https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv
- https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/electricity.csv
- https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/traffic.csv

Python test:

```python
import pandas as pd

url = "https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTh1.csv"
df = pd.read_csv(url)
print(df.shape)
print(df.head())
```

## B. monash_tsf (Monash Time Series Forecasting Repository)

- Dataset: https://huggingface.co/datasets/monash_tsf
- Multiple benchmark subsets (configs), for example:
  - `nn5_daily`
  - `tourism_monthly`
  - `m4_hourly`

Load specific config:

```python
from datasets import load_dataset

nn5 = load_dataset("monash_tsf", "nn5_daily")
tourism = load_dataset("monash_tsf", "tourism_monthly")
print(nn5)
print(tourism)
```

## C. Additional HF time-series datasets (search-driven)

Use task/category filters directly on Hugging Face and pick a dataset card:
- https://huggingface.co/datasets?task_categories=task_categories%3Atime-series-forecasting

If a dataset card has files in the Files tab, you can usually download by either:
1. `snapshot_download(...)` for the whole dataset repository
2. `load_dataset(dataset_id, config_name)` for split-based access

## 3) Batch Download Script (multiple dataset repos)

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

repos = [
    "pkr7098/time-series-forecasting-datasets",
    "monash_tsf",
]

for repo_id in repos:
    local_dir = f"./data/csv/hf_time_series/{repo_id.replace('/', '_')}"
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
print("Done")
PY
```

## 4) Notes for Tollama Usage

- Many HF datasets are not single CSV files; some are parquet/json with dataset scripts.
- For Tollama quick experiments, `pkr7098/time-series-forecasting-datasets` is convenient because it exposes benchmark CSV files directly.
- Keep your local path convention stable (for example: `data/csv/hf_time_series/...`) so experiments and tests are reproducible.

## 5) Troubleshooting

- 401/403 error:
  - Public datasets usually do not require auth, but gated datasets do.
  - Run `huggingface-cli login` if authentication is required.
- Slow download:
  - Retry later or mirror only required files.
- Large dataset:
  - Use `allow_patterns` in `snapshot_download` to fetch only required files.

Example with file filtering:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="pkr7098/time-series-forecasting-datasets",
    repo_type="dataset",
    local_dir="./data/csv/hf_time_series/pkr7098_filtered",
    allow_patterns=["*ETT*.csv", "electricity.csv", "traffic.csv"],
)
```
