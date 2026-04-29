# Hugging Face Time-Series Forecasting Datasets

Last checked: 2026-04-30 KST.

Primary listing:
- https://huggingface.co/datasets?task_categories=task_categories%3Atime-series-forecasting

API endpoints used for this snapshot:
- https://huggingface.co/api/datasets?filter=task_categories:time-series-forecasting&limit=100&full=true
- https://huggingface.co/api/datasets/pkr7098/time-series-forecasting-datasets
- https://huggingface.co/api/datasets/Monash-University/monash_tsf

The filtered Hugging Face page showed 938 time-series forecasting datasets at the time of this refresh. Do not blindly snapshot all of them: several are multi-GB or multi-TB benchmark corpora. The local download below focuses on the small, directly useful benchmark repository that works well with CSV-first Tollama and macOS app testing.

## Local Downloaded Data

Downloaded root:

```text
data/csv/hf_time_series/pkr7098_time-series-forecasting-datasets
```

Source dataset:
- https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets

Local size after refresh: about 592 MB.

| File | Format | Notes |
| --- | --- | --- |
| `ETTh1.csv` | CSV | ETT hourly benchmark; timestamp column `date`, target `OT`. |
| `ETTh2.csv` | CSV | ETT hourly benchmark; timestamp column `date`, target `OT`. |
| `ETTm1.csv` | CSV | ETT 15-minute benchmark; timestamp column `date`, target `OT`. |
| `ETTm2.csv` | CSV | ETT 15-minute benchmark; timestamp column `date`, target `OT`. |
| `electricity.csv` | CSV | Wide electricity benchmark; timestamp column `date`, target defaults well to `OT`. |
| `exchange_rate.csv` | CSV | Exchange-rate benchmark; timestamp column `date`, target `OT`. |
| `national_illness.csv` | CSV | Weekly illness benchmark; timestamp column `date`, target `OT`. |
| `traffic.csv` | CSV | Wide traffic benchmark; timestamp column `date`, target defaults well to `OT`. |
| `weather.csv` | CSV | Weather benchmark; timestamp column `date`, target `OT`. |
| `PEMS03.npz` | NumPy archive | Traffic sensor array; convert before using in CSV-only UI flows. |
| `PEMS04.npz` | NumPy archive | Traffic sensor array; convert before using in CSV-only UI flows. |
| `PEMS07.npz` | NumPy archive | Traffic sensor array; convert before using in CSV-only UI flows. |
| `PEMS08.npz` | NumPy archive | Traffic sensor array; convert before using in CSV-only UI flows. |
| `solar_AL.txt` | TXT | Solar benchmark text format. |
| `solar_AL.pkl` | Pickle | Solar benchmark serialized format. |
| `README.md` | Markdown | Upstream dataset notes. |
| `.gitattributes` | Text | Upstream repository metadata. |
| `prompt_bank/TimeVLM/*.txt` | TXT | Upstream prompt bank files: `ECL`, `ETT`, `Traffic`, `Weather`. |

The CSV files are the best first choices for Tollama and the macOS app. The `.npz`, `.pkl`, and raw `.txt` files are useful benchmark artifacts, but they usually need a conversion step before they can be previewed as regular app CSV inputs.

## Current High-Signal HF Datasets

Top datasets from the first 100 API results, sorted by downloads at refresh time:

| Dataset | Downloads | License | Why it matters |
| --- | ---: | --- | --- |
| `autogluon/chronos_datasets` | 26010 | other | Chronos training/evaluation corpus; many configs and parquet-backed files. |
| `autogluon/fev_datasets` | 16813 | other | Forecast evaluation datasets; useful for benchmark-style evaluation. |
| `Traders-Lab/TroveLedger` | 14673 | other | Finance/trading oriented time-series dataset. |
| `misikoff/zillow` | 4035 | other | Zillow-style real-estate time-series/regression data. |
| `Salesforce/GiftEval` | 3619 | apache-2.0 | Forecasting evaluation benchmark from Salesforce. |
| `phanerozoic/qiskit-calibration-drift` | 3337 | cc-by-4.0 | Calibration drift time-series data. |
| `autogluon/chronos_datasets_extra` | 2456 | apache-2.0 | Extra Chronos datasets. |
| `Monash-University/monash_tsf` | 2159 | cc-by-4.0 | Large canonical forecasting benchmark collection. |
| `thuml/UTSD` | 1705 | apache-2.0 | Large time-series pretraining dataset. |
| `Maple728/Time-300B` | 1585 | apache-2.0 | Very large time-series corpus; do not snapshot casually. |
| `commanderzee/1s-crypto-data` | 1330 | mit | High-frequency crypto time-series data. |
| `ChengsenWang/TSQA` | 1062 | apache-2.0 | Time-series QA / forecasting data. |
| `AutonLab/Timeseries-PILE` | 802 | mit | Broad time-series corpus. |
| `ibm-research/AssetOpsBench` | 697 | apache-2.0 | Asset operations benchmark with time-series tasks. |
| `Shashkovich/Telecommunication_SMS_time_series` | 374 | gpl-3.0 | Smaller telecom SMS time-series data. |

For quick local app experiments, prefer `pkr7098/time-series-forecasting-datasets`. For broader benchmark coverage, load individual configs from `Monash-University/monash_tsf`, `autogluon/chronos_datasets`, or `Salesforce/GiftEval` rather than downloading entire repositories.

## Recommended Datasets

### `pkr7098/time-series-forecasting-datasets`

Best for Tollama smoke tests and macOS app CSV import testing.

- Card: https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets
- License: MIT
- Last modified in HF API snapshot: 2026-02-26
- Local status: downloaded under `data/csv/hf_time_series/...`
- CSV target convention: most files expose `date` plus one or more numeric series and a final `OT` target.

### `Monash-University/monash_tsf`

Best for broad forecasting benchmark coverage.

- Card: https://huggingface.co/datasets/Monash-University/monash_tsf
- License: CC BY 4.0
- Last modified in HF API snapshot: 2023-06-13
- Size: large, about 24.5 GB reported by the dataset API
- Load one config at a time with `datasets`; do not snapshot the entire repository unless you really need it.

Common configs include `weather`, `tourism_monthly`, `tourism_quarterly`, `tourism_yearly`, `nn5_daily`, `nn5_weekly`, `kaggle_web_traffic_weekly`, `traffic_hourly`, `traffic_weekly`, `hospital`, `covid_deaths`, `sunspot`, and `temperature_rain`.

### `ETDataset/ett`

Best when you want the canonical ETT dataset entry rather than the CSV mirror.

- Card: https://huggingface.co/datasets/ETDataset/ett
- License: CC BY 4.0
- Configs: `h1`, `h2`, `m1`, `m2`
- Note: the same practical CSV files are already available locally through the `pkr7098` mirror.

### `tulipa762/electricity_load_diagrams`

Best for electricity load experiments through the Hugging Face `datasets` loader.

- Card: https://huggingface.co/datasets/tulipa762/electricity_load_diagrams
- Size reported by API: about 946 MB
- Note: the local `electricity.csv` file in the `pkr7098` mirror is usually simpler for CSV-first testing.

## Refresh Commands

Refresh the Hugging Face top-list metadata:

```bash
curl -sSL 'https://huggingface.co/api/datasets?filter=task_categories:time-series-forecasting&limit=100&full=true' \
  -o /tmp/hf_time_series_datasets.json

jq -r '.[] | [.id, (.downloads // 0), ((.cardData.license // .license // "unknown")|tostring)] | @tsv' \
  /tmp/hf_time_series_datasets.json \
  | sort -t $'\t' -k2,2nr \
  | head -20
```

Refresh the local `pkr7098` dataset with `huggingface_hub`:

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

local_dir = "data/csv/hf_time_series/pkr7098_time-series-forecasting-datasets"
snapshot_download(
    repo_id="pkr7098/time-series-forecasting-datasets",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=[
        "*.csv",
        "*.npz",
        "*.pkl",
        "*.txt",
        "README.md",
        "prompt_bank/**/*.txt",
    ],
)
print(f"Downloaded to: {local_dir}")
PY
```

Download a single raw file with resume support:

```bash
curl -L --fail --retry 3 -C - \
  -o data/csv/hf_time_series/pkr7098_time-series-forecasting-datasets/ETTm1.csv \
  https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/resolve/main/ETTm1.csv
```

Load a Monash config without downloading the whole repository:

```bash
python -m pip install -U datasets
python - <<'PY'
from datasets import load_dataset

ds = load_dataset(
    "Monash-University/monash_tsf",
    "nn5_daily",
    trust_remote_code=True,
)
print(ds)
print(ds["train"][0])
PY
```

## Tollama Usage Notes

- In the macOS app, point the CSV browser at `data/csv/hf_time_series`.
- Start with `ETTm1.csv`, `ETTm2.csv`, `ETTh1.csv`, or `exchange_rate.csv` for fast forecasting checks.
- For wide CSV files such as `electricity.csv` and `traffic.csv`, set the target column explicitly if the app does not infer the intended series.
- If a Hugging Face dataset is parquet, arrow, JSON, NPZ, pickle, or a dataset-script repo, convert it to the canonical CSV shape before using CSV-only flows: `timestamp,target` or `timestamp,series,target`.
- For missing timestamp handling, prefer unique timestamps in CSV exports. If duplicate timestamps exist, use Tollama's missing preprocessing path only when duplicate aggregation by timestamp is intended.

## Troubleshooting

- `401` or `403`: public datasets do not usually require auth, but gated datasets do. Run `huggingface-cli login` when needed.
- Slow or interrupted download: use `curl -C -` or `snapshot_download`, both of which can resume safely.
- Disk pressure: use `allow_patterns` and download only CSV/configs you need.
- App preview is slow: try a smaller CSV such as `ETTm1.csv` before opening `traffic.csv` or `electricity.csv`.
