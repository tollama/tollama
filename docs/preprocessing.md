# Preprocessing Pipeline

Built-in time-series preprocessing with schema validation, spline interpolation, smoothing,
leakage-safe train-fit scaling, and sliding window generation.

Install optional dependency:

```bash
python -m pip install "tollama[preprocess]"
```

## Usage

Standalone usage (no daemon required):

```python
import numpy as np
from tollama.preprocess import run_pipeline, PreprocessConfig

x = np.arange(200, dtype=float)
y = np.sin(x * 0.05) * 10 + np.random.randn(200) * 0.5

result = run_pipeline(x, y, config=PreprocessConfig(lookback=12, horizon=6))
print(result.X.shape, result.y.shape)  # [batch, 12, 1], [batch, 6]
```

With `SeriesInput` integration:

```python
from tollama.preprocess.bridge import preprocess_series_input

result = preprocess_series_input(series_input, config=PreprocessConfig(horizon=7))
```

## Forecast Ingest Missing-Value Repair

CSV/Parquet forecast ingest keeps its default behavior unchanged: null target
rows are omitted unless missing preprocessing is explicitly enabled. For
`data_url` requests, enable opt-in target repair with:

```json
{
  "ingest": {
    "preprocessing": {
      "missing": {
        "enabled": true,
        "method": "auto",
        "max_missing_ratio": 0.3,
        "max_gap": 24,
        "edge_strategy": "nearest"
      }
    }
  }
}
```

Supported methods are `auto`, `bspline`, `linear`, and `seasonal`. When enabled,
ingest resolves the series frequency, builds a regular timestamp grid, treats
null targets and missing timestamps as gaps, validates missing-ratio/gap limits,
and returns preprocessing diagnostics on forecast responses. B-spline uses the
optional `tollama[preprocess]` SciPy dependency; explicit `bspline` fails with a
dependency error when unavailable, while `auto` falls back to linear with a
warning.

## Pipeline Stages

| Stage           | Description                                                        |
| --------------- | ------------------------------------------------------------------ |
| **Validate**    | Monotonic timestamps, missing ratio, max gap, non-constant target  |
| **Interpolate** | Spline-based NaN filling (cubic/linear fallback)                   |
| **Smooth**      | Savitzky-Golay filter or P-spline passthrough                      |
| **Scale**       | Train-fit only (standard or min-max) to prevent data leakage       |
| **Window**      | Sliding windows `[batch, lookback, features]` + `[batch, horizon]` |

Individual components (`SplinePreprocessor`, `StandardScaler1D`, `MinMaxScaler1D`,
`make_windows`, `chronological_split`, etc.) are also usable independently.

Implementation: `src/tollama/preprocess/`.
