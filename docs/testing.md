# End-to-End Testing

## Local Model E2E (All Registered Models)

Use the local helper script to run pull + forecast smoke checks across all
registered models with model-appropriate payloads:

```bash
export PATH="$PWD/.venv_langchain_e2e/bin:$PATH"
scripts/run_all_models_e2e_local.sh
```

Notes:

- The helper auto-generates long-context payloads for families that need long
  history windows (for example Granite TTM, TiDE, N-HiTS, N-BEATSx).
- PatchTST is executed with a dedicated compatible payload/horizon in this
  smoke flow.

For the per-family suite plus full-model smoke in one command:

```bash
export PATH="$PWD/.venv_langchain_e2e/bin:$PATH"
bash scripts/e2e_all_families.sh
```

## Real-Data E2E (7 TSFMs + 4 Neural Baselines)

Run the real-data gate + benchmark harness locally:

```bash
# PR-like smoke mode (1 sample per dataset, open-data fallback if Kaggle creds are missing)
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode pr \
  --model all \
  --gate-profile strict \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/local-pr

# Nightly-like mode (requires KAGGLE_USERNAME/KAGGLE_KEY)
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode nightly \
  --model all \
  --gate-profile strict \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/local-nightly

# Local mode without Kaggle credentials (explicit fallback)
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model all \
  --gate-profile strict \
  --allow-kaggle-fallback \
  --base-url http://127.0.0.1:11435 \
  --output-dir artifacts/realdata/local-open-fallback
```

Wrapper script:

```bash
bash scripts/e2e_realdata_tsfm.sh pr all http://127.0.0.1:11435 artifacts/realdata/wrapper false
# 5th arg=true enables explicit local fallback when Kaggle credentials are missing
```

Artifacts include `result.json`, `summary.json`, `summary.md`, and raw per-call payloads.
`result.json` entries now carry `status` (`pass`/`fail`/`skip`) and `retry_count`.

## HuggingFace Datasets (Optional Local Benchmark)

HuggingFace datasets are kept as an optional local/manual path and are not part of PR/nightly strict CI gating.
First gather a deterministic catalog:

```bash
# Requires `datasets`. Produces both accepted catalog and rejection reasons.
python scripts/e2e_realdata/gather_hf_datasets.py \
  --output scripts/e2e_realdata/hf_dataset_catalog.yaml \
  --rejections-output scripts/e2e_realdata/hf_dataset_rejections.json
```

Then run HF local evaluation with optional gate profile:

```bash
python scripts/e2e_realdata/run_tsfm_realdata.py \
  --mode local \
  --model all \
  --catalog-path scripts/e2e_realdata/hf_dataset_catalog.yaml \
  --gate-profile hf_optional \
  --allow-kaggle-fallback \
  --output-dir artifacts/realdata/hf-local
```

Convenience wrapper:

```bash
bash scripts/e2e_realdata_hf.sh all http://127.0.0.1:11435 artifacts/realdata/hf-local
```
