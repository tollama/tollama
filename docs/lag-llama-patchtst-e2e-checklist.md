# Lag-Llama + PatchTST E2E Verification Checklist

Use this checklist when validating the `lag_llama` and `patchtst` runner integrations before merge/release.

## 0) Environment setup

- [ ] Python environment activated
- [ ] Install runtime extras:

```bash
python -m pip install -e ".[dev,runner_lag_llama,runner_patchtst]"
```

- [ ] Start daemon:

```bash
tollama serve
```

- [ ] Pull model manifests/snapshots:

```bash
tollama pull lag-llama
tollama pull patchtst
```

## 1) Success path smoke checks

- [ ] Lag-Llama forecast succeeds:

```bash
tollama run lag-llama --input examples/lag_llama_request.json --no-stream
```

- [ ] PatchTST forecast succeeds:

```bash
tollama run patchtst --input examples/request.json --no-stream
```

Expected: non-empty `forecasts[]` with `mean` values and no protocol/runner crash.

## 2) Expected failure modes (must be explicit)

### A. Missing optional runner dependency

Run from a clean env without the corresponding extra installed.

- [ ] Lag-Llama returns install hint mentioning `runner_lag_llama`
- [ ] PatchTST returns install hint mentioning `runner_patchtst`

Expected behavior:
- Runner protocol error code: `DEPENDENCY_MISSING`
- Daemon HTTP surface: request fails with install guidance (do **not** silently fallback).

### B. Invalid payload shape rejected before model invocation

- [ ] `model_local_dir` non-string is rejected with `BAD_REQUEST`
- [ ] `model_source` non-object is rejected with `BAD_REQUEST`

Expected behavior:
- Validation error is returned before dependency/model execution path.

### C. Unsupported covariate/static features

- [ ] In `best_effort`, unsupported covariates are ignored (warning allowed)
- [ ] In `strict`, unsupported covariates are rejected (`400`)

## 3) Regression tests to run (targeted)

```bash
PYTHONPATH=src pytest -q tests/test_lag_llama_runner.py tests/test_patchtst_runner.py tests/test_runner_manager.py
```

## 4) Optional scripted check

A helper script is available:

```bash
bash scripts/e2e_lag_llama_patchtst_check.sh
```

Use `--with-success` if runtime deps/models are installed and pulled.
