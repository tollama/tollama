# Public Release Checklist

This checklist is focused on **open-source publication readiness** for `tollama`,
with extra attention on licensing and redistributability.

## 1) Repository baseline

- Confirm root package metadata is aligned:
  - `pyproject.toml` project license points to `LICENSE`.
  - root `LICENSE` file is present and accurate.
- Confirm docs and status trackers are in sync:
  - `README.md`
  - `docs/covariates.md`
  - `roadmap.md`
  - `todo-list.md`

## 2) License / compliance gate (required)

### 2.1 Registry model license verification

Run the optional upstream validation test with network access:

```bash
TOLLAMA_VALIDATE_REGISTRY_REMOTE=1 PYTHONPATH=src pytest -q tests/test_registry_upstream.py
```

This checks each Hugging Face registry entry against upstream metadata and flags
license mismatches.

### 2.2 High-risk model entry review

Current registry includes `moirai-2.0-R-small` with `cc-by-nc-4.0`
(`needs_acceptance: true`). This is **non-commercial** and can conflict with
commercial/public distribution goals.

Before public release, decide and document one policy:

1. Keep entry, but clearly mark it as non-commercial/research-only in release notes.
2. Move it to an optional "restricted" registry variant.
3. Remove it from the default public registry.

### 2.3 Third-party Python dependency license inventory

Generate a dependency license report in a clean environment:

```bash
python -m pip install pip-licenses
python -m pip install -e ".[dev]"
pip-licenses --format=markdown --with-authors --with-license-file > THIRD_PARTY_LICENSES.md
```

If publishing wheels/containers, keep this artifact in the release bundle.

### 2.4 Source provenance checks for non-PyPI pins

`runner_timesfm` currently uses a Git commit pin (`timesfm[torch] @ git+...`).
Before release:

- Verify upstream license at that exact commit.
- Capture commit hash + license in your release notes or NOTICE file.
- Re-test lock/pin reproducibility in CI.

## 3) Technical quality gate

Run baseline checks:

```bash
ruff check .
PYTHONPATH=src pytest -q
```

For optional runner families, run focused tests with corresponding extras
installed.

## 4) Operational and UX gate

- Validate `tollama info --remote` on a fresh machine.
- Validate auto-bootstrap runtime flow from a clean `TOLLAMA_HOME`.
- Confirm `/api/info` redaction and capability visibility are stable.
- Smoke test pull/list/show/run/ps/rm lifecycle end-to-end.

## 5) Release decision record

Before tagging public release, record:

- commit SHA
- checks run + outcomes
- known limitations/skips
- explicit licensing decision for restricted-license models

A short markdown file under `docs/releases/` is sufficient.
