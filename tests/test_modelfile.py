"""Tests for TSModelfile parser/storage helpers."""

from __future__ import annotations

import pytest

from tollama.core.modelfile import (
    ModelfileUpsertRequest,
    TSModelfile,
    list_modelfiles,
    load_modelfile,
    merge_modelfile_defaults,
    remove_modelfile,
    write_modelfile,
)
from tollama.core.storage import TollamaPaths


def test_modelfile_write_and_load_roundtrip(tmp_path) -> None:
    paths = TollamaPaths(base_dir=tmp_path / ".tollama")
    profile = TSModelfile.model_validate(
        {
            "model": "mock",
            "horizon": 6,
            "quantiles": [0.1, 0.9],
            "options": {"seed": 7},
        }
    )

    stored = write_modelfile("baseline", profile, paths=paths)
    loaded = load_modelfile("baseline", paths=paths)
    listed = list_modelfiles(paths=paths)

    assert stored.name == "baseline"
    assert loaded.profile.model == "mock"
    assert loaded.profile.horizon == 6
    assert [item.name for item in listed] == ["baseline"]

    assert remove_modelfile("baseline", paths=paths) is True
    assert remove_modelfile("baseline", paths=paths) is False


def test_merge_modelfile_defaults_request_has_precedence() -> None:
    profile = TSModelfile.model_validate(
        {
            "model": "mock",
            "horizon": 12,
            "quantiles": [0.1, 0.9],
            "options": {"seed": 1, "temperature": 0.2},
            "parameters": {"covariates_mode": "strict"},
        }
    )
    request_payload = {
        "model": "mock",
        "horizon": 3,
        "quantiles": [],
        "options": {"seed": 7},
        "parameters": {"covariates_mode": "best_effort"},
    }

    merged = merge_modelfile_defaults(
        request_payload=request_payload,
        profile=profile,
        request_fields_set={"model", "horizon", "options", "parameters"},
    )

    assert merged["horizon"] == 3
    assert merged["options"] == {"seed": 7, "temperature": 0.2}
    assert merged["quantiles"] == [0.1, 0.9]
    assert merged["parameters"].covariates_mode == "best_effort"


def test_modelfile_upsert_request_requires_profile_or_content() -> None:
    request = ModelfileUpsertRequest.model_validate(
        {
            "name": "demo",
            "content": "model: mock\nhorizon: 3\n",
        }
    )
    assert request.resolved_profile().model == "mock"

    with pytest.raises(ValueError):
        ModelfileUpsertRequest.model_validate({"name": "demo"}).resolved_profile()
