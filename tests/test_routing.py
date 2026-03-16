"""Tests for benchmark-backed routing manifest helpers."""

from __future__ import annotations

import json

from tollama.core.config import TollamaConfig
from tollama.core.routing import (
    RoutingManifest,
    get_routing_manifest_path,
    load_routing_manifest,
    resolve_effective_routing_defaults,
    save_routing_manifest,
)
from tollama.core.storage import TollamaPaths


def _paths(monkeypatch, tmp_path) -> TollamaPaths:
    home = tmp_path / "routing-home"
    monkeypatch.setenv("TOLLAMA_HOME", str(home))
    return TollamaPaths.default()


def test_save_and_load_routing_manifest(monkeypatch, tmp_path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    manifest = RoutingManifest.model_validate(
        {
            "generated_at": "2026-03-16T00:00:00Z",
            "run_id": "20260316T000000Z",
            "source": "cross_model_tsfm",
            "routing": {
                "default": "chronos2",
                "fast_path": "timesfm-2.5-200m",
                "high_accuracy": "moirai-2.0-R-small",
            },
            "policy": "benchmark-backed defaults",
            "caveats": ["validate against production data"],
        }
    )

    saved_path = save_routing_manifest(manifest, paths=paths)
    loaded = load_routing_manifest(paths=paths)

    assert saved_path == get_routing_manifest_path(paths)
    assert loaded == manifest


def test_load_routing_manifest_accepts_benchmark_result_payload(monkeypatch, tmp_path) -> None:
    paths = _paths(monkeypatch, tmp_path)
    payload = {
        "run_id": "20260316T000000Z",
        "generated_at": "2026-03-16T00:00:00Z",
        "routing_recommendation": {
            "default": "chronos2",
            "fast_path": "timesfm-2.5-200m",
            "high_accuracy": "moirai-2.0-R-small",
            "policy": "use benchmark recommendation",
            "caveats": ["synthetic benchmark"],
        },
    }
    path = tmp_path / "benchmark-result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TOLLAMA_ROUTING_MANIFEST", str(path))

    loaded = load_routing_manifest(paths=paths)

    assert loaded is not None
    assert loaded.routing.default == "chronos2"
    assert loaded.routing.fast_path == "timesfm-2.5-200m"
    assert loaded.routing.high_accuracy == "moirai-2.0-R-small"


def test_resolve_effective_routing_defaults_prefers_config_over_manifest(
    monkeypatch,
    tmp_path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    save_routing_manifest(
        RoutingManifest.model_validate(
            {
                "routing": {
                    "default": "chronos2",
                    "fast_path": "timesfm-2.5-200m",
                    "high_accuracy": "moirai-2.0-R-small",
                }
            }
        ),
        paths=paths,
    )

    effective = resolve_effective_routing_defaults(
        config=TollamaConfig.model_validate(
            {
                "routing": {
                    "default": "granite-ttm-r2",
                    "fast_path": None,
                    "high_accuracy": "chronos2",
                }
            }
        ),
        paths=paths,
    )

    assert effective.default == "granite-ttm-r2"
    assert effective.fast_path == "timesfm-2.5-200m"
    assert effective.high_accuracy == "chronos2"
