"""Tests for model recommendation logic and wrappers."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from tollama.core.recommend import recommend_models
from tollama.mcp.tools import MCPToolError, tollama_recommend

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
if _HAS_LANGCHAIN:
    _HAS_LANGCHAIN = importlib.util.find_spec("langchain_core.tools") is not None


@pytest.fixture
def langchain_tools():
    if not _HAS_LANGCHAIN:
        pytest.skip("langchain_core.tools is not installed")
    return importlib.import_module("tollama.skill.langchain")


def test_recommend_models_prefers_categorical_covariate_support() -> None:
    payload = recommend_models(
        horizon=12,
        freq="D",
        has_past_covariates=True,
        has_future_covariates=True,
        covariates_type="categorical",
        top_k=3,
    )

    assert payload["recommendations"]
    assert payload["recommendations"][0]["model"] == "chronos2"
    assert all(
        "categorical" in " ".join(item["reasons"])
        for item in payload["recommendations"][:1]
    )


def test_recommend_models_excludes_restricted_license_by_default() -> None:
    payload = recommend_models(
        horizon=24,
        has_past_covariates=True,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=10,
    )

    excluded_models = {item["model"]: item["reasons"] for item in payload["excluded"]}
    assert "moirai-2.0-R-small" in excluded_models
    assert "restricted_license" in excluded_models["moirai-2.0-R-small"]


def test_recommend_models_allows_restricted_license_when_enabled() -> None:
    payload = recommend_models(
        horizon=24,
        has_past_covariates=True,
        has_future_covariates=True,
        covariates_type="numeric",
        allow_restricted_license=True,
        top_k=10,
    )

    recommended_models = [item["model"] for item in payload["recommendations"]]
    assert "moirai-2.0-R-small" in recommended_models


def test_recommend_models_rejects_non_positive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        recommend_models(horizon=0)


def test_recommend_models_include_models_filters_candidates() -> None:
    payload = recommend_models(
        horizon=12,
        include_models=["mock"],
        top_k=5,
    )

    assert payload["total_candidates"] == 1
    assert payload["request"]["include_models"] == ["mock"]
    assert all(item["model"] == "mock" for item in payload["recommendations"])


def test_mcp_tollama_recommend_invalid_request_maps_to_mcp_error() -> None:
    with pytest.raises(MCPToolError) as exc_info:
        tollama_recommend(horizon=12, top_k=0)

    assert exc_info.value.exit_code == 2
    assert exc_info.value.category == "INVALID_REQUEST"


def test_mcp_tollama_recommend_returns_ranked_payload() -> None:
    payload = tollama_recommend(
        horizon=48,
        freq="D",
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=3,
    )

    recommendations = payload["recommendations"]
    assert len(recommendations) <= 3
    assert recommendations
    assert recommendations[0]["rank"] == 1
    assert recommendations[0]["score"] >= recommendations[-1]["score"]


def test_langchain_recommend_tool_success(langchain_tools) -> None:
    tool = langchain_tools.TollamaRecommendTool()
    payload = tool._run(
        horizon=24,
        has_past_covariates=True,
        has_future_covariates=True,
        covariates_type="numeric",
        top_k=3,
    )

    assert "recommendations" in payload
    assert payload["recommendations"]
    assert payload["request"]["horizon"] == 24


def test_langchain_recommend_tool_invalid_request_maps_error(langchain_tools) -> None:
    tool = langchain_tools.TollamaRecommendTool()
    payload = tool._run(horizon=24, top_k=0)

    assert payload["error"]["category"] == "INVALID_REQUEST"
    assert payload["error"]["exit_code"] == 2
