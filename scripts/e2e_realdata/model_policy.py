"""Shared model/scenario policy for real-data E2E harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BENCHMARK_TARGET_ONLY = "benchmark_target_only"
CONTRACT_BEST_EFFORT = "contract_best_effort_covariates"
CONTRACT_STRICT = "contract_strict_covariates"
SCENARIOS = [
    BENCHMARK_TARGET_ONLY,
    CONTRACT_BEST_EFFORT,
    CONTRACT_STRICT,
]
BENCHMARK_METRIC_NAMES = ["mae", "rmse", "smape", "mape", "mase"]
PRIMARY_RANKING_METRIC = "smape"


@dataclass(frozen=True)
class ModelPolicy:
    """Harness-level model expectations for HF real-data runs."""

    name: str
    context_cap: int
    options: dict[str, Any]
    benchmark_enabled: bool = True
    contract_enabled: bool = False
    strict_expected_status: int | None = None
    best_effort_warning_required: bool = False


STRICT_TSFM_MODELS = [
    "chronos2",
    "granite-ttm-r2",
    "timesfm-2.5-200m",
    "moirai-2.0-R-small",
    "sundial-base-128m",
    "toto-open-base-1.0",
]

NEURAL_BASELINE_MODELS = [
    "lag-llama",
    "patchtst",
    "tide",
    "nhits",
    "nbeatsx",
]

HF_STARTER_MODELS = [*STRICT_TSFM_MODELS, *NEURAL_BASELINE_MODELS]

MODEL_POLICIES: dict[str, ModelPolicy] = {
    "chronos2": ModelPolicy(
        name="chronos2",
        context_cap=1024,
        options={},
        contract_enabled=True,
        strict_expected_status=200,
    ),
    "granite-ttm-r2": ModelPolicy(
        name="granite-ttm-r2",
        context_cap=512,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "timesfm-2.5-200m": ModelPolicy(
        name="timesfm-2.5-200m",
        context_cap=1024,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "moirai-2.0-R-small": ModelPolicy(
        name="moirai-2.0-R-small",
        context_cap=512,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "sundial-base-128m": ModelPolicy(
        name="sundial-base-128m",
        context_cap=2048,
        options={"num_samples": 100},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "toto-open-base-1.0": ModelPolicy(
        name="toto-open-base-1.0",
        context_cap=2048,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "lag-llama": ModelPolicy(name="lag-llama", context_cap=1024, options={}),
    "patchtst": ModelPolicy(name="patchtst", context_cap=512, options={}),
    "tide": ModelPolicy(name="tide", context_cap=512, options={}),
    "nhits": ModelPolicy(
        name="nhits",
        context_cap=512,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
    "nbeatsx": ModelPolicy(
        name="nbeatsx",
        context_cap=512,
        options={},
        contract_enabled=True,
        strict_expected_status=400,
        best_effort_warning_required=True,
    ),
}

MODEL_ALIASES: dict[str, list[str]] = {
    "all": STRICT_TSFM_MODELS,
    "hf_all": HF_STARTER_MODELS,
    "neural": NEURAL_BASELINE_MODELS,
}


def resolve_models(raw: str) -> list[str]:
    """Resolve one alias or a comma-separated explicit model list."""
    normalized = raw.strip().lower()
    if normalized in MODEL_ALIASES:
        return list(MODEL_ALIASES[normalized])

    selected = [item.strip() for item in raw.split(",") if item.strip()]
    if not selected:
        raise ValueError(
            "--model must be a model name, comma-separated list, or one of: "
            f"{', '.join(sorted(MODEL_ALIASES))}"
        )

    unknown = [item for item in selected if item not in MODEL_POLICIES]
    if unknown:
        raise ValueError(f"unsupported model(s): {', '.join(unknown)}")
    return selected


def get_policy(model: str) -> ModelPolicy:
    """Return the policy for one supported model."""
    try:
        return MODEL_POLICIES[model]
    except KeyError as exc:
        raise ValueError(f"unsupported model: {model!r}") from exc


def context_cap_for_model(model: str) -> int:
    """Return the harness-level context cap for one model."""
    return get_policy(model).context_cap


def model_options(model: str) -> dict[str, Any]:
    """Return a copy of harness-level runtime options for one model."""
    return dict(get_policy(model).options)


def scenarios_for_model(model: str) -> list[str]:
    """Return the enabled scenario list for one model."""
    policy = get_policy(model)
    scenarios = [BENCHMARK_TARGET_ONLY]
    if policy.contract_enabled:
        scenarios.extend([CONTRACT_BEST_EFFORT, CONTRACT_STRICT])
    return scenarios


def scenario_enabled(model: str, scenario: str) -> bool:
    """Return whether one scenario is enabled for the model."""
    return scenario in scenarios_for_model(model)


def strict_expected_status_for_model(model: str) -> int:
    """Return the expected strict covariates status for one model."""
    policy = get_policy(model)
    if policy.strict_expected_status is None:
        raise ValueError(f"unsupported model for strict status: {model!r}")
    return policy.strict_expected_status


def best_effort_warning_required(model: str) -> bool:
    """Return whether best-effort covariates should emit warnings."""
    return get_policy(model).best_effort_warning_required


def scenario_policy_summary() -> dict[str, dict[str, list[str]]]:
    """Return a JSON-friendly summary of scenario participation."""
    return {
        BENCHMARK_TARGET_ONLY: {"models": list(HF_STARTER_MODELS)},
        CONTRACT_BEST_EFFORT: {
            "models": [
                model
                for model in HF_STARTER_MODELS
                if scenario_enabled(model, CONTRACT_BEST_EFFORT)
            ]
        },
        CONTRACT_STRICT: {
            "models": [
                model for model in HF_STARTER_MODELS if scenario_enabled(model, CONTRACT_STRICT)
            ]
        },
    }
