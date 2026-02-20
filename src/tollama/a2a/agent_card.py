"""Helpers for constructing A2A Agent Card payloads."""

from __future__ import annotations

from dataclasses import dataclass

A2A_PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class AgentCardContext:
    """Inputs required to construct one Agent Card payload."""

    interface_url: str
    version: str
    require_authentication: bool
    installed_models: tuple[str, ...]
    available_models: tuple[str, ...]


def build_agent_card(context: AgentCardContext) -> dict[str, object]:
    """Build a latest-spec A2A Agent Card document."""
    installed_models = tuple(sorted({item for item in context.installed_models if item}))
    available_models = tuple(sorted({item for item in context.available_models if item}))
    model_example = _model_example(
        installed_models=installed_models,
        available_models=available_models,
    )

    skills: list[dict[str, object]] = [
        _skill(
            skill_id="analyze",
            name="Analyze Time Series",
            description=(
                "Diagnose frequency, seasonality, trend, anomalies, stationarity, and "
                "data quality before forecasting."
            ),
            tags=("analysis", "diagnostics", "timeseries"),
            examples=(
                "Analyze this historical series and summarize risks before forecasting.",
            ),
        ),
        _skill(
            skill_id="recommend",
            name="Recommend Model",
            description=(
                "Recommend compatible forecasting models using horizon and covariate constraints."
            ),
            tags=("recommendation", "model-selection", "timeseries"),
            examples=(
                "Recommend the best model for horizon 14 with known-future numeric covariates.",
            ),
        ),
        _skill(
            skill_id="generate",
            name="Generate Synthetic Series",
            description=(
                "Generate model-free synthetic time series from statistical profiles of "
                "historical input."
            ),
            tags=("generation", "simulation", "timeseries"),
            examples=(
                "Generate 5 synthetic weekly sales series using statistical method.",
            ),
        ),
    ]

    if installed_models:
        skills.extend(
            [
                _skill(
                    skill_id="forecast",
                    name="Forecast",
                    description="Generate forecasts for one model.",
                    tags=("forecast", "timeseries"),
                    examples=(
                        f"Forecast horizon 12 using model {model_example}.",
                    ),
                ),
                _skill(
                    skill_id="auto_forecast",
                    name="Auto Forecast",
                    description=(
                        "Run model auto-selection and forecasting with optional ensemble strategy."
                    ),
                    tags=("auto-forecast", "model-selection", "timeseries"),
                    examples=(
                        "Auto-forecast horizon 24 with fallback enabled.",
                    ),
                ),
                _skill(
                    skill_id="compare",
                    name="Compare Models",
                    description="Run side-by-side forecasts across multiple models.",
                    tags=("compare", "evaluation", "timeseries"),
                    examples=(
                        "Compare mock and chronos2 for horizon 7 on the same series.",
                    ),
                ),
                _skill(
                    skill_id="what_if",
                    name="What-If Scenarios",
                    description=(
                        "Apply structured scenario transforms and compare scenario "
                        "forecast outcomes."
                    ),
                    tags=("scenario", "simulation", "timeseries"),
                    examples=(
                        "Run a +10% demand scenario against the baseline forecast.",
                    ),
                ),
                _skill(
                    skill_id="pipeline",
                    name="Forecast Pipeline",
                    description=(
                        "Execute analyze -> recommend -> optional pull -> auto-forecast "
                        "in one task."
                    ),
                    tags=("pipeline", "automation", "timeseries"),
                    examples=(
                        "Run full pipeline for horizon 30 with pull_if_missing=true.",
                    ),
                ),
            ],
        )

    security_schemes: dict[str, dict[str, str]] = {}
    security_requirements: list[dict[str, list[str]]] = []
    if context.require_authentication:
        security_schemes = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "API Key",
            },
        }
        security_requirements = [{"bearerAuth": []}]

    return {
        "name": "tollama",
        "description": "Local-first time-series forecasting specialist",
        "supportedInterfaces": [
            {
                "url": context.interface_url,
                "protocolBinding": "JSONRPC",
                "protocolVersion": A2A_PROTOCOL_VERSION,
            }
        ],
        "provider": {
            "organization": "tollama",
            "url": "https://github.com/yongchoelchoi/tollama",
        },
        "version": context.version,
        "documentationUrl": "https://github.com/yongchoelchoi/tollama/blob/main/README.md",
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "securitySchemes": security_schemes,
        "securityRequirements": security_requirements,
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": skills,
    }


def _skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    tags: tuple[str, ...],
    examples: tuple[str, ...],
) -> dict[str, object]:
    return {
        "id": skill_id,
        "name": name,
        "description": description,
        "tags": list(tags),
        "examples": list(examples),
        "inputModes": ["application/json"],
        "outputModes": ["application/json"],
    }


def _model_example(*, installed_models: tuple[str, ...], available_models: tuple[str, ...]) -> str:
    if installed_models:
        return installed_models[0]
    if available_models:
        return available_models[0]
    return "mock"
