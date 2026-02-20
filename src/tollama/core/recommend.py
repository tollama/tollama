"""Model recommendation helpers based on registry metadata and capabilities."""

from __future__ import annotations

from typing import Any, Literal

from .registry import ModelCapabilities, ModelSpec, list_registry_models

CovariatesType = Literal["numeric", "categorical"]


def recommend_models(
    *,
    horizon: int,
    freq: str | None = None,
    has_past_covariates: bool = False,
    has_future_covariates: bool = False,
    has_static_covariates: bool = False,
    covariates_type: CovariatesType = "numeric",
    allow_restricted_license: bool = False,
    top_k: int = 3,
    include_models: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Recommend models ranked by compatibility and simple heuristics."""
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    include_model_set: set[str] | None = None
    if include_models is not None:
        include_model_set = {
            item.strip()
            for item in include_models
            if isinstance(item, str) and item.strip()
        }
        if not include_model_set:
            raise ValueError("include_models must include at least one model name")

    registry = list_registry_models()

    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    considered_specs = 0
    for spec in registry:
        if include_model_set is not None and spec.name not in include_model_set:
            continue
        considered_specs += 1
        exclusion_reasons = _collect_exclusion_reasons(
            spec=spec,
            horizon=horizon,
            has_past_covariates=has_past_covariates,
            has_future_covariates=has_future_covariates,
            has_static_covariates=has_static_covariates,
            covariates_type=covariates_type,
            allow_restricted_license=allow_restricted_license,
        )
        if exclusion_reasons:
            excluded.append(
                {
                    "model": spec.name,
                    "family": spec.family,
                    "reasons": exclusion_reasons,
                },
            )
            continue

        score, reasons = _score_recommendation(
            spec=spec,
            horizon=horizon,
            freq=freq,
            has_past_covariates=has_past_covariates,
            has_future_covariates=has_future_covariates,
            has_static_covariates=has_static_covariates,
            covariates_type=covariates_type,
        )
        license_payload = {
            "type": spec.license.type,
            "needs_acceptance": spec.license.needs_acceptance,
        }
        if spec.license.notice:
            license_payload["notice"] = spec.license.notice

        candidates.append(
            {
                "model": spec.name,
                "family": spec.family,
                "license": license_payload,
                "score": score,
                "reasons": reasons,
            },
        )

    ranked = sorted(candidates, key=lambda item: (-int(item["score"]), str(item["model"])))
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    request_payload: dict[str, Any] = {
        "horizon": horizon,
        "freq": freq or "auto",
        "has_past_covariates": has_past_covariates,
        "has_future_covariates": has_future_covariates,
        "has_static_covariates": has_static_covariates,
        "covariates_type": covariates_type,
        "allow_restricted_license": allow_restricted_license,
        "top_k": top_k,
    }
    if include_model_set is not None:
        request_payload["include_models"] = sorted(include_model_set)

    return {
        "request": {
            **request_payload,
        },
        "recommendations": ranked[:top_k],
        "excluded": sorted(excluded, key=lambda item: str(item["model"])),
        "total_candidates": considered_specs,
        "compatible_candidates": len(ranked),
    }


def _collect_exclusion_reasons(
    *,
    spec: ModelSpec,
    horizon: int,
    has_past_covariates: bool,
    has_future_covariates: bool,
    has_static_covariates: bool,
    covariates_type: CovariatesType,
    allow_restricted_license: bool,
) -> list[str]:
    reasons: list[str] = []
    capabilities = spec.capabilities or ModelCapabilities()

    if _is_restricted_license(spec) and not allow_restricted_license:
        reasons.append("restricted_license")

    if has_past_covariates and not _supports_covariates(
        capabilities=capabilities,
        prefix="past",
        covariates_type=covariates_type,
    ):
        reasons.append(f"missing_past_covariates_{covariates_type}")

    if has_future_covariates and not _supports_covariates(
        capabilities=capabilities,
        prefix="future",
        covariates_type=covariates_type,
    ):
        reasons.append(f"missing_future_covariates_{covariates_type}")

    if has_static_covariates and not capabilities.static_covariates:
        reasons.append("missing_static_covariates")

    horizon_limit = _horizon_limit(spec)
    if horizon_limit is not None and horizon > horizon_limit:
        reasons.append(f"horizon_exceeds_limit:{horizon_limit}")

    return reasons


def _score_recommendation(
    *,
    spec: ModelSpec,
    horizon: int,
    freq: str | None,
    has_past_covariates: bool,
    has_future_covariates: bool,
    has_static_covariates: bool,
    covariates_type: CovariatesType,
) -> tuple[int, list[str]]:
    score = 100
    reasons: list[str] = ["matches required constraints"]

    if has_past_covariates:
        score += 10
        reasons.append(f"supports past {covariates_type} covariates")
    if has_future_covariates:
        score += 15
        reasons.append(f"supports known-future {covariates_type} covariates")
    if has_static_covariates:
        score += 10
        reasons.append("supports static covariates")

    horizon_limit = _horizon_limit(spec)
    if horizon_limit is None:
        score += 4
        reasons.append(f"no explicit horizon cap for requested horizon {horizon}")
    else:
        score += 8
        headroom = horizon_limit - horizon
        if headroom <= max(5, horizon_limit // 10):
            score += 4
        reasons.append(f"horizon {horizon} fits declared limit {horizon_limit}")

    if freq and freq.lower() != "auto":
        score += 1
        reasons.append(f"frequency hint provided ({freq})")

    if spec.license.needs_acceptance:
        score -= 2
    if spec.family == "mock":
        score -= 25
        reasons.append("mock runner deprioritized for production recommendations")

    return score, reasons


def _supports_covariates(
    *,
    capabilities: ModelCapabilities,
    prefix: Literal["past", "future"],
    covariates_type: CovariatesType,
) -> bool:
    if prefix == "past" and covariates_type == "numeric":
        return capabilities.past_covariates_numeric
    if prefix == "past" and covariates_type == "categorical":
        return capabilities.past_covariates_categorical
    if prefix == "future" and covariates_type == "numeric":
        return capabilities.future_covariates_numeric
    return capabilities.future_covariates_categorical


def _horizon_limit(spec: ModelSpec) -> int | None:
    metadata = spec.metadata
    if not isinstance(metadata, dict):
        return None
    for key in ("max_horizon", "prediction_length"):
        raw = metadata.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int) and raw > 0:
            return raw
        if isinstance(raw, float) and raw > 0 and raw.is_integer():
            return int(raw)
    return None


def _is_restricted_license(spec: ModelSpec) -> bool:
    license_type = spec.license.type.strip().lower()
    notice = (spec.license.notice or "").strip().lower()
    if spec.license.needs_acceptance:
        return True
    return "-nc" in license_type or "non-commercial" in notice
