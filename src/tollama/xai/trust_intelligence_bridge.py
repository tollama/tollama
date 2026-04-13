"""Bridge between tollama XAI structures and the trust_intelligence pipeline.

Adapts TrustIntelligencePipeline I/O to/from the existing ExplanationEngine
data structures and DecisionExplanation schemas.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:
    HAS_TRUST_INTELLIGENCE = True
except Exception:
    HAS_TRUST_INTELLIGENCE = False


def pipeline_result_to_metadata(result: Any) -> dict[str, Any]:
    """Convert a PipelineResult into a dict suitable for explanation metadata.

    Parameters
    ----------
    result : PipelineResult
        Output from TrustIntelligencePipeline.run().

    Returns
    -------
    dict
        Flattened summary for explanation metadata.
    """
    if not HAS_TRUST_INTELLIGENCE:
        return {}

    return {
        "trust_intelligence": {
            "version": result.pipeline_version,
            "trust_score": result.trust.trust_score,
            "calibration_status": result.trust.calibration_status,
            "weights": result.trust.weights,
            "components": {
                "uncertainty": result.uncertainty.normalized_uncertainty,
                "coverage_tightness": result.conformal.coverage_tightness,
                "coverage_validity": result.conformal.coverage_validity,
                "shap_stability": result.shap.shap_stability,
                "risk_category": result.constraints.risk_category.value,
                "constraint_satisfied": result.constraints.constraint_satisfied,
            },
            "shap_top_features": [
                {
                    "feature": fc.feature_name,
                    "shap_value": fc.shap_value,
                    "rank": fc.rank,
                    "direction": fc.direction,
                }
                for fc in result.shap.feature_contributions[:5]
            ],
            "violations": [
                {
                    "name": v.constraint_name,
                    "type": v.constraint_type,
                    "severity": v.severity,
                }
                for v in result.constraints.violations
            ],
            "meta_metrics": {
                "ece": result.trust.ece,
                "ocr": result.trust.ocr,
            },
        }
    }


def run_trust_pipeline(
    pipeline: Any,
    prediction_probability: float,
    features: dict[str, float] | None = None,
    context: dict[str, Any] | None = None,
    predict_fn: Callable | None = None,
) -> dict[str, Any] | None:
    """Run the trust intelligence pipeline and return metadata.

    Safe wrapper that returns None if the pipeline is unavailable or fails.

    Parameters
    ----------
    pipeline : TrustIntelligencePipeline or None
    prediction_probability : float
    features : dict, optional
    context : dict, optional
    predict_fn : callable, optional
        Model prediction function for faithful SHAP attribution.
        Signature: (features_array: np.ndarray) -> np.ndarray.
        When None, SHAP falls back to a built-in surrogate predictor.

    Returns
    -------
    dict or None
        Trust intelligence metadata, or None if unavailable.
    """
    if pipeline is None or not HAS_TRUST_INTELLIGENCE:
        return None

    try:
        result = pipeline.run(
            prediction_probability=prediction_probability,
            features=features,
            context=context,
            predict_fn=predict_fn,
        )
        return pipeline_result_to_metadata(result)
    except Exception:
        logger.exception("Trust intelligence pipeline failed")
        return None
