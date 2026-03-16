"""
tollama.xai.scenario_rationale — Scenario Rationale

v3.8: Scenario Rationale 설명 제품화
/api/scenario-tree, /api/what-if, /api/counterfactual 결과를
evidence-backed rationale로 변환.
"""

from __future__ import annotations

from typing import Any


class ScenarioRationale:
    """
    Converts scenario tree and counterfactual outputs into
    structured explanations.

    Answers: "Why were these scenarios generated? What drives
    the differences between base, upside, and downside cases?"
    """

    def explain(
        self,
        scenarios: dict[str, Any],
        forecast_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate scenario rationale from scenario tree output.

        Parameters
        ----------
        scenarios : dict
            Scenario tree output with branches (base, upside, downside, etc.)
        forecast_result : dict, optional
            Base forecast for context

        Returns
        -------
        dict with structured rationale per scenario
        """
        rationale = {
            "n_scenarios": 0,
            "scenarios": {},
            "key_drivers": [],
            "summary": "",
        }

        scenario_entries = scenarios.get("branches", scenarios)
        if isinstance(scenario_entries, dict):
            for name, details in scenario_entries.items():
                rationale["n_scenarios"] += 1
                rationale["scenarios"][name] = self._explain_scenario(
                    name, details, scenario_entries
                )
        elif isinstance(scenario_entries, list):
            for i, details in enumerate(scenario_entries):
                name = details.get("name", f"scenario_{i}")
                rationale["n_scenarios"] += 1
                rationale["scenarios"][name] = self._explain_scenario(
                    name, details, {}
                )

        # Key drivers across scenarios
        rationale["key_drivers"] = self._identify_key_drivers(
            rationale["scenarios"]
        )

        # Summary
        rationale["summary"] = self._generate_summary(rationale)

        return rationale

    def _explain_scenario(
        self,
        name: str,
        details: Any,
        all_scenarios: dict,
    ) -> dict[str, Any]:
        """Explain a single scenario branch."""
        if isinstance(details, (int, float)):
            return {
                "name": name,
                "probability": float(details),
                "conditions": [],
                "rationale": f"Scenario '{name}' with probability {details:.1%}",
            }

        if isinstance(details, dict):
            probability = details.get("probability", details.get("prob", 0))
            conditions = details.get("conditions", [])
            value = details.get("value", details.get("forecast", None))

            # Generate rationale based on scenario type
            if "base" in name.lower():
                scenario_type = "base"
                rationale = (
                    f"Base case ({probability:.0%} probability): "
                    f"continuation of current trend and conditions"
                )
            elif any(w in name.lower() for w in ["upside", "bull", "optimistic"]):
                scenario_type = "upside"
                rationale = (
                    f"Upside case ({probability:.0%} probability): "
                    f"favorable conditions materialize"
                )
            elif any(w in name.lower() for w in ["downside", "bear", "pessimistic"]):
                scenario_type = "downside"
                rationale = (
                    f"Downside case ({probability:.0%} probability): "
                    f"adverse conditions materialize"
                )
            else:
                scenario_type = "alternative"
                rationale = f"Alternative scenario ({probability:.0%} probability)"

            if conditions:
                condition_text = "; ".join(str(c) for c in conditions[:3])
                rationale += f". Key conditions: {condition_text}"

            return {
                "name": name,
                "type": scenario_type,
                "probability": float(probability) if probability else None,
                "value": value,
                "conditions": conditions,
                "rationale": rationale,
            }

        return {"name": name, "rationale": str(details)}

    def _identify_key_drivers(
        self, explained_scenarios: dict[str, dict]
    ) -> list[dict[str, str]]:
        """Identify key drivers that differentiate scenarios."""
        drivers = []

        # Collect all conditions
        all_conditions = set()
        for scenario in explained_scenarios.values():
            for cond in scenario.get("conditions", []):
                if isinstance(cond, str):
                    all_conditions.add(cond)

        for condition in list(all_conditions)[:5]:
            drivers.append({
                "driver": condition,
                "impact": "Differentiates scenario outcomes",
            })

        # Add probability spread driver
        probs = [
            s.get("probability", 0)
            for s in explained_scenarios.values()
            if s.get("probability") is not None
        ]
        if probs and max(probs) - min(probs) > 0.3:
            drivers.append({
                "driver": "High uncertainty spread",
                "impact": (
                    f"Probability range {min(probs):.0%}-{max(probs):.0%} "
                    f"indicates significant outcome uncertainty"
                ),
            })

        return drivers

    def _generate_summary(self, rationale: dict[str, Any]) -> str:
        """Generate natural language summary of scenario analysis."""
        n = rationale["n_scenarios"]
        if n == 0:
            return "No scenarios generated."

        summary = f"{n} scenarios analyzed. "

        # Find dominant scenario
        scenarios = rationale["scenarios"]
        highest_prob = 0
        dominant = ""
        for name, details in scenarios.items():
            prob = details.get("probability", 0) or 0
            if prob > highest_prob:
                highest_prob = prob
                dominant = name

        if dominant:
            summary += (
                f"Most likely: '{dominant}' ({highest_prob:.0%}). "
            )

        if rationale["key_drivers"]:
            driver = rationale["key_drivers"][0]["driver"]
            summary += f"Primary driver: {driver}."

        return summary
