"""Optional CI guard for validating registry entries against Hugging Face metadata."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pytest

from tollama.core.registry import load_registry

_REMOTE_VALIDATION_ENV = "TOLLAMA_VALIDATE_REGISTRY_REMOTE"


@dataclass(frozen=True)
class _ValidationFailure:
    model_name: str
    repo_id: str
    reason: str


@pytest.mark.integration
def test_registry_huggingface_entries_resolve_and_match_declared_license() -> None:
    if os.environ.get(_REMOTE_VALIDATION_ENV) != "1":
        pytest.skip(f"set {_REMOTE_VALIDATION_ENV}=1 to run remote registry validation")

    pytest.importorskip("huggingface_hub")

    from huggingface_hub import HfApi

    api = HfApi()
    failures: list[_ValidationFailure] = []
    specs = sorted(load_registry().values(), key=lambda item: item.name)

    for spec in specs:
        if spec.source.type != "huggingface":
            continue

        info, error_message = _fetch_model_info_with_retries(
            api=api,
            repo_id=spec.source.repo_id,
            revision=spec.source.revision,
        )
        if info is None:
            failures.append(
                _ValidationFailure(
                    model_name=spec.name,
                    repo_id=spec.source.repo_id,
                    reason=f"model_info lookup failed ({error_message})",
                ),
            )
            continue

        upstream_license = _extract_upstream_license(info)
        if upstream_license is None:
            failures.append(
                _ValidationFailure(
                    model_name=spec.name,
                    repo_id=spec.source.repo_id,
                    reason="upstream license is missing from model metadata",
                ),
            )
            continue

        expected = _normalize_license(spec.license.type)
        observed = _normalize_license(upstream_license)
        if expected != observed:
            failures.append(
                _ValidationFailure(
                    model_name=spec.name,
                    repo_id=spec.source.repo_id,
                    reason=f"license mismatch (registry={expected}, upstream={observed})",
                ),
            )

    assert not failures, _format_failures(failures)


def _fetch_model_info_with_retries(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    attempts: int = 3,
) -> tuple[Any | None, str | None]:
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            info = api.model_info(repo_id=repo_id, revision=revision)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(0.5 * attempt)
            continue
        return info, None
    return None, last_error


def _extract_upstream_license(info: Any) -> str | None:
    direct = _nonempty_str(getattr(info, "license", None))
    if direct is not None:
        return direct

    card_data = getattr(info, "cardData", None)
    if isinstance(card_data, dict):
        card_license = _nonempty_str(card_data.get("license"))
        if card_license is not None:
            return card_license

    tags = getattr(info, "tags", None)
    if isinstance(tags, list):
        for item in tags:
            if not isinstance(item, str):
                continue
            if not item.startswith("license:"):
                continue
            tagged = _nonempty_str(item.split(":", 1)[1])
            if tagged is not None:
                return tagged
    return None


def _normalize_license(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def _format_failures(failures: list[_ValidationFailure]) -> str:
    lines = ["registry upstream validation failures:"]
    for failure in failures:
        lines.append(f"{failure.model_name} ({failure.repo_id}): {failure.reason}")
    return "\n".join(lines)
