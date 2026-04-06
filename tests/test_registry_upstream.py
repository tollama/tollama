"""Optional CI guard for validating registry entries against Hugging Face metadata."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pytest

from tollama.core.registry import load_registry

_REMOTE_VALIDATION_ENV = "TOLLAMA_VALIDATE_REGISTRY_REMOTE"
_HF_TOKEN_ENV_NAMES = (
    "TOLLAMA_HF_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


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
    token = _hf_token_from_env()
    failures: list[_ValidationFailure] = []
    inaccessible: list[_ValidationFailure] = []
    specs = sorted(load_registry().values(), key=lambda item: item.name)

    for spec in specs:
        if spec.source.type != "huggingface":
            continue

        info, error_message = _fetch_model_info_with_retries(
            api=api,
            repo_id=spec.source.repo_id,
            revision=spec.source.revision,
            token=token,
        )
        if info is None:
            issue = _ValidationFailure(
                model_name=spec.name,
                repo_id=spec.source.repo_id,
                reason=f"model_info lookup failed ({error_message})",
            )
            if _is_inaccessible_repo_error(error_message):
                inaccessible.append(issue)
            else:
                failures.append(issue)
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

    if failures:
        message = _format_failures(failures)
        if inaccessible:
            inaccessible_message = _format_failures(
                inaccessible,
                heading="registry upstream validation skipped inaccessible repos:",
            )
            message = (
                f"{message}\n\n"
                f"{inaccessible_message}"
            )
        pytest.fail(message)

    if inaccessible:
        pytest.skip(
            _format_failures(
                inaccessible,
                heading="registry upstream validation skipped inaccessible repos:",
            )
        )


def _fetch_model_info_with_retries(
    *,
    api: Any,
    repo_id: str,
    revision: str,
    token: str | None = None,
    attempts: int = 3,
) -> tuple[Any | None, str | None]:
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        try:
            info = api.model_info(repo_id=repo_id, revision=revision, token=token)
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


def _hf_token_from_env() -> str | None:
    for env_name in _HF_TOKEN_ENV_NAMES:
        token = _nonempty_str(os.environ.get(env_name))
        if token is not None:
            return token
    return None


def _is_inaccessible_repo_error(error_message: str | None) -> bool:
    if error_message is None:
        return False

    normalized = error_message.lower()
    return any(
        marker in normalized
        for marker in (
            "401 client error",
            "403 client error",
            "private or gated repo",
            "gated repo",
            "authentication",
            "invalid username or password",
        )
    )


def _format_failures(
    failures: list[_ValidationFailure],
    *,
    heading: str = "registry upstream validation failures:",
) -> str:
    lines = [heading]
    for failure in failures:
        lines.append(f"{failure.model_name} ({failure.repo_id}): {failure.reason}")
    return "\n".join(lines)


def test_inaccessible_repo_error_classifies_auth_failures() -> None:
    error_message = (
        "RepositoryNotFoundError: 401 Client Error. "
        "If you are trying to access a private or gated repo, make sure you are authenticated. "
        "Invalid username or password."
    )

    assert _is_inaccessible_repo_error(error_message) is True


def test_inaccessible_repo_error_does_not_hide_true_not_found() -> None:
    error_message = (
        "RepositoryNotFoundError: 404 Client Error. "
        "Repository Not Found for url: https://huggingface.co/api/models/example/revision/main."
    )

    assert _is_inaccessible_repo_error(error_message) is False
