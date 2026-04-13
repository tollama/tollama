"""Resolve the optional runner extra required for a registry model.

This helper intentionally stays stdlib-only because the GitHub Actions
workflow uses it before installing the project's runtime dependencies.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "model-registry" / "registry.yaml"

# Keep this mapping aligned with ``tollama.core.runtime_bootstrap.FAMILY_EXTRAS``.
_FAMILY_EXTRAS: dict[str, str] = {
    "torch": "runner_torch",
    "timesfm": "runner_timesfm",
    "uni2ts": "runner_uni2ts",
    "sundial": "runner_sundial",
    "toto": "runner_toto",
    "lag_llama": "runner_lag_llama",
    "patchtst": "runner_patchtst",
    "tide": "runner_tide",
    "nhits": "runner_nhits",
    "nbeatsx": "runner_nbeatsx",
    "timer": "runner_timer",
    "timemixer": "runner_timemixer",
    "forecastpfn": "runner_forecastpfn",
}


def _parse_scalar(value: str) -> str:
    parsed = value.strip()
    if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {'"', "'"}:
        return parsed[1:-1]
    return parsed


@lru_cache(maxsize=1)
def _load_registry_entries() -> dict[str, tuple[str | None, str | None]]:
    entries: dict[str, tuple[str | None, str | None]] = {}
    current_name: str | None = None
    current_family: str | None = None
    current_install_extra: str | None = None

    for raw_line in _REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()

        if raw_line.startswith("  - name: "):
            if current_name is not None:
                entries[current_name] = (current_family, current_install_extra)
            current_name = _parse_scalar(stripped.removeprefix("- name: "))
            current_family = None
            current_install_extra = None
            continue

        if current_name is None:
            continue

        if raw_line.startswith("    family: "):
            current_family = _parse_scalar(stripped.removeprefix("family: "))
            continue

        if raw_line.startswith("      install_extra: "):
            current_install_extra = _parse_scalar(stripped.removeprefix("install_extra: "))

    if current_name is not None:
        entries[current_name] = (current_family, current_install_extra)

    return entries


def resolve_runner_extra(model: str) -> str:
    """Return the optional dependency extra required for *model*'s runner."""
    family, install_extra = _load_registry_entries().get(model, (None, None))
    if family is None and install_extra is None:
        raise ValueError(f"unknown registry model {model!r}")

    if install_extra:
        return install_extra

    extra = _FAMILY_EXTRAS.get(family or "")
    if extra is not None:
        return extra

    raise ValueError(f"model {model!r} (family={family!r}) does not declare a runner extra")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the optional runner extra needed for a model."
    )
    parser.add_argument("--model", required=True, help="Registry model name.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(resolve_runner_extra(args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
