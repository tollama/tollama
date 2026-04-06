"""Resolve the optional runner extra required for a registry model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _SRC_DIR = _REPO_ROOT / "src"
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

from tollama.core.registry import get_model_spec
from tollama.core.runtime_bootstrap import FAMILY_EXTRAS


def resolve_runner_extra(model: str) -> str:
    """Return the optional dependency extra required for *model*'s runner."""
    spec = get_model_spec(model)

    metadata = spec.metadata or {}
    install_extra = metadata.get("install_extra")
    if isinstance(install_extra, str) and install_extra.strip():
        return install_extra.strip()

    extra = FAMILY_EXTRAS.get(spec.family)
    if extra is not None:
        return extra

    raise ValueError(
        f"model {model!r} (family={spec.family!r}) does not declare a runner extra"
    )


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
