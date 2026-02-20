"""tollama package."""

from typing import Any

__all__ = ["Tollama", "__version__"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "Tollama":
        from .sdk import Tollama

        return Tollama
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
