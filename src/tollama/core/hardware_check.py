"""Hardware requirements check before model pull.

Probes available GPU VRAM and compares against the model's ``min_vram_gb``
metadata from the registry. Returns warnings (not errors) so pulls are
never blocked — users with CPU-only setups can still download models.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default VRAM requirements (GB) per model family when not specified in registry.
# Conservative estimates for fp32 inference on the smallest variant.
_DEFAULT_MIN_VRAM_GB: dict[str, float] = {
    "torch": 4.0,
    "timesfm": 4.0,
    "uni2ts": 4.0,
    "sundial": 4.0,
    "toto": 8.0,
    "lag_llama": 4.0,
    "patchtst": 2.0,
    "tide": 2.0,
    "nhits": 2.0,
    "nbeatsx": 2.0,
    "timer": 4.0,
    "timemixer": 4.0,
    "forecastpfn": 2.0,
}


def probe_available_vram_gb() -> float | None:
    """Return total available VRAM in GB, or None if no GPU detected."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total_bytes = 0
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_bytes += props.total_mem
        return total_bytes / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def check_hardware_requirements(
    *,
    model_name: str,
    family: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Check if hardware meets model requirements.

    Returns a list of warning strings. Empty list means no issues detected.
    """
    warnings: list[str] = []

    # Determine min VRAM requirement
    min_vram_gb: float | None = None
    if metadata and "min_vram_gb" in metadata:
        try:
            min_vram_gb = float(metadata["min_vram_gb"])
        except (ValueError, TypeError):
            pass

    if min_vram_gb is None and family:
        min_vram_gb = _DEFAULT_MIN_VRAM_GB.get(family)

    if min_vram_gb is None:
        return warnings

    available = probe_available_vram_gb()
    if available is None:
        warnings.append(
            f"model {model_name!r} recommends {min_vram_gb:.1f}GB VRAM "
            f"but no GPU was detected. CPU inference may be slow."
        )
        return warnings

    if available < min_vram_gb:
        warnings.append(
            f"model {model_name!r} recommends {min_vram_gb:.1f}GB VRAM "
            f"but only {available:.1f}GB available. "
            f"Inference may fail or be slow."
        )

    return warnings
