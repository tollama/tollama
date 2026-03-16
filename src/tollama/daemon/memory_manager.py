"""GPU memory pressure monitoring and LRU-based automatic model unloading."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VRAM_PRESSURE_THRESHOLD = 0.85  # 85%
DEFAULT_POLL_INTERVAL_SECONDS = 30.0

# ---------------------------------------------------------------------------
# GPU memory probing — works without torch installed (returns None gracefully)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUMemoryInfo:
    """Snapshot of a single GPU device's memory."""

    device_index: int
    device_name: str
    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int

    @property
    def utilization(self) -> float:
        """Fraction of total memory currently allocated (0.0–1.0)."""
        if self.total_bytes <= 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes


def probe_gpu_memory() -> list[GPUMemoryInfo]:
    """Probe CUDA GPU memory via torch.cuda if available.

    Returns an empty list when CUDA is unavailable or torch is not installed.
    """
    try:
        import torch  # type: ignore[import-untyped]

        if not torch.cuda.is_available():
            return []

        result: list[GPUMemoryInfo] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            result.append(
                GPUMemoryInfo(
                    device_index=i,
                    device_name=props.name,
                    total_bytes=props.total_mem,
                    allocated_bytes=allocated,
                    reserved_bytes=reserved,
                )
            )
        return result
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Family-level last-used tracking for LRU eviction
# ---------------------------------------------------------------------------


@dataclass
class _FamilyUsage:
    family: str
    last_used_at: float = field(default_factory=time.monotonic)


class MemoryPressureManager:
    """Monitor GPU memory and trigger LRU unloads when pressure exceeds threshold.

    This manager is designed to be called periodically (e.g. from a background
    thread or after each forecast request) rather than running its own thread
    to keep the implementation simple and testable.
    """

    def __init__(
        self,
        *,
        pressure_threshold: float = DEFAULT_VRAM_PRESSURE_THRESHOLD,
    ) -> None:
        self._pressure_threshold = max(0.0, min(pressure_threshold, 1.0))
        self._family_usage: dict[str, _FamilyUsage] = {}
        self._lock = threading.Lock()

    def record_usage(self, family: str) -> None:
        """Record that a model family was just used (updates LRU order)."""
        with self._lock:
            usage = self._family_usage.get(family)
            if usage is None:
                self._family_usage[family] = _FamilyUsage(family=family)
            else:
                usage.last_used_at = time.monotonic()

    def remove_family(self, family: str) -> None:
        """Remove a family from tracking (after unload)."""
        with self._lock:
            self._family_usage.pop(family, None)

    def families_to_evict(self) -> list[str]:
        """Return families that should be unloaded to relieve memory pressure.

        Checks GPU utilization against the threshold. If pressure is high,
        returns the least-recently-used families (oldest first), stopping
        once only one family remains (never evict the last active family).
        """
        snapshots = probe_gpu_memory()
        if not snapshots:
            return []

        max_utilization = max(s.utilization for s in snapshots)
        if max_utilization < self._pressure_threshold:
            return []

        with self._lock:
            sorted_families = sorted(
                self._family_usage.values(),
                key=lambda u: u.last_used_at,
            )

        if len(sorted_families) <= 1:
            return []

        # Evict oldest-first, but keep at least the most-recently-used family.
        return [u.family for u in sorted_families[:-1]]

    def get_status(self) -> dict[str, Any]:
        """Return memory pressure status for diagnostics."""
        snapshots = probe_gpu_memory()
        gpu_info = [
            {
                "device": s.device_index,
                "name": s.device_name,
                "total_mb": round(s.total_bytes / (1024 * 1024), 1),
                "allocated_mb": round(s.allocated_bytes / (1024 * 1024), 1),
                "utilization": round(s.utilization, 3),
            }
            for s in snapshots
        ]

        with self._lock:
            tracked = {
                u.family: round(time.monotonic() - u.last_used_at, 1)
                for u in self._family_usage.values()
            }

        return {
            "pressure_threshold": self._pressure_threshold,
            "gpu_devices": gpu_info,
            "tracked_families": tracked,
        }
