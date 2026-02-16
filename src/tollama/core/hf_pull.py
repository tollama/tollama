"""Hugging Face snapshot pull helpers with progress callbacks."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_THROTTLE_SECONDS = 0.2
_THROTTLE_BYTES = 1_000_000

ProgressCallback = Callable[[dict[str, Any]], None]


def pull_snapshot_to_local_dir(
    repo_id: str,
    revision: str,
    local_dir: str | Path,
    *,
    token: str | None = None,
    max_workers: int = 8,
    progress_cb: ProgressCallback | None = None,
) -> tuple[str, str, int]:
    """Download a model snapshot into a local directory."""
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _emit(progress_cb, {"status": "pulling manifest"})
    commit_sha = _resolve_commit_sha(repo_id=repo_id, revision=revision, token=token)
    _emit(progress_cb, {"status": "resolving digest", "digest": commit_sha})

    tqdm_base = _tqdm_base_class()
    progress_tqdm_class = _build_progress_tqdm_class(tqdm_base=tqdm_base, progress_cb=progress_cb)
    snapshot_path = _hf_snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        token=token,
        max_workers=max_workers,
        tqdm_class=progress_tqdm_class,
    )

    resolved_snapshot = str(Path(snapshot_path).resolve())
    size_bytes = _directory_size_bytes(Path(resolved_snapshot))
    return commit_sha, resolved_snapshot, size_bytes


def _resolve_commit_sha(*, repo_id: str, revision: str, token: str | None) -> str:
    try:
        info = _hf_model_info(repo_id=repo_id, revision=revision, token=token)
    except Exception:
        return "unknown"

    sha = getattr(info, "sha", None)
    if isinstance(sha, str) and sha:
        return sha
    return "unknown"


def _build_progress_tqdm_class(
    *,
    tqdm_base: type[Any],
    progress_cb: ProgressCallback | None,
) -> type[Any]:
    class _ProgressTqdm(tqdm_base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._last_emit_ts = 0.0
            self._last_emit_bytes = 0

        def update(self, n: int = 1) -> Any:
            result = super().update(n)
            self._emit_progress(force=False)
            return result

        def close(self) -> None:
            self._emit_progress(force=True)
            super().close()

        def _emit_progress(self, *, force: bool) -> None:
            if progress_cb is None:
                return

            completed = _to_nonnegative_int(getattr(self, "n", 0))
            total = _to_nonnegative_int(getattr(self, "total", 0))
            now = time.monotonic()

            should_emit = force
            if not should_emit:
                progressed_enough = (completed - self._last_emit_bytes) >= _THROTTLE_BYTES
                waited_enough = (now - self._last_emit_ts) >= _THROTTLE_SECONDS
                should_emit = progressed_enough or waited_enough

            if not should_emit:
                return

            self._last_emit_ts = now
            self._last_emit_bytes = completed
            progress_cb(
                {
                    "status": "downloading",
                    "completed": completed,
                    "total": total,
                },
            )

    return _ProgressTqdm


def _to_nonnegative_int(value: Any) -> int:
    if isinstance(value, (int, float)):
        if value < 0:
            return 0
        return int(value)
    return 0


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total

    for file_path in path.rglob("*"):
        if ".cache" in file_path.parts:
            continue
        if not file_path.is_file():
            continue
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def _emit(progress_cb: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_cb is None:
        return
    progress_cb(payload)


def _hf_model_info(*, repo_id: str, revision: str, token: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing Hugging Face dependency; install with `pip install -e \".[dev]\"`",
        ) from exc

    client = HfApi(token=token)
    return client.model_info(repo_id=repo_id, revision=revision, token=token)


def _hf_snapshot_download(
    *,
    repo_id: str,
    revision: str,
    local_dir: str,
    token: str | None,
    max_workers: int,
    tqdm_class: type[Any],
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing Hugging Face dependency; install with `pip install -e \".[dev]\"`",
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        token=token,
        max_workers=max_workers,
        tqdm_class=tqdm_class,
    )


def _tqdm_base_class() -> type[Any]:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing tqdm dependency; install with `pip install -e \".[dev]\"`",
        ) from exc
    return tqdm
