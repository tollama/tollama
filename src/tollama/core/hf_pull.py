"""Hugging Face snapshot pull helpers with progress callbacks."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_THROTTLE_SECONDS = 0.2
_THROTTLE_BYTES = 1_000_000
_FILE_SCAN_INTERVAL_SECONDS = 0.75

ProgressCallback = Callable[[dict[str, Any]], None]


class PullError(RuntimeError):
    """Base pull error for user-facing failures."""


class OfflineModelUnavailableError(PullError):
    """Raised when offline/local-files-only pull cannot find a cached snapshot."""


@dataclass(frozen=True)
class _RepoMetadata:
    commit_sha: str
    total_bytes: int
    files_total: int


@dataclass(frozen=True)
class _HFClientFactoryTools:
    set_factory: Callable[[Any], None]
    default_factory: Any | None
    request_hook: Any | None
    backend: str


def pull_snapshot_to_local_dir(
    repo_id: str,
    revision: str,
    local_dir: str | Path,
    *,
    token: str | None = None,
    max_workers: int = 8,
    progress_cb: ProgressCallback | None = None,
    local_files_only: bool = False,
    offline: bool = False,
    insecure: bool = False,
    known_commit_sha: str | None = None,
    known_total_bytes: int = 0,
    known_files_total: int = 0,
) -> tuple[str, str, int]:
    """Download a model snapshot into a local directory."""
    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _emit(progress_cb, {"status": "pulling manifest"})

    if offline or local_files_only:
        metadata = _RepoMetadata(
            commit_sha=_commit_or_unknown(known_commit_sha),
            total_bytes=max(known_total_bytes, 0),
            files_total=max(known_files_total, 0),
        )
    else:
        metadata = _resolve_repo_metadata(
            repo_id=repo_id,
            revision=revision,
            token=token,
            fallback_commit_sha=known_commit_sha,
            fallback_total_bytes=known_total_bytes,
            fallback_files_total=known_files_total,
        )

    _emit(progress_cb, {"status": "resolving digest", "digest": metadata.commit_sha})
    if insecure:
        _emit(
            progress_cb,
            {
                "status": "warning",
                "message": "SSL verification disabled (--insecure). Use only for debugging.",
            },
        )

    tqdm_base = _tqdm_base_class()
    progress_tqdm_class = _build_progress_tqdm_class(
        tqdm_base=tqdm_base,
        progress_cb=progress_cb,
        local_dir=destination,
        total_bytes_hint=metadata.total_bytes,
        files_total_hint=metadata.files_total,
    )

    with _hf_client_factory_override(insecure=insecure):
        try:
            snapshot_path = _hf_snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(destination),
                token=token,
                max_workers=max_workers,
                tqdm_class=progress_tqdm_class,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            if local_files_only and _is_local_entry_not_found(exc):
                raise OfflineModelUnavailableError(
                    "Offline/local-files-only mode: model not found locally. "
                    "Re-run without --offline or --local-files-only to download.",
                ) from exc
            raise

    resolved_snapshot = str(Path(snapshot_path).resolve())
    size_bytes = _directory_size_bytes(Path(resolved_snapshot))
    return metadata.commit_sha, resolved_snapshot, size_bytes


def _resolve_repo_metadata(
    *,
    repo_id: str,
    revision: str,
    token: str | None,
    fallback_commit_sha: str | None,
    fallback_total_bytes: int,
    fallback_files_total: int,
) -> _RepoMetadata:
    try:
        info = _hf_model_info(repo_id=repo_id, revision=revision, token=token)
    except Exception:
        return _RepoMetadata(
            commit_sha=_commit_or_unknown(fallback_commit_sha),
            total_bytes=max(fallback_total_bytes, 0),
            files_total=max(fallback_files_total, 0),
        )

    commit_sha = _commit_or_unknown(getattr(info, "sha", None))
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, list):
        return _RepoMetadata(
            commit_sha=commit_sha,
            total_bytes=max(fallback_total_bytes, 0),
            files_total=max(fallback_files_total, 0),
        )

    files_total = len(siblings)
    total_bytes = 0
    for sibling in siblings:
        sibling_size = getattr(sibling, "size", None)
        if isinstance(sibling_size, int) and sibling_size > 0:
            total_bytes += sibling_size

    if total_bytes <= 0:
        total_bytes = max(fallback_total_bytes, 0)
    if files_total <= 0:
        files_total = max(fallback_files_total, 0)

    return _RepoMetadata(commit_sha=commit_sha, total_bytes=total_bytes, files_total=files_total)


def _build_progress_tqdm_class(
    *,
    tqdm_base: type[Any],
    progress_cb: ProgressCallback | None,
    local_dir: Path,
    total_bytes_hint: int,
    files_total_hint: int,
) -> type[Any]:
    class _ProgressTqdm(tqdm_base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._last_emit_ts = 0.0
            self._last_emit_bytes = 0
            self._last_file_scan_ts = 0.0
            self._last_file_count = 0

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

            completed_bytes = _to_nonnegative_int(getattr(self, "n", 0))
            total_bytes = _to_nonnegative_int(getattr(self, "total", total_bytes_hint))
            if total_bytes <= 0:
                total_bytes = max(total_bytes_hint, 0)

            now = time.monotonic()
            should_emit = force
            if not should_emit:
                progressed_enough = (completed_bytes - self._last_emit_bytes) >= _THROTTLE_BYTES
                waited_enough = (now - self._last_emit_ts) >= _THROTTLE_SECONDS
                should_emit = progressed_enough or waited_enough

            if not should_emit:
                return

            files_completed = self._last_file_count
            if force or (now - self._last_file_scan_ts) >= _FILE_SCAN_INTERVAL_SECONDS:
                self._last_file_scan_ts = now
                files_completed = _count_regular_files(local_dir)
                self._last_file_count = files_completed

            self._last_emit_ts = now
            self._last_emit_bytes = completed_bytes
            progress_cb(
                {
                    "status": "downloading",
                    "completed_bytes": completed_bytes,
                    "total_bytes": total_bytes,
                    "files_completed": files_completed,
                    "files_total": max(files_total_hint, 0),
                },
            )

    return _ProgressTqdm


def _count_regular_files(path: Path) -> int:
    count = 0
    if not path.exists():
        return count
    for file_path in path.rglob("*"):
        if ".cache" in file_path.parts:
            continue
        if file_path.is_file():
            count += 1
    return count


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


def _commit_or_unknown(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return "unknown"


def _emit(progress_cb: ProgressCallback | None, payload: dict[str, Any]) -> None:
    if progress_cb is None:
        return
    progress_cb(payload)


@contextmanager
def _hf_client_factory_override(*, insecure: bool) -> Iterator[None]:
    if not insecure:
        yield
        return

    tools = _hf_client_tools()
    if tools.backend == "httpx":
        request_hook = tools.request_hook

        def _insecure_client_factory() -> Any:
            import httpx

            event_hooks = {"request": [request_hook]} if request_hook is not None else None
            kwargs: dict[str, Any] = {
                "verify": False,
                "follow_redirects": True,
                "timeout": None,
            }
            if event_hooks is not None:
                kwargs["event_hooks"] = event_hooks
            return httpx.Client(**kwargs)

    else:

        def _insecure_client_factory() -> Any:
            import requests

            session = requests.Session()
            session.verify = False
            session.trust_env = True
            return session

    tools.set_factory(_insecure_client_factory)
    try:
        yield
    finally:
        if tools.default_factory is not None:
            tools.set_factory(tools.default_factory)


def _hf_client_tools() -> _HFClientFactoryTools:
    try:
        from huggingface_hub import set_client_factory
    except (ImportError, ModuleNotFoundError):
        return _hf_requests_client_tools()

    try:
        from huggingface_hub.utils._http import hf_request_event_hook
    except Exception:
        hf_request_event_hook = None

    default_client_factory: Any | None = None
    try:
        from huggingface_hub import default_client_factory as importable_default_client_factory

        default_client_factory = importable_default_client_factory
    except Exception:
        try:
            from huggingface_hub.utils._http import (
                default_client_factory as internal_default_client_factory,
            )

            default_client_factory = internal_default_client_factory
        except Exception:
            default_client_factory = None

    return _HFClientFactoryTools(
        set_factory=set_client_factory,
        default_factory=default_client_factory,
        request_hook=hf_request_event_hook,
        backend="httpx",
    )


def _hf_requests_client_tools() -> _HFClientFactoryTools:
    try:
        from huggingface_hub.utils._http import configure_http_backend
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing Hugging Face dependency; install with `pip install -e \".[dev]\"`",
        ) from exc

    default_client_factory: Any | None = None
    try:
        from huggingface_hub.utils._http import _default_backend_factory

        default_client_factory = _default_backend_factory
    except Exception:
        default_client_factory = None

    return _HFClientFactoryTools(
        set_factory=configure_http_backend,
        default_factory=default_client_factory,
        request_hook=None,
        backend="requests",
    )


def _is_local_entry_not_found(exc: Exception) -> bool:
    for current in _walk_exception_chain(exc):
        if current.__class__.__name__ == "LocalEntryNotFoundError":
            return True
        message = str(current).lower()
        if "local entry not found" in message:
            return True
        if "not found in local cache" in message:
            return True
    return False


def _walk_exception_chain(exc: Exception) -> Iterator[Exception]:
    seen: set[int] = set()
    current: Exception | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        if isinstance(current.__cause__, Exception):
            current = current.__cause__
            continue
        if isinstance(current.__context__, Exception):
            current = current.__context__
            continue
        current = None


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
    local_files_only: bool,
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
        local_files_only=local_files_only,
    )


def _tqdm_base_class() -> type[Any]:
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "missing tqdm dependency; install with `pip install -e \".[dev]\"`",
        ) from exc
    return tqdm
