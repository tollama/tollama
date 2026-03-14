"""Pluggable secrets manager integration for Tollama daemon.

Supports multiple backends for API key retrieval and rotation:
- env: Environment variables (default, zero-config)
- file: JSON/TOML config file
- vault: HashiCorp Vault (requires hvac)
- aws: AWS Secrets Manager (requires boto3)

Configure via TOLLAMA_SECRETS_BACKEND env var.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_ENV = "TOLLAMA_SECRETS_BACKEND"
_CACHE_TTL_ENV = "TOLLAMA_SECRETS_CACHE_TTL"
_DEFAULT_CACHE_TTL = 300  # 5 minutes


class SecretsBackend(ABC):
    """Abstract interface for secrets retrieval."""

    @abstractmethod
    def get_api_keys(self) -> list[str]:
        """Return the current list of valid API keys."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieve a named secret value."""


class EnvSecretsBackend(SecretsBackend):
    """Read secrets from environment variables (default backend)."""

    def get_api_keys(self) -> list[str]:
        raw = os.environ.get("TOLLAMA_API_KEYS", "").strip()
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]

    def get_secret(self, key: str) -> str | None:
        return os.environ.get(key)


class FileSecretsBackend(SecretsBackend):
    """Read secrets from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to secrets JSON file. Format::

            {
                "api_keys": ["key1", "key2"],
                "secrets": {"name": "value"}
            }
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._loaded_at: float = 0.0
        self._ttl = float(os.environ.get(_CACHE_TTL_ENV, _DEFAULT_CACHE_TTL))

    def _load(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._data and (now - self._loaded_at) < self._ttl:
            return self._data

        if not self._path.exists():
            logger.warning("Secrets file not found: %s", self._path)
            return {}

        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            self._loaded_at = now
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load secrets file: %s", exc)
            self._data = {}

        return self._data

    def get_api_keys(self) -> list[str]:
        data = self._load()
        keys = data.get("api_keys", [])
        return [k for k in keys if isinstance(k, str) and k.strip()]

    def get_secret(self, key: str) -> str | None:
        data = self._load()
        secrets_map = data.get("secrets", {})
        val = secrets_map.get(key)
        return str(val) if val is not None else None


class VaultSecretsBackend(SecretsBackend):
    """Read secrets from HashiCorp Vault.

    Parameters
    ----------
    url : str
        Vault server URL.
    token : str
        Vault authentication token.
    mount_point : str
        KV secrets engine mount point.
    path : str
        Secret path within the mount.
    """

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        mount_point: str = "secret",
        path: str = "tollama",
    ) -> None:
        self._url = url or os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200")
        self._token = token or os.environ.get("VAULT_TOKEN", "")
        self._mount_point = mount_point
        self._path = path
        self._cache: dict[str, Any] = {}
        self._cache_at: float = 0.0
        self._ttl = float(os.environ.get(_CACHE_TTL_ENV, _DEFAULT_CACHE_TTL))

    def _read(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._cache and (now - self._cache_at) < self._ttl:
            return self._cache

        try:
            import hvac
        except ImportError:
            raise ImportError(
                "hvac package required for Vault integration. "
                "Install with: pip install hvac"
            )

        client = hvac.Client(url=self._url, token=self._token)
        try:
            response = client.secrets.kv.v2.read_secret_version(
                path=self._path,
                mount_point=self._mount_point,
            )
            self._cache = response.get("data", {}).get("data", {})
            self._cache_at = now
        except Exception as exc:
            logger.error("Vault read failed: %s", exc)
            # Return stale cache if available
            if not self._cache:
                self._cache = {}

        return self._cache

    def get_api_keys(self) -> list[str]:
        data = self._read()
        keys = data.get("api_keys", "")
        if isinstance(keys, str):
            return [k.strip() for k in keys.split(",") if k.strip()]
        if isinstance(keys, list):
            return [str(k) for k in keys if k]
        return []

    def get_secret(self, key: str) -> str | None:
        data = self._read()
        val = data.get(key)
        return str(val) if val is not None else None


class AWSSecretsBackend(SecretsBackend):
    """Read secrets from AWS Secrets Manager.

    Parameters
    ----------
    secret_name : str
        Name of the AWS secret.
    region : str
        AWS region.
    """

    def __init__(
        self,
        secret_name: str | None = None,
        region: str | None = None,
    ) -> None:
        self._secret_name = secret_name or os.environ.get(
            "TOLLAMA_AWS_SECRET_NAME", "tollama/api-keys"
        )
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._cache: dict[str, Any] = {}
        self._cache_at: float = 0.0
        self._ttl = float(os.environ.get(_CACHE_TTL_ENV, _DEFAULT_CACHE_TTL))

    def _read(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._cache and (now - self._cache_at) < self._ttl:
            return self._cache

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package required for AWS Secrets Manager. "
                "Install with: pip install boto3"
            )

        try:
            client = boto3.client("secretsmanager", region_name=self._region)
            response = client.get_secret_value(SecretId=self._secret_name)
            secret_string = response.get("SecretString", "{}")
            self._cache = json.loads(secret_string)
            self._cache_at = now
        except Exception as exc:
            logger.error("AWS Secrets Manager read failed: %s", exc)
            if not self._cache:
                self._cache = {}

        return self._cache

    def get_api_keys(self) -> list[str]:
        data = self._read()
        keys = data.get("api_keys", "")
        if isinstance(keys, str):
            return [k.strip() for k in keys.split(",") if k.strip()]
        if isinstance(keys, list):
            return [str(k) for k in keys if k]
        return []

    def get_secret(self, key: str) -> str | None:
        data = self._read()
        val = data.get(key)
        return str(val) if val is not None else None


def create_secrets_backend(
    *,
    backend: str | None = None,
    config: dict[str, Any] | None = None,
) -> SecretsBackend:
    """Create a secrets backend from configuration.

    Parameters
    ----------
    backend : str, optional
        Backend type: ``"env"``, ``"file"``, ``"vault"``, ``"aws"``.
        Defaults to ``TOLLAMA_SECRETS_BACKEND`` env var or ``"env"``.
    config : dict, optional
        Backend-specific configuration.

    Returns
    -------
    SecretsBackend
    """
    backend_type = backend or os.environ.get(_BACKEND_ENV, "env").strip().lower()
    cfg = config or {}

    if backend_type == "env":
        return EnvSecretsBackend()

    if backend_type == "file":
        path = cfg.get("path") or os.environ.get("TOLLAMA_SECRETS_FILE", "")
        if not path:
            raise ValueError("File secrets backend requires 'path' config or TOLLAMA_SECRETS_FILE env var")
        return FileSecretsBackend(path)

    if backend_type == "vault":
        return VaultSecretsBackend(
            url=cfg.get("url"),
            token=cfg.get("token"),
            mount_point=cfg.get("mount_point", "secret"),
            path=cfg.get("path", "tollama"),
        )

    if backend_type == "aws":
        return AWSSecretsBackend(
            secret_name=cfg.get("secret_name"),
            region=cfg.get("region"),
        )

    raise ValueError(f"Unknown secrets backend: {backend_type!r}")
