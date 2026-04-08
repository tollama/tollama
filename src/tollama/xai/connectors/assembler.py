"""
tollama.xai.connectors.assembler — PayloadAssembler for building typed payloads from connectors.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tollama.xai.connectors.protocol import (
    ConnectorError,
    ConnectorFetchError,
    ConnectorResult,
)
from tollama.xai.connectors.registry import AsyncConnectorRegistry, ConnectorRegistry
from tollama.xai.trust_contract import TrustEvidence


class AssemblyResult(BaseModel):
    """Result of assembling a payload from a connector."""

    payload: dict[str, Any]
    trust_context: dict[str, Any]
    evidence: TrustEvidence
    connector_name: str = Field(min_length=1)


def _missing_connector_error(
    *,
    domain: str,
    identifier: str,
    connector_label: str,
) -> ConnectorFetchError:
    return ConnectorFetchError(
        ConnectorError(
            domain=domain,
            source_id=identifier,
            error_type="not_found",
            message=f"No {connector_label} for domain={domain!r}, identifier={identifier!r}",
            retryable=False,
        )
    )


def _build_assembly_result(
    *,
    result: ConnectorResult,
    connector_name: str,
) -> AssemblyResult:
    return AssemblyResult(
        payload=result.payload,
        trust_context={"domain": result.domain, "source_type": result.source_type},
        evidence=TrustEvidence(
            source_type=result.source_type,
            source_ids=[result.source_id],
            freshness_seconds=result.freshness_seconds,
            attributes=result.metadata,
        ),
        connector_name=connector_name,
    )


class PayloadAssembler:
    """Uses connectors to fetch data and produce typed payloads for trust agents."""

    def __init__(self, registry: ConnectorRegistry) -> None:
        self.registry = registry

    def assemble(
        self,
        domain: str,
        identifier: str,
        context: dict[str, Any] | None = None,
    ) -> AssemblyResult:
        """Fetch data via connector and produce a typed payload + evidence.

        Raises ConnectorFetchError if no connector is found or fetch fails.
        """
        ctx = context or {}
        connector = self.registry.get(domain, identifier, ctx)
        if connector is None:
            raise _missing_connector_error(
                domain=domain,
                identifier=identifier,
                connector_label="connector",
            )

        return _build_assembly_result(
            result=connector.fetch(identifier, ctx),
            connector_name=connector.connector_name,
        )


class AsyncPayloadAssembler:
    """Async variant of PayloadAssembler for non-blocking connector fetch."""

    def __init__(self, registry: AsyncConnectorRegistry) -> None:
        self.registry = registry

    async def assemble(
        self,
        domain: str,
        identifier: str,
        context: dict[str, Any] | None = None,
    ) -> AssemblyResult:
        """Fetch data via async connector and produce a typed payload + evidence.

        Raises ConnectorFetchError if no connector is found or fetch fails.
        """
        ctx = context or {}
        connector = self.registry.get(domain, identifier, ctx)
        if connector is None:
            raise _missing_connector_error(
                domain=domain,
                identifier=identifier,
                connector_label="async connector",
            )

        return _build_assembly_result(
            result=await connector.fetch(identifier, ctx),
            connector_name=connector.connector_name,
        )


__all__ = ["AssemblyResult", "AsyncPayloadAssembler", "PayloadAssembler"]
