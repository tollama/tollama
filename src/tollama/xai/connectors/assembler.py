"""
tollama.xai.connectors.assembler — PayloadAssembler for building typed payloads from connectors.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from tollama.xai.connectors.protocol import ConnectorError, ConnectorFetchError
from tollama.xai.connectors.registry import AsyncConnectorRegistry, ConnectorRegistry
from tollama.xai.trust_contract import TrustEvidence


class AssemblyResult(BaseModel):
    """Result of assembling a payload from a connector."""

    payload: dict[str, Any]
    trust_context: dict[str, Any]
    evidence: TrustEvidence
    connector_name: str = Field(min_length=1)


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
            raise ConnectorFetchError(
                ConnectorError(
                    domain=domain,
                    source_id=identifier,
                    error_type="not_found",
                    message=(
                        f"No connector for "
                        f"domain={domain!r}, identifier={identifier!r}"
                    ),
                    retryable=False,
                )
            )

        result = connector.fetch(identifier, ctx)

        return AssemblyResult(
            payload=result.payload,
            trust_context={"domain": result.domain, "source_type": result.source_type},
            evidence=TrustEvidence(
                source_type=result.source_type,
                source_ids=[result.source_id],
                freshness_seconds=result.freshness_seconds,
                attributes=result.metadata,
            ),
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
            raise ConnectorFetchError(
                ConnectorError(
                    domain=domain,
                    source_id=identifier,
                    error_type="not_found",
                    message=(
                        f"No async connector for "
                        f"domain={domain!r}, identifier={identifier!r}"
                    ),
                    retryable=False,
                )
            )

        result = await connector.fetch(identifier, ctx)

        return AssemblyResult(
            payload=result.payload,
            trust_context={"domain": result.domain, "source_type": result.source_type},
            evidence=TrustEvidence(
                source_type=result.source_type,
                source_ids=[result.source_id],
                freshness_seconds=result.freshness_seconds,
                attributes=result.metadata,
            ),
            connector_name=connector.connector_name,
        )


__all__ = ["AssemblyResult", "AsyncPayloadAssembler", "PayloadAssembler"]
