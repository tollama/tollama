"""Input schemas for tollama MCP tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


class MCPInputBase(BaseModel):
    """Strict base schema for MCP tool arguments."""

    model_config = ConfigDict(extra="forbid", strict=True)


class HealthToolInput(MCPInputBase):
    base_url: StrictStr | None = None
    timeout: StrictFloat | None = Field(default=None, gt=0)


class ModelsToolInput(MCPInputBase):
    mode: Literal["installed", "loaded", "available"] = "installed"
    base_url: StrictStr | None = None
    timeout: StrictFloat | None = Field(default=None, gt=0)


class ForecastToolInput(MCPInputBase):
    request: dict[str, Any]
    base_url: StrictStr | None = None
    timeout: StrictFloat | None = Field(default=None, gt=0)


class PullToolInput(MCPInputBase):
    model: StrictStr = Field(min_length=1)
    accept_license: StrictBool = False
    base_url: StrictStr | None = None
    timeout: StrictFloat | None = Field(default=None, gt=0)


class ShowToolInput(MCPInputBase):
    model: StrictStr = Field(min_length=1)
    base_url: StrictStr | None = None
    timeout: StrictFloat | None = Field(default=None, gt=0)


class RecommendToolInput(MCPInputBase):
    horizon: StrictInt = Field(gt=0)
    freq: StrictStr | None = None
    has_past_covariates: StrictBool = False
    has_future_covariates: StrictBool = False
    has_static_covariates: StrictBool = False
    covariates_type: Literal["numeric", "categorical"] = "numeric"
    allow_restricted_license: StrictBool = False
    top_k: StrictInt = Field(default=3, ge=1, le=20)
