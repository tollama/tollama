"""Input schemas for tollama MCP tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr


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
