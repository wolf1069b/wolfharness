"""ACP Agent configuration models."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from wolfharness.models.acp_agents.base import (
    ACPAgentConfig,
    BaseACPAgentConfig,
    RegistryACPAgentConfig,
)

# Union of all ACP agent config types (discriminated by 'provider')
ACPAgentConfigTypes = Annotated[
    ACPAgentConfig | RegistryACPAgentConfig,
    Field(discriminator="provider"),
]

__all__ = [
    "ACPAgentConfig",
    "ACPAgentConfigTypes",
    "BaseACPAgentConfig",
    "RegistryACPAgentConfig",
]
