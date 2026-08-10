"""Package resources for AgentPool configuration."""

from __future__ import annotations

import importlib.resources
from typing import Final

_RESOURCES = importlib.resources.files("wolfharness.config_resources")

# Template configuration
AGENTS_TEMPLATE: Final[str] = str(_RESOURCES / "agents_template.yml")
"""Path to the agents template configuration."""

# Pool configurations
ACP_ASSISTANT: Final[str] = str(_RESOURCES / "acp_assistant.yml")
"""Path to default ACP assistant configuration."""

AGENTS: Final[str] = str(_RESOURCES / "agents.yml")
"""Path to the main agents configuration."""

EXTERNAL_ACP_AGENTS: Final[str] = str(_RESOURCES / "external_acp_agents.yml")
"""Path to external ACP agents configuration."""

TTS_TEST_AGENTS: Final[str] = str(_RESOURCES / "tts_test_agents.yml")
"""Path to TTS test agents configuration."""

# All pool configuration paths for validation
ALL_POOL_CONFIGS: Final[tuple[str, ...]] = (
    ACP_ASSISTANT,
    AGENTS,
    EXTERNAL_ACP_AGENTS,
    TTS_TEST_AGENTS,
)
"""All pool configuration file paths."""

__all__ = [
    "ACP_ASSISTANT",
    "AGENTS",
    "AGENTS_TEMPLATE",
    "ALL_POOL_CONFIGS",
    "EXTERNAL_ACP_AGENTS",
    "TTS_TEST_AGENTS",
]
