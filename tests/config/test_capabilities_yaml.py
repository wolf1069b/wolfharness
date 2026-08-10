"""Tests for YAML capabilities list parsing in NativeAgentConfig."""

from __future__ import annotations

import pytest

from wolfharness import AgentsManifest, NativeAgentConfig
from wolfharness_config.capabilities import (
    EntryPointCapabilityConfig,
    GenericCapabilityConfig,
    LoopDetectionCapabilityConfig,
    TokenBudgetCapabilityConfig,
)


pytestmark = pytest.mark.unit


YAML_WITH_CAPABILITIES = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: pydantic_ai.capabilities.Instrumentation
        args:
          service_name: test
"""

YAML_WITH_MULTIPLE_CAPABILITIES = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities:
      - type: pydantic_ai.capabilities.Instrumentation
        args:
          service_name: test
      - type: pydantic_ai.capabilities.RetryStrategy
        args:
          max_retries: 3
"""

YAML_WITH_EMPTY_CAPABILITIES = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
    capabilities: []
"""

YAML_WITHOUT_CAPABILITIES = """\
agents:
  test_agent:
    type: native
    model: openai:gpt-4o-mini
"""

_CAPABILITY_CONFIG_TYPES = (
    LoopDetectionCapabilityConfig,
    TokenBudgetCapabilityConfig,
    EntryPointCapabilityConfig,
    GenericCapabilityConfig,
)


def _is_capability_config(obj: object) -> bool:
    return isinstance(obj, _CAPABILITY_CONFIG_TYPES)


def test_yaml_capabilities_parsing():
    """Test that YAML capabilities list parses into CapabilityConfig objects."""
    manifest = AgentsManifest.from_yaml(YAML_WITH_CAPABILITIES)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 1
    cap = agent.capabilities[0]
    assert _is_capability_config(cap)
    assert cap.type == "pydantic_ai.capabilities.Instrumentation"
    assert cap.args == {"service_name": "test"}


def test_yaml_multiple_capabilities_parsing():
    """Test parsing multiple capabilities from YAML."""
    manifest = AgentsManifest.from_yaml(YAML_WITH_MULTIPLE_CAPABILITIES)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert len(agent.capabilities) == 2
    assert _is_capability_config(agent.capabilities[0])
    assert agent.capabilities[0].type == "pydantic_ai.capabilities.Instrumentation"
    assert agent.capabilities[0].args == {"service_name": "test"}

    assert _is_capability_config(agent.capabilities[1])
    assert agent.capabilities[1].type == "pydantic_ai.capabilities.RetryStrategy"
    assert agent.capabilities[1].args == {"max_retries": 3}


def test_yaml_empty_capabilities():
    """Test that empty capabilities list parses correctly."""
    manifest = AgentsManifest.from_yaml(YAML_WITH_EMPTY_CAPABILITIES)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert agent.capabilities == []


def test_yaml_no_capabilities():
    """Test that omitted capabilities defaults to empty list."""
    manifest = AgentsManifest.from_yaml(YAML_WITHOUT_CAPABILITIES)
    agent = manifest.agents["test_agent"]
    assert isinstance(agent, NativeAgentConfig)

    assert agent.capabilities == []


if __name__ == "__main__":
    pytest.main(["-v", __file__])
