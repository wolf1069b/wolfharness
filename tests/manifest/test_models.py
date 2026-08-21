"""Tests for agent configuration models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest
from schemez import InlineSchemaDef
import yamling

from wolfharness import AgentsManifest


pytestmark = pytest.mark.unit


VALID_AGENT_CONFIG = """\
responses:
  TestResponse:
    response_schema:
        description: Test response
        type: inline
        fields:
            message:
                type: str
                description: A message
            score:
                type: int
                ge: 0
                le: 100

agents:
  test_agent:  # Key is the agent ID
    type: native
    name: Test Agent
    description: A test agent
    model: test
    output_type: TestResponse
    system_prompt: You are a test agent
"""

INVALID_RESPONSE_CONFIG = """\
responses:
  InvalidResponse:
    type: object

agents:
  test_agent:
    type: native
    model: "openai:gpt-4o"
    system_prompt: "test"
    output_type: NonExistentResponse
"""


ENV_CONFIG = """\
{}
"""

ENV_AGENT = """\
responses:
    BasicResult:
       response_schema:
            description: Test result
            type: inline
            fields:
                message:
                    type: str
                    description: Test message

agents:
    test_agent:
        type: native
        name: test
        model: test
        output_type: BasicResult
"""


def test_valid_agent_definition():
    """Test valid complete agent configuration."""
    agent_def = AgentsManifest.model_validate(yamling.load_yaml(VALID_AGENT_CONFIG))
    schema = agent_def.responses["TestResponse"].response_schema
    assert isinstance(schema, InlineSchemaDef)
    score = schema.fields["score"]  # pyright: ignore
    assert score.ge == 0
    assert score.le == 100


def test_missing_referenced_response():
    """Test referencing non-existent response model."""
    config = yamling.load_yaml(INVALID_RESPONSE_CONFIG)
    with pytest.raises(ValidationError):
        AgentsManifest.model_validate(config)


def _agent_config_with_mode(mode: str) -> str:
    """Build a minimal agent YAML declaring the given mode."""
    return f"""\
agents:
  test_agent:
    type: native
    name: Test Agent
    model: test
    mode: {mode}
    system_prompt: You are a test agent
"""


def test_agent_mode_subagent_parses():
    """A manifest agent may declare mode: subagent."""
    manifest = AgentsManifest.model_validate(yamling.load_yaml(_agent_config_with_mode("subagent")))
    assert manifest.agents["test_agent"].mode == "subagent"


def test_agent_mode_all_parses():
    """A manifest agent may declare mode: all."""
    manifest = AgentsManifest.model_validate(yamling.load_yaml(_agent_config_with_mode("all")))
    assert manifest.agents["test_agent"].mode == "all"


def test_agent_mode_defaults_to_primary():
    """An omitted mode field defaults to primary (backward compat)."""
    manifest = AgentsManifest.model_validate(yamling.load_yaml(VALID_AGENT_CONFIG))
    assert manifest.agents["test_agent"].mode == "primary"


def test_agent_mode_invalid_value_rejected():
    """An unknown mode value is rejected at config parse time."""
    config = yamling.load_yaml(_agent_config_with_mode("invalid-mode"))
    with pytest.raises(ValidationError):
        AgentsManifest.model_validate(config)
