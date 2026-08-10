"""Tests for BackgroundTaskCapability schema loading and tool registration."""

# pyright: reportAttributeAccessIssue=false
# Mock-heavy test code: accessing tools on AbstractToolset (runtime FunctionToolset) is expected.

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

from pydantic_ai import RunContext
import pytest

from wolfharness import Agent, AgentContext
from wolfharness.agents.base_agent import BaseAgent
from wolfharness.capabilities.background_task.capability import BackgroundTaskCapability
from wolfharness.capabilities.background_task.manager import BackgroundTaskManager
from wolfharness.delegation import AgentPool
from wolfharness.tools.exceptions import ToolError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap_in_run_context(agent_ctx):
    """Wrap an AgentContext in a mock RunContext for capability tool methods."""
    run_ctx = MagicMock()
    run_ctx.deps = agent_ctx
    run_ctx.tool_call_id = agent_ctx.tool_call_id
    return run_ctx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pool_with_agents() -> AgentPool:
    """Create a mock AgentPool with agents.

    Uses MagicMock (not AsyncMock) for the pool itself so that
    ``bool(pool)`` returns True, matching real AgentPool behavior.
    """
    pool = MagicMock(spec=AgentPool)

    mock_agent = MagicMock(spec=Agent)
    mock_agent.description = "A test agent"
    mock_agent.name = "test_agent"

    other_agent = MagicMock(spec=Agent)
    other_agent.description = "Another agent"
    other_agent.name = "other_agent"

    agents_dict = {"test_agent": mock_agent, "other_agent": other_agent}
    pool.nodes = agents_dict
    pool.agent_configs = agents_dict
    pool.manifest = MagicMock()
    pool.manifest.agents = agents_dict

    return pool


@pytest.fixture
def valid_task_schema_yaml() -> str:
    """Return a valid YAML schema for the task tool."""
    return """
name: task
description: Custom description for task tool
parameters:
  type: object
  properties:
    mode:
      type: string
      description: The specialized mode for the new task.
    load_skills:
      type: array
      description: Skill names to inject.
      items:
        type: string
    title:
      type: string
      description: Optional title for the subtask.
    message:
      type: string
      description: A clear, concise statement of what the task entails.
    expected_output:
      type: string
      description: A precise definition of the successful outcome.
    async_mode:
      type: boolean
      description: When true, run in background.
      default: false
  required:
    - mode
    - load_skills
    - message
    - expected_output
  strict: true
"""


@pytest.fixture
def valid_background_output_schema_yaml() -> str:
    """Return a valid YAML schema for background_output."""
    return """
name: background_output
description: Custom description for background_output tool
parameters:
  type: object
  properties:
    task_id:
      type: string
      description: The ID of the background task.
    block:
      type: boolean
      description: Whether to block until task completes.
      default: false
  required:
    - task_id
"""


@pytest.fixture
def valid_background_cancel_schema_yaml() -> str:
    """Return a valid YAML schema for background_cancel."""
    return """
name: background_cancel
description: Custom description for background_cancel tool
parameters:
  type: object
  properties:
    task_id:
      type: string
      description: The ID of the background task to cancel.
    cancel_all:
      type: boolean
      description: When true, cancel all running tasks.
      default: false
"""


def _write_temp_yaml(content: str) -> str:
    """Write content to a temporary YAML file and return the path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(content)
        return f.name


# ---------------------------------------------------------------------------
# Capability instantiation
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_capability_without_schema_paths():
    """Test capability initialization without schema paths (None)."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)

    assert capability is not None
    assert capability._task_schema is None
    assert capability._background_output_schema is None
    assert capability._background_cancel_schema is None

    toolset = capability.get_toolset()
    assert toolset is not None
    assert len(toolset.tools) == 4
    tool_names = set(toolset.tools.keys())
    assert tool_names == {"task", "background_output", "background_cancel", "steer_task"}


@pytest.mark.unit
async def test_capability_with_valid_task_yaml_schema(
    valid_task_schema_yaml: str,
):
    """Test capability initialization with valid YAML schema for task."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    schema_path = _write_temp_yaml(valid_task_schema_yaml)

    try:
        capability = BackgroundTaskCapability(
            schemas={"task": schema_path},
        )

        assert capability._task_schema is not None
        assert capability._task_schema["name"] == "task"
        assert capability._task_schema["description"] == "Custom description for task tool"

        # background_output and background_cancel schemas should be None
        assert capability._background_output_schema is None
        assert capability._background_cancel_schema is None

        toolset = capability.get_toolset()
        assert toolset is not None
        assert len(toolset.tools) == 4
    finally:
        Path(schema_path).unlink(missing_ok=True)


@pytest.mark.unit
async def test_capability_with_all_valid_yaml_schemas(
    valid_task_schema_yaml: str,
    valid_background_output_schema_yaml: str,
    valid_background_cancel_schema_yaml: str,
):
    """Test capability initialization with valid YAML schemas for all tools."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    task_path = _write_temp_yaml(valid_task_schema_yaml)
    output_path = _write_temp_yaml(valid_background_output_schema_yaml)
    cancel_path = _write_temp_yaml(valid_background_cancel_schema_yaml)

    try:
        capability = BackgroundTaskCapability(
            schemas={
                "task": task_path,
                "background_output": output_path,
                "background_cancel": cancel_path,
            },
        )

        assert capability._task_schema is not None
        assert capability._background_output_schema is not None
        assert capability._background_cancel_schema is not None

        assert capability._task_schema["name"] == "task"
        assert capability._background_output_schema["name"] == "background_output"
        assert capability._background_cancel_schema["name"] == "background_cancel"

        toolset = capability.get_toolset()
        assert toolset is not None
        assert len(toolset.tools) == 4
    finally:
        Path(task_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        Path(cancel_path).unlink(missing_ok=True)


@pytest.mark.unit
def test_capability_with_nonexistent_schema_path():
    """Test capability initialization fails with nonexistent schema path (Fail Fast)."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    nonexistent_path = "/path/that/does/not/exist/schema.yaml"

    with pytest.raises(FileNotFoundError, match="Tool schema file not found"):
        BackgroundTaskCapability(
            schemas={"task": nonexistent_path},
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_all_three_tools_registered():
    """Test that all three tools are registered by default."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)
    toolset = capability.get_toolset()

    assert toolset is not None
    assert len(toolset.tools) == 4
    tool_names = list(toolset.tools.keys())
    assert "task" in tool_names
    assert "background_output" in tool_names
    assert "background_cancel" in tool_names
    assert "steer_task" in tool_names


@pytest.mark.unit
async def test_enabled_tools_filtering():
    """Test that enabled_tools correctly filters registered tools."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    # Only enable task and background_output
    capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=["task", "background_output"],
    )
    toolset = capability.get_toolset()

    assert toolset is not None
    assert len(toolset.tools) == 2
    tool_names = set(toolset.tools.keys())
    assert tool_names == {"task", "background_output"}


@pytest.mark.unit
async def test_enabled_tools_only_task():
    """Test that only task tool is registered when only it is enabled."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=["task"],
    )
    toolset = capability.get_toolset()

    assert toolset is not None
    assert len(toolset.tools) == 1
    assert "task" in toolset.tools


@pytest.mark.unit
async def test_enabled_tools_empty_list_enables_all():
    """Test that empty enabled_tools list enables all tools."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(
        schemas=None,
        enabled_tools=[],
    )
    toolset = capability.get_toolset()

    assert toolset is not None
    assert len(toolset.tools) == 4


# ---------------------------------------------------------------------------
# Tool method stubs
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_task_method_no_pool_raises_tool_error():
    """Test that task method raises ToolError when no pool is available."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="test_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    with pytest.raises(ToolError, match="No agent pool available"):
        await capability._task(
            _wrap_in_run_context(agent_ctx), agent="test", message="test", expected_output="test"
        )


@pytest.mark.unit
async def test_background_output_nonexistent_task_returns_message():
    """Test that background_output returns not-found message for missing task."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="test_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    result = await capability._background_output(
        _wrap_in_run_context(agent_ctx), task_id="nonexistent"
    )
    assert "not found" in result


@pytest.mark.unit
async def test_background_cancel_no_args_raises_tool_error():
    """Test that background_cancel raises ToolError when no args provided."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)
    agent = Agent(name="test_agent", model="test")
    agent_ctx = AgentContext(node=agent, pool=None)

    with pytest.raises(ToolError, match="Either task_id or cancel_all"):
        await capability._background_cancel(_wrap_in_run_context(agent_ctx))


# ---------------------------------------------------------------------------
# Task manager instance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_task_manager_instance_stored():
    """Test that a BackgroundTaskManager instance is stored in session state."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    capability = BackgroundTaskCapability(schemas=None)

    # _task_manager is now per-session, accessed via _get_session_state()
    # Verify the capability creates a SessionTaskState with a BackgroundTaskManager
    agent = MagicMock(spec=BaseAgent)
    agent.name = "coordinator"
    agent.session_id = "ses_parent_001"
    agent.agent_pool = None

    ctx = MagicMock(spec=AgentContext)
    ctx.node = agent
    ctx.pool = None
    ctx.data = {}
    ctx.tool_call_id = "tc_001"

    mock_run_ctx = MagicMock()
    mock_run_ctx.session_id = "ses_parent_001"
    mock_run_ctx._run_handle = None
    mock_run_ctx.child_done_events = {}
    ctx.run_ctx = mock_run_ctx

    wrapped = MagicMock(spec=RunContext)
    wrapped.deps = ctx

    state = capability._get_session_state(wrapped)
    assert isinstance(state.task_manager, BackgroundTaskManager)


# ---------------------------------------------------------------------------
# Schema override on tools
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_schema_override_applied_to_tools(
    valid_task_schema_yaml: str,
    valid_background_output_schema_yaml: str,
    valid_background_cancel_schema_yaml: str,
):
    """Test that schema overrides are correctly applied to tools."""
    os.environ["OBSERVABILITY_ENABLED"] = "false"

    task_path = _write_temp_yaml(valid_task_schema_yaml)
    output_path = _write_temp_yaml(valid_background_output_schema_yaml)
    cancel_path = _write_temp_yaml(valid_background_cancel_schema_yaml)

    try:
        capability = BackgroundTaskCapability(
            schemas={
                "task": task_path,
                "background_output": output_path,
                "background_cancel": cancel_path,
            },
        )

        toolset = capability.get_toolset()
        assert toolset is not None
        tool_map = toolset.tools

        # Verify descriptions come from schemas
        assert tool_map["task"].description == "Custom description for task tool"
        assert (
            tool_map["background_output"].description
            == "Custom description for background_output tool"
        )
        assert (
            tool_map["background_cancel"].description
            == "Custom description for background_cancel tool"
        )
    finally:
        Path(task_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        Path(cancel_path).unlink(missing_ok=True)
