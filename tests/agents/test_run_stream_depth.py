"""Tests for depth parameter plumbing in AgentRunContext and BaseAgent.run_stream()."""

from __future__ import annotations

import pytest

from wolfharness.agents.context import AgentRunContext


pytestmark = pytest.mark.unit


def test_agent_run_context_depth_default() -> None:
    """AgentRunContext should default depth to 0."""
    ctx = AgentRunContext()
    assert ctx.depth == 0


def test_agent_run_context_depth_explicit() -> None:
    """AgentRunContext should accept explicit depth value."""
    ctx = AgentRunContext(depth=3)
    assert ctx.depth == 3


def test_agent_run_context_depth_with_deps() -> None:
    """AgentRunContext should accept depth alongside deps."""
    ctx = AgentRunContext(deps="some_deps", depth=1)
    assert ctx.depth == 1
    assert ctx.deps == "some_deps"


def test_agent_run_context_depth_zero_explicit() -> None:
    """AgentRunContext should accept depth=0 explicitly."""
    ctx = AgentRunContext(depth=0)
    assert ctx.depth == 0


def test_agent_run_context_terminal_tool_state_default() -> None:
    """AgentRunContext should default terminal-tool state to empty."""
    ctx = AgentRunContext()
    assert ctx.terminal_tool_name is None
    assert ctx.terminal_tool_result is None


def test_terminal_tool_metadata_marks_run_completion() -> None:
    """Terminal-tool behavior should be declared by metadata, not by tool name."""
    from wolfharness.tools.base import TERMINAL_TOOL_METADATA_KEY, has_terminal_tool_metadata

    assert has_terminal_tool_metadata({TERMINAL_TOOL_METADATA_KEY: "true"})
    assert not has_terminal_tool_metadata({"name": "attempt_completion"})


def test_run_stream_accepts_depth_param() -> None:
    """BaseAgent.run_stream() should accept depth parameter without TypeError.

    We verify the method signature accepts depth by inspecting the parameter.
    A full integration test would require an agent with a model, which is
    covered by integration tests.
    """
    import inspect

    from wolfharness.agents.base_agent import BaseAgent

    sig = inspect.signature(BaseAgent.run_stream)
    assert "depth" in sig.parameters
    assert sig.parameters["depth"].default == 0


def test_run_accepts_depth_param() -> None:
    """BaseAgent.run() should accept depth parameter without TypeError."""
    import inspect

    from wolfharness.agents.base_agent import BaseAgent

    sig = inspect.signature(BaseAgent.run)
    assert "depth" in sig.parameters
    assert sig.parameters["depth"].default == 0
