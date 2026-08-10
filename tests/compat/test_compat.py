"""Backward compatibility and deprecation warning tests for shim APIs.

Consolidated from:
- test_backward_compat.py (deprecated classes still work with _warn=False)
- test_deprecation_warnings.py (DeprecationWarning emitted correctly)

Note: ToolManager was removed in M3. DeprecationWarning tests for
AgentHooks, MCPManager, and wrap_instruction were removed because those
APIs no longer emit warnings. Only _resolve_history_processors
still emits its deprecation warning.
"""

from __future__ import annotations

import pytest

from wolfharness.hooks.agent_hooks import AgentHooks
from wolfharness.utils.context_wrapping import wrap_instruction


pytestmark = pytest.mark.unit


# ============================================================================
# Backward compatibility (_warn=False)
# ============================================================================


def test_agent_hooks_still_works() -> None:
    """AgentHooks with _warn=False initializes and accepts hooks."""
    ah = AgentHooks(_warn=False)
    assert ah.has_hooks() is False
    assert ah.pre_turn == []
    assert ah.post_turn == []
    assert ah.pre_tool_use == []
    assert ah.post_tool_use == []


def test_mcp_manager_still_works() -> None:
    """MCPManager initializes and accepts server configs."""
    from wolfharness.mcp_server.manager import MCPManager

    mm = MCPManager()
    assert mm.name == "mcp"
    assert mm.servers == []
    assert mm.providers == []


def test_resolve_history_processors_still_works() -> None:
    """_resolve_history_processors with _warn=False returns list."""
    from wolfharness.agents.native_agent.agent import Agent

    agent = Agent.__new__(Agent)
    agent._resolved_history_processors = None
    agent._direct_history_processors = None

    # Mock conversation with a config that has no history_processors
    class FakeConfig:
        history_processors = None

    class FakeConversation:
        _config = FakeConfig()

    agent.conversation = FakeConversation()

    result = agent._resolve_history_processors(_warn=False)
    assert isinstance(result, list)
    assert result == []


def test_wrap_instruction_still_works() -> None:
    """wrap_instruction with _warn=False returns callable."""

    def simple_instruction() -> str:
        return "hello"

    wrapped = wrap_instruction(simple_instruction, _warn=False)
    assert callable(wrapped)


# ============================================================================
# Remaining deprecation warning (only _resolve_history_processors still emits)
# ============================================================================


def test_resolve_history_processors_emits_deprecation_warning() -> None:
    """_resolve_history_processors emits DeprecationWarning with v0.5.0 and alternative."""
    from wolfharness.agents.native_agent.agent import Agent

    agent = Agent.__new__(Agent)
    agent._resolved_history_processors = None
    agent._direct_history_processors = None

    class FakeConfig:
        history_processors = None

    class FakeConversation:
        _config = FakeConfig()

    agent.conversation = FakeConversation()

    with pytest.warns(DeprecationWarning, match="v0\\.5\\.0") as warning_list:
        agent._resolve_history_processors()
    assert len(warning_list) == 1
    msg = str(warning_list[0].message)
    assert "_resolve_history_processors() is deprecated" in msg
    assert "ProcessHistoryAdapter" in msg
