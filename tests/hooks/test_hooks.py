"""Tests for agent lifecycle hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from wolfharness import Agent
from wolfharness.hooks import AgentHooks, CallableHook


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from wolfharness.hooks import HookResult


# Hook state for testing
hook_state: dict[str, list] = {"calls": [], "results": []}


def reset_hook_state():
    """Reset hook state between tests."""
    hook_state["calls"] = []
    hook_state["results"] = []


# Hook functions


def allow_hook(**kwargs) -> HookResult:
    """Hook that allows the action."""
    hook_state["calls"].append(("allow", kwargs.get("event")))
    return {"decision": "allow"}


def deny_hook(**kwargs) -> HookResult:
    """Hook that denies the action."""
    hook_state["calls"].append(("deny", kwargs.get("event")))
    return {"decision": "deny", "reason": "Denied by test hook"}


def record_result_hook(**kwargs) -> HookResult:
    """Hook that records data passed to it."""
    hook_state["results"].append(kwargs)
    return {"decision": "allow"}


def modify_input_hook(**kwargs) -> HookResult:
    """Hook that modifies tool input."""
    hook_state["calls"].append(("modify", kwargs.get("tool_input")))
    return {"decision": "allow", "modified_input": {"modified": True}}


# Tests for pre_turn hooks


async def test_pre_turn_hook_allow():
    """Pre-turn hooks do not fire in standalone native mode.

    Native agents fire pre_turn/post_turn hooks via HookAwareTurn in
    NativeTurn.execute() (SessionPool path). Standalone runs (agent.run())
    no longer fire these hooks. See test_hook_smoke_matrix.py for
    SessionPool coverage.
    """
    reset_hook_state()

    hooks = AgentHooks(pre_turn=[CallableHook(event="pre_turn", fn=allow_hook)])
    agent = Agent(model="test", hooks=hooks)

    async with agent:
        result = await agent.run("Hello")

    # Hooks are not fired in standalone native mode
    assert len(hook_state["calls"]) == 0
    assert result.content is not None


async def test_pre_turn_hook_deny():
    """Pre-turn deny does not block in standalone native mode.

    Native agents fire pre_turn/post_turn hooks via HookAwareTurn in
    NativeTurn.execute() (SessionPool path). Standalone runs (agent.run())
    no longer fire these hooks. See test_hook_smoke_matrix.py for
    SessionPool coverage.
    """
    reset_hook_state()

    hooks = AgentHooks(pre_turn=[CallableHook(event="pre_turn", fn=deny_hook)])
    agent = Agent(model="test", hooks=hooks)

    async with agent:
        result = await agent.run("Hello")

    # Hooks are not fired in standalone native mode, so run proceeds
    assert result is not None
    assert len(hook_state["calls"]) == 0


# Tests for post_turn hooks


async def test_post_turn_hook():
    """Post-turn hooks do not fire in standalone native mode.

    Native agents fire pre_turn/post_turn hooks via HookAwareTurn in
    NativeTurn.execute() (SessionPool path). Standalone runs (agent.run())
    no longer fire these hooks. See test_hook_smoke_matrix.py for
    SessionPool coverage.
    """
    reset_hook_state()

    hooks = AgentHooks(post_turn=[CallableHook(event="post_turn", fn=record_result_hook)])
    agent = Agent(model="test", hooks=hooks)

    async with agent:
        await agent.run("Hello")

    # Hooks are not fired in standalone native mode
    assert len(hook_state["results"]) == 0


# Tests for pre_tool_use hooks


async def test_pre_tool_hook_allow():
    """Test pre-tool hook that allows tool execution."""
    reset_hook_state()

    def simple_tool() -> str:
        """A simple test tool."""
        return "tool_result"

    hooks = AgentHooks(pre_tool_use=[CallableHook(event="pre_tool_use", fn=allow_hook)])
    async with Agent(model="test", hooks=hooks, tools=[simple_tool]) as agent:
        # Just verify the hooks are set up correctly
        assert agent._hook_manager.has_hooks()
        assert agent._hook_manager.agent_hooks
        assert len(agent._hook_manager.agent_hooks.pre_tool_use) == 1


async def test_pre_tool_hook_with_matcher():
    """Test that matcher filters which tools trigger hooks."""
    reset_hook_state()

    # Hook that only matches "other_tool", not our actual tool
    hooks = AgentHooks(
        pre_tool_use=[CallableHook(event="pre_tool_use", fn=deny_hook, matcher="other_tool")]
    )
    async with Agent(model="test", hooks=hooks) as agent:
        # Run should succeed because matcher doesn't match any tool
        result = await agent.run("Hello")
        assert result.content is not None


# Tests for post_tool_use hooks


async def test_post_tool_hook():
    """Test post-tool hook setup."""
    reset_hook_state()
    hooks = AgentHooks(post_tool_use=[CallableHook(event="post_tool_use", fn=record_result_hook)])
    async with Agent(model="test", hooks=hooks) as agent:
        assert agent._hook_manager.has_hooks()
        assert agent._hook_manager.agent_hooks
        assert len(agent._hook_manager.agent_hooks.post_tool_use) == 1


# Tests for AgentHooks class


def test_agent_hooks_has_hooks():
    """Test has_hooks method."""
    empty = AgentHooks()
    assert not empty.has_hooks()
    with_pre_turn = AgentHooks(pre_turn=[CallableHook(event="pre_turn", fn=allow_hook)])
    assert with_pre_turn.has_hooks()


def test_agent_hooks_repr():
    """Test AgentHooks string representation."""
    empty = AgentHooks()
    assert repr(empty) == "AgentHooks(empty)"

    with_hooks = AgentHooks(
        pre_turn=[CallableHook(event="pre_turn", fn=allow_hook)],
        post_tool_use=[CallableHook(event="post_tool_use", fn=allow_hook)],
    )
    assert "pre_turn=1" in repr(with_hooks)
    assert "post_tool_use=1" in repr(with_hooks)


# Tests for multiple hooks


async def test_multiple_hooks_all_allow():
    """Multiple pre_turn hooks do not fire in standalone native mode.

    Native agents fire pre_turn/post_turn hooks via HookAwareTurn in
    NativeTurn.execute() (SessionPool path). Standalone runs (agent.run())
    no longer fire these hooks. See test_hook_smoke_matrix.py for
    SessionPool coverage.
    """
    reset_hook_state()

    hooks = AgentHooks(
        pre_turn=[
            CallableHook(event="pre_turn", fn=allow_hook),
            CallableHook(event="pre_turn", fn=allow_hook),
        ]
    )
    async with Agent(model="test", hooks=hooks) as agent:
        result = await agent.run("Hello")

    # Hooks are not fired in standalone native mode
    assert len(hook_state["calls"]) == 0
    assert result.content is not None


async def test_multiple_hooks_one_denies():
    """Multiple hooks with one deny do not fire in standalone native mode.

    Native agents fire pre_turn/post_turn hooks via HookAwareTurn in
    NativeTurn.execute() (SessionPool path). Standalone runs (agent.run())
    no longer fire these hooks. See test_hook_smoke_matrix.py for
    SessionPool coverage.
    """
    reset_hook_state()

    hooks = AgentHooks(
        pre_turn=[
            CallableHook(event="pre_turn", fn=allow_hook),
            CallableHook(event="pre_turn", fn=deny_hook),
        ]
    )
    async with Agent(model="test", hooks=hooks) as agent:
        result = await agent.run("Hello")

    # Hooks are not fired in standalone native mode, so run proceeds
    assert result is not None
    assert len(hook_state["calls"]) == 0


# Tests for input_match


def test_input_match_matches_when_field_present():
    """input_match should match when the specified field value satisfies the regex."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$"},
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "new_task",
        "tool_input": {"mode": "libarian", "task_tag": "diagnosis_planning"},
    }
    assert hook.matches(data) is True


def test_input_match_rejects_wrong_value():
    """input_match should reject when the field value doesn't match."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$"},
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "new_task",
        "tool_input": {"mode": "libarian", "task_tag": "general_research"},
    }
    assert hook.matches(data) is False


def test_input_match_rejects_missing_field():
    """input_match should reject when the field is absent from tool_input."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$"},
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "new_task",
        "tool_input": {"mode": "libarian"},
    }
    assert hook.matches(data) is False


def test_input_match_multiple_fields_all_must_match():
    """All input_match patterns must match for the hook to trigger."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$", "mode": "^libarian$"},
    )
    # Both match
    assert (
        hook.matches({
            "event": "post_tool_use",
            "tool_name": "new_task",
            "tool_input": {"mode": "libarian", "task_tag": "diagnosis_planning"},
        })
        is True
    )
    # One mismatch
    assert (
        hook.matches({
            "event": "post_tool_use",
            "tool_name": "new_task",
            "tool_input": {"mode": "rebuttal_agent", "task_tag": "diagnosis_planning"},
        })
        is False
    )


def test_input_match_with_tool_name_mismatch():
    """Hook should not match when tool_name doesn't match even if input_match does."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$"},
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "other_tool",
        "tool_input": {"task_tag": "diagnosis_planning"},
    }
    assert hook.matches(data) is False


def test_no_input_match_matches_all():
    """Without input_match, hook should match based on tool_name alone."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "new_task",
        "tool_input": {"mode": "anything", "task_tag": "whatever"},
    }
    assert hook.matches(data) is True


def test_input_match_no_recursive_trigger():
    """Hook on new_task+diagnosis_planning must NOT match quality_review."""
    hook = CallableHook(
        event="post_tool_use",
        fn=allow_hook,
        matcher="new_task",
        input_match={"task_tag": "^diagnosis_planning$"},
    )
    data = {
        "event": "post_tool_use",
        "tool_name": "new_task",
        "tool_input": {"mode": "rebuttal_agent", "task_tag": "quality_review"},
    }
    assert hook.matches(data) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
