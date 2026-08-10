"""Integration tests for NativeTurn with hooks.

Verifies that HookAwareTurn's pre_turn and post_turn hooks fire during
NativeTurn.execute(), and that a pre_turn deny cancels the turn.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import Agent
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events.events import StreamCompleteEvent
from wolfharness.agents.native_agent.turn import NativeTurn
from wolfharness.hooks import AgentHooks, CallableHook, HookResult


# ---------------------------------------------------------------------------
# Test state
# ---------------------------------------------------------------------------

hook_calls: list[tuple[str, dict[str, Any]]] = []


def _reset_calls() -> None:
    hook_calls.clear()


def _make_recorder(event: str) -> CallableHook:
    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append((event, kwargs))
        return {"decision": "allow"}

    return CallableHook(event=event, fn=_fn)  # type: ignore[arg-type]


def _make_denyer(event: str) -> CallableHook:
    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append((event, kwargs))
        return {"decision": "deny", "reason": "blocked by test"}

    return CallableHook(event=event, fn=_fn)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test: hooks fire during turn execution
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_and_post_turn_fire_during_native_turn() -> None:
    """Given a NativeTurn with hooks, pre_turn and post_turn fire during execute()."""
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")],
        post_turn=[_make_recorder("post_turn")],
    )
    agent = Agent(
        name="test-hooks-native",
        model=TestModel(custom_output_text="response"),
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["hello"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )

        events = [event async for event in turn.execute()]

    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names
    assert event_names.index("pre_turn") < event_names.index("post_turn")

    assert any(isinstance(e, StreamCompleteEvent) for e in events)


# ---------------------------------------------------------------------------
# Test: pre_turn deny cancels the turn
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_deny_cancels_native_turn() -> None:
    """Given a denying pre_turn hook, the turn yields StreamCompleteEvent with cancelled=True."""
    _reset_calls()
    hooks = AgentHooks(pre_turn=[_make_denyer("pre_turn")])
    agent = Agent(
        name="test-deny-native",
        model=TestModel(custom_output_text="response"),
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["hello"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )

        events = [event async for event in turn.execute()]

    assert run_ctx.cancelled is True

    stream_complete = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert len(stream_complete) == 1
    assert stream_complete[0].cancelled is True


# ---------------------------------------------------------------------------
# Test: post_turn fires even when turn is denied
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_post_turn_fires_even_when_pre_turn_denies() -> None:
    """Given pre_turn deny, post_turn still fires in the finally block."""
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_denyer("pre_turn")],
        post_turn=[_make_recorder("post_turn")],
    )
    agent = Agent(
        name="test-deny-post-native",
        model=TestModel(custom_output_text="response"),
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["hello"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )

        _ = [event async for event in turn.execute()]

    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names


# ---------------------------------------------------------------------------
# Test: tool hooks NOT fired by HookAwareTurn for native agents
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_tool_hooks_not_fired_by_hook_aware_turn_for_native() -> None:
    """Given a NativeTurn with tool hooks, HookAwareTurn does not fire them.

    Native agents handle tool hooks via ``ToolInterceptCapability``
    (registered in ``get_agentlet()``), not via HookAwareTurn's
    ``_fire_pre_tool_hooks`` / ``_fire_post_tool_hooks`` methods. The mixin
    methods are never called by ``NativeTurn.execute()``.

    We verify this by checking that ``_logged_tools`` does NOT contain any
    tool log keys. If the mixin's ``_fire_post_tool_hooks`` were called,
    it would have invoked ``_log_tool_execution`` which adds keys to
    ``_logged_tools``.
    """
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")],
        post_turn=[_make_recorder("post_turn")],
        pre_tool_use=[_make_recorder("pre_tool_use")],
        post_tool_use=[_make_recorder("post_tool_use")],
    )

    def simple_tool() -> str:
        """A simple tool."""
        return "result"

    agent = Agent(
        name="test-no-tool-hooks-native",
        model=TestModel(call_tools=["simple_tool"], custom_output_text="done"),
        tools=[simple_tool],
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["call the tool"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )

        _ = [event async for event in turn.execute()]

    # pre_turn and post_turn fire via HookAwareTurn
    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names

    # Tool hooks fire via pydantic-ai Hooks capability, but HookAwareTurn's
    # _log_tool_execution is NOT called because NativeTurn.execute() never
    # calls _fire_pre_tool_hooks() / _fire_post_tool_hooks().
    assert len(turn._logged_tools) == 0, (
        f"HookAwareTurn should not log tool executions for native agents, "
        f"but found: {turn._logged_tools}"
    )


# ---------------------------------------------------------------------------
# Test: hooks fire correctly without double-fire guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_hooks_fired_prevents_double_firing_via_old_path() -> None:
    """Given the removal of hooks_fired guard, hooks fire on each call.

    The old ``hooks_fired`` double-fire guard was removed after T3 eliminated
    the ACP standalone path that caused double-firing. With only the
    ``Turn.execute()`` path active, hooks fire exactly once per turn.

    This test verifies that calling ``_fire_pre_turn_hooks`` after
    ``execute()`` completes fires hooks again (no guard to block them).
    This is the expected behavior — the guard is no longer needed.
    """
    _reset_calls()
    hooks = AgentHooks(pre_turn=[_make_recorder("pre_turn")])
    agent = Agent(
        name="test-dedup-native",
        model=TestModel(custom_output_text="response"),
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="test-session")
        turn = NativeTurn(
            agent=agent,
            prompts=["hello"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )

        # Execute the turn (fires hooks once)
        _ = [event async for event in turn.execute()]

        # Without the hooks_fired guard, calling again fires again.
        result = await turn._fire_pre_turn_hooks()
        assert result is not None

    # Two pre_turn calls: one from execute(), one from manual call
    pre_turn_count = sum(1 for name, _ in hook_calls if name == "pre_turn")
    assert pre_turn_count == 2
