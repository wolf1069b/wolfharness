"""Unit tests for HookAwareTurn mixin in isolation.

Tests the mixin's hook firing logic and no-op behavior when hooks are None.
The hooks_fired double-fire guard was removed in T4 — hooks now fire on
every call without dedup. Tool execution logging idempotency is handled
by the per-Turn ``_logged_tools`` set. Uses a minimal host class that
implements the three required abstract properties.
"""

from __future__ import annotations

from typing import Any

import pytest

from wolfharness.agents.context import AgentRunContext
from wolfharness.hooks import AgentHooks, CallableHook, HookResult
from wolfharness.orchestrator.turn import HookAwareTurn


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

hook_calls: list[tuple[str, dict[str, Any]]] = []
"""Records (event_name, kwargs) for each hook invocation."""


def _reset_calls() -> None:
    hook_calls.clear()


def _make_recording_hook(event: str) -> CallableHook:
    """Create a CallableHook that records its call into hook_calls."""

    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append((event, kwargs))
        return {"decision": "allow"}

    return CallableHook(event=event, fn=_fn)  # type: ignore[arg-type]


def _make_deny_hook(event: str) -> CallableHook:
    """Create a CallableHook that denies."""

    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append((event, kwargs))
        return {"decision": "deny", "reason": "test deny"}

    return CallableHook(event=event, fn=_fn)  # type: ignore[arg-type]


class _MockHost(HookAwareTurn):
    """Minimal host class for testing HookAwareTurn in isolation."""

    def __init__(
        self,
        hooks: AgentHooks | None,
        run_ctx: AgentRunContext,
        agent_name: str = "test-agent",
        prompt: str = "hello",
    ) -> None:
        super().__init__()
        self._hooks = hooks
        self._run_ctx = run_ctx
        self._agent_name = agent_name
        self._prompt = prompt

    @property
    def _hook_env(self) -> Any | None:
        return None

    @property
    def _hook_agent_name(self) -> str:
        return self._agent_name

    @property
    def _hook_prompt(self) -> str:
        return self._prompt


def _make_host(
    hooks: AgentHooks | None = None,
    run_ctx: AgentRunContext | None = None,
) -> _MockHost:
    return _MockHost(
        hooks=hooks,
        run_ctx=run_ctx or AgentRunContext(session_id="test-session"),
    )


# ---------------------------------------------------------------------------
# Test: all 4 hooks fire in correct order
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_all_four_hooks_fire_in_order() -> None:
    """Given hooks for all 4 events, firing them produces the correct order."""
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recording_hook("pre_turn")],
        post_turn=[_make_recording_hook("post_turn")],
        pre_tool_use=[_make_recording_hook("pre_tool_use")],
        post_tool_use=[_make_recording_hook("post_tool_use")],
    )
    host = _make_host(hooks=hooks)

    await host._fire_pre_turn_hooks()
    await host._fire_pre_tool_hooks("my_tool", {"arg": 1}, "call-1")
    await host._fire_post_tool_hooks("my_tool", {"arg": 1}, "result", 10.0, "call-1")
    await host._fire_post_turn_hooks(None)

    assert [name for name, _ in hook_calls] == [
        "pre_turn",
        "pre_tool_use",
        "post_tool_use",
        "post_turn",
    ]


# ---------------------------------------------------------------------------
# Test: hooks fire on every call (hooks_fired guard removed in T4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_hooks_fire_on_double_call() -> None:
    """Given two calls to _fire_pre_turn_hooks, both fire (no dedup guard)."""
    _reset_calls()
    hooks = AgentHooks(pre_turn=[_make_recording_hook("pre_turn")])
    host = _make_host(hooks=hooks)

    result1 = await host._fire_pre_turn_hooks()
    result2 = await host._fire_pre_turn_hooks()

    assert result1 is not None
    assert result2 is not None
    assert len(hook_calls) == 2


@pytest.mark.unit
async def test_post_turn_hooks_fire_on_double_call() -> None:
    """Given two calls to _fire_post_turn_hooks, both fire (no dedup guard)."""
    _reset_calls()
    hooks = AgentHooks(post_turn=[_make_recording_hook("post_turn")])
    host = _make_host(hooks=hooks)

    await host._fire_post_turn_hooks(None)
    await host._fire_post_turn_hooks(None)

    assert len(hook_calls) == 2


# ---------------------------------------------------------------------------
# Test: no-op when hooks is None
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_noop_when_hooks_none() -> None:
    """Given hooks=None, _fire_pre_turn_hooks returns None without raising."""
    host = _make_host(hooks=None)
    result = await host._fire_pre_turn_hooks()
    assert result is None


@pytest.mark.unit
async def test_post_turn_noop_when_hooks_none() -> None:
    """Given hooks=None, _fire_post_turn_hooks returns None without raising."""
    host = _make_host(hooks=None)
    result = await host._fire_post_turn_hooks(None)
    assert result is None


@pytest.mark.unit
async def test_pre_tool_noop_when_hooks_none() -> None:
    """Given hooks=None, _fire_pre_tool_hooks returns None without raising."""
    host = _make_host(hooks=None)
    result = await host._fire_pre_tool_hooks("tool", {}, "call-1")
    assert result is None


@pytest.mark.unit
async def test_post_tool_noop_when_hooks_none() -> None:
    """Given hooks=None, _fire_post_tool_hooks returns None without raising."""
    host = _make_host(hooks=None)
    result = await host._fire_post_tool_hooks("tool", {}, "result", 0.0, "call-1")
    assert result is None


# ---------------------------------------------------------------------------
# Test: pre_turn fires before execute body
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_fires_and_records_call() -> None:
    """Given a pre_turn hook, firing it records the call."""
    _reset_calls()
    hooks = AgentHooks(pre_turn=[_make_recording_hook("pre_turn")])
    run_ctx = AgentRunContext(session_id="test-session")
    host = _make_host(hooks=hooks, run_ctx=run_ctx)

    await host._fire_pre_turn_hooks()

    assert len(hook_calls) == 1
    assert hook_calls[0][0] == "pre_turn"


# ---------------------------------------------------------------------------
# Test: post_turn fires even when execute raises (simulated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_post_turn_fires_in_finally_block() -> None:
    """Given a simulated finally block, post_turn hooks fire even when an error occurs.

    This simulates the Turn.execute() pattern where _fire_post_turn_hooks
    is called in a finally block.
    """
    _reset_calls()
    hooks = AgentHooks(post_turn=[_make_recording_hook("post_turn")])
    host = _make_host(hooks=hooks)

    # Wrap the try/finally in a helper to satisfy PT012 (single statement
    # in pytest.raises block).
    async def _raise_and_fire() -> None:
        try:
            raise RuntimeError("simulated error")
        finally:
            await host._fire_post_turn_hooks(None)

    with pytest.raises(RuntimeError, match="simulated error"):
        await _raise_and_fire()

    # post_turn must have fired despite the exception
    assert any(name == "post_turn" for name, _ in hook_calls)


# ---------------------------------------------------------------------------
# Test: different tool_call_id allows both hooks to fire (no dedup guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_tool_hooks_fire_for_different_call_ids() -> None:
    """Given same tool_name but different tool_call_id, both pre_tool hooks fire."""
    _reset_calls()
    hooks = AgentHooks(pre_tool_use=[_make_recording_hook("pre_tool_use")])
    run_ctx = AgentRunContext(session_id="test-session")
    host = _make_host(hooks=hooks, run_ctx=run_ctx)

    await host._fire_pre_tool_hooks("same_tool", {"x": 1}, "call-A")
    await host._fire_pre_tool_hooks("same_tool", {"x": 2}, "call-B")

    assert len(hook_calls) == 2


@pytest.mark.unit
async def test_tool_hooks_fire_on_same_call_id() -> None:
    """Given same tool_call_id twice, both pre_tool hooks fire (no dedup guard)."""
    _reset_calls()
    hooks = AgentHooks(pre_tool_use=[_make_recording_hook("pre_tool_use")])
    run_ctx = AgentRunContext(session_id="test-session")
    host = _make_host(hooks=hooks, run_ctx=run_ctx)

    await host._fire_pre_tool_hooks("tool", {}, "call-X")
    result2 = await host._fire_pre_tool_hooks("tool", {}, "call-X")

    assert result2 is not None
    assert len(hook_calls) == 2


@pytest.mark.unit
async def test_post_tool_hooks_fire_for_different_call_ids() -> None:
    """Given same tool_name but different tool_call_id, both post_tool hooks fire."""
    _reset_calls()
    hooks = AgentHooks(post_tool_use=[_make_recording_hook("post_tool_use")])
    run_ctx = AgentRunContext(session_id="test-session")
    host = _make_host(hooks=hooks, run_ctx=run_ctx)

    await host._fire_post_tool_hooks("tool", {}, "out1", 1.0, "call-A")
    await host._fire_post_tool_hooks("tool", {}, "out2", 2.0, "call-B")

    assert len(hook_calls) == 2


# ---------------------------------------------------------------------------
# Test: pre_turn deny returns deny result
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_pre_turn_deny_returns_deny_result() -> None:
    """Given a denying pre_turn hook, _fire_pre_turn_hooks returns deny."""
    _reset_calls()
    hooks = AgentHooks(pre_turn=[_make_deny_hook("pre_turn")])
    host = _make_host(hooks=hooks)

    result = await host._fire_pre_turn_hooks()

    assert result is not None
    assert result.get("decision") == "deny"
