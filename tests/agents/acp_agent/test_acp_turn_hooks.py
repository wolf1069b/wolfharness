"""Integration tests for ACPTurn with hooks.

Verifies that HookAwareTurn's hooks fire during ACPTurn.execute(),
including advisory tool hooks during streaming and permission blocking
via pre_turn deny.
"""

from __future__ import annotations

from typing import Any

import pytest

from acp.schema import (
    AgentMessageChunk,
    PromptResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    TurnCompleteUpdate,
)
from wolfharness.agents.acp_agent.turn import ACPTurn
from wolfharness.agents.context import AgentRunContext
from wolfharness.agents.events import (
    StreamCompleteEvent,
)
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
# Fake ACP client
# ---------------------------------------------------------------------------


class FakeACPClient:
    """Fake ACP client implementing ACPClientProtocol for testing."""

    def __init__(
        self,
        *,
        updates: list[Any] | None = None,
        messages: list[Any] | None = None,
    ) -> None:
        self._updates = updates or []
        self._messages = messages or []
        self.prompt_calls: list[tuple[str, list[Any]]] = []

    async def prompt(self, session_id: str, content: list[Any]) -> PromptResponse:
        self.prompt_calls.append((session_id, content))
        return PromptResponse(stop_reason="end_turn")

    async def stream_events(self, response: PromptResponse) -> Any:
        for update in self._updates:
            yield update

    async def get_messages(self, session_id: str) -> list[Any]:
        return list(self._messages)


def _text_update(text: str) -> AgentMessageChunk:
    return AgentMessageChunk(content=TextContentBlock(text=text))


def _tool_call_start(
    tool_call_id: str = "tc-1",
    title: str = "read_file",
) -> ToolCallStart:
    return ToolCallStart(
        tool_call_id=tool_call_id,
        title=title,
    )


def _tool_call_complete(
    tool_call_id: str = "tc-1",
    title: str = "read_file",
    raw_output: Any = "file contents",
) -> ToolCallProgress:
    return ToolCallProgress(
        tool_call_id=tool_call_id,
        title=title,
        status="completed",
        raw_output=raw_output,
    )


def _make_run_ctx() -> AgentRunContext:
    return AgentRunContext(session_id="test-acp-session")


def _make_turn(
    hooks: AgentHooks | None = None,
    *,
    updates: list[Any] | None = None,
    messages: list[Any] | None = None,
) -> tuple[ACPTurn, FakeACPClient]:
    client = FakeACPClient(updates=updates, messages=messages)
    turn = ACPTurn(
        acp_client=client,  # type: ignore[arg-type]
        prompts=["do something"],
        run_ctx=_make_run_ctx(),
        session_id="test-acp-session",
        agent_name="test-acp-agent",
        hooks=hooks,
    )
    return turn, client


# ---------------------------------------------------------------------------
# Test: hooks fire during ACP turn execution
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_pre_turn_and_post_turn_fire_during_acp_turn() -> None:
    """Given an ACPTurn with hooks, pre_turn and post_turn fire during execute()."""
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_recorder("pre_turn")],
        post_turn=[_make_recorder("post_turn")],
    )
    updates = [_text_update("Hello"), TurnCompleteUpdate()]
    messages = [_text_update("Hello")]
    turn, _ = _make_turn(hooks=hooks, updates=updates, messages=messages)

    events = [event async for event in turn.execute()]

    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names
    assert event_names.index("pre_turn") < event_names.index("post_turn")
    assert any(isinstance(e, StreamCompleteEvent) for e in events)


# ---------------------------------------------------------------------------
# Test: pre_turn deny blocks the turn
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_pre_turn_deny_blocks_acp_turn() -> None:
    """Given a denying pre_turn hook, ACPTurn cancels and post_turn still fires.

    pre_turn deny yields StreamCompleteEvent(cancelled=True) and returns.
    The return hits the finally block, so post_turn hooks must fire.
    """
    _reset_calls()
    hooks = AgentHooks(
        pre_turn=[_make_denyer("pre_turn")],
        post_turn=[_make_recorder("post_turn")],
    )
    turn, client = _make_turn(hooks=hooks)

    events = [event async for event in turn.execute()]

    # Client.prompt should never be called because pre_turn denied
    assert len(client.prompt_calls) == 0

    stream_complete = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert len(stream_complete) == 1
    assert stream_complete[0].cancelled is True

    # post_turn must fire in the finally block even on deny
    event_names = [name for name, _ in hook_calls]
    assert "pre_turn" in event_names
    assert "post_turn" in event_names


# ---------------------------------------------------------------------------
# Test: advisory tool hooks fire during streaming
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_advisory_pre_tool_hook_fires_on_tool_call_start() -> None:
    """Given an ACPTurn with pre_tool_use hooks, they fire when ToolCallStartEvent is yielded."""
    _reset_calls()
    hooks = AgentHooks(pre_tool_use=[_make_recorder("pre_tool_use")])
    updates = [
        _tool_call_start(tool_call_id="tc-1", title="read_file"),
        _tool_call_complete(tool_call_id="tc-1", title="read_file"),
        TurnCompleteUpdate(),
    ]
    messages = [_text_update("done")]
    turn, _ = _make_turn(hooks=hooks, updates=updates, messages=messages)

    _ = [event async for event in turn.execute()]

    event_names = [name for name, _ in hook_calls]
    assert "pre_tool_use" in event_names


@pytest.mark.integration
async def test_advisory_post_tool_hook_fires_on_tool_call_complete() -> None:
    """Given an ACPTurn with post_tool_use hooks, they fire on ToolCallCompleteEvent."""
    _reset_calls()
    hooks = AgentHooks(post_tool_use=[_make_recorder("post_tool_use")])
    updates = [
        _tool_call_start(tool_call_id="tc-1", title="read_file"),
        _tool_call_complete(tool_call_id="tc-1", title="read_file"),
        TurnCompleteUpdate(),
    ]
    messages = [_text_update("done")]
    turn, _ = _make_turn(hooks=hooks, updates=updates, messages=messages)

    _ = [event async for event in turn.execute()]

    event_names = [name for name, _ in hook_calls]
    assert "post_tool_use" in event_names


# ---------------------------------------------------------------------------
# Test: no hooks configured → no-op
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_acp_turn_no_hooks_no_exception() -> None:
    """Given an ACPTurn with hooks=None, execute() completes without raising."""
    updates = [_text_update("Hello"), TurnCompleteUpdate()]
    messages = [_text_update("Hello")]
    turn, _ = _make_turn(hooks=None, updates=updates, messages=messages)

    events = [event async for event in turn.execute()]

    assert any(isinstance(e, StreamCompleteEvent) for e in events)
