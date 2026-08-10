"""16-cell smoke test matrix for the unified hook system.

Tests {pre_turn, post_turn, pre_tool_use, post_tool_use} x
{native standalone, native SessionPool, ACP standalone, ACP SessionPool}.

Each cell verifies the corresponding hook type fires in the corresponding mode.

ACP SessionPool cells are skipped because the full ACP+SessionPool setup
requires a real ACP subprocess, which is too heavy for a smoke test.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic_ai.models.test import TestModel
import pytest

from acp.schema import (
    AgentMessageChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    TurnCompleteUpdate,
)
from wolfharness import Agent
from wolfharness.agents.acp_agent.turn import ACPTurn
from wolfharness.agents.context import AgentRunContext
from wolfharness.hooks import AgentHooks, CallableHook, HookResult
from wolfharness.lifecycle import DirectChannel, MemoryJournal
from wolfharness.orchestrator.core import EventBus, SessionState
from wolfharness.orchestrator.run import RunHandle


# ---------------------------------------------------------------------------
# Hook type and mode types
# ---------------------------------------------------------------------------

HookType = Literal["pre_turn", "post_turn", "pre_tool_use", "post_tool_use"]
Mode = Literal["native_standalone", "native_sessionpool", "acp_standalone", "acp_sessionpool"]


# ---------------------------------------------------------------------------
# Shared hook call tracking
# ---------------------------------------------------------------------------

hook_calls: list[str] = []


def _reset_calls() -> None:
    hook_calls.clear()


def _make_recorder(event: str) -> CallableHook:
    def _fn(**kwargs: Any) -> HookResult:
        hook_calls.append(event)
        return {"decision": "allow"}

    return CallableHook(event=event, fn=_fn)  # type: ignore[arg-type]


def _make_hooks_for(hook_type: HookType) -> AgentHooks:
    """Create AgentHooks with only the specified hook type configured."""
    kwargs: dict[str, Any] = {
        "pre_turn": [],
        "post_turn": [],
        "pre_tool_use": [],
        "post_tool_use": [],
    }
    kwargs[hook_type] = [_make_recorder(hook_type)]
    return AgentHooks(**kwargs)


# ---------------------------------------------------------------------------
# Native standalone helper
# ---------------------------------------------------------------------------


async def _run_native_standalone(hook_type: HookType) -> None:
    """Run a native agent standalone with the given hook type."""
    _reset_calls()
    hooks = _make_hooks_for(hook_type)

    def simple_tool() -> str:
        """A simple tool."""
        return "tool_result"

    agent = Agent(
        name="smoke-native-standalone",
        model=TestModel(
            call_tools=["simple_tool"] if "tool" in hook_type else None,
            custom_output_text="response",
        ),
        tools=[simple_tool] if "tool" in hook_type else None,
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="smoke-session")
        from wolfharness.agents.native_agent.turn import NativeTurn

        turn = NativeTurn(
            agent=agent,
            prompts=["hello"],
            run_ctx=run_ctx,
            message_history=[],
            hooks=hooks,
        )
        _ = [event async for event in turn.execute()]


# ---------------------------------------------------------------------------
# Native SessionPool helper
# ---------------------------------------------------------------------------


async def _run_native_sessionpool(hook_type: HookType) -> None:
    """Run a native agent through the SessionPool (RunHandle.start()) path."""
    import asyncio as _asyncio

    _reset_calls()
    hooks = _make_hooks_for(hook_type)

    def simple_tool() -> str:
        """A simple tool."""
        return "tool_result"

    agent = Agent(
        name="smoke-native-pool",
        model=TestModel(
            call_tools=["simple_tool"] if "tool" in hook_type else None,
            custom_output_text="response",
        ),
        tools=[simple_tool] if "tool" in hook_type else None,
        hooks=hooks,
    )
    async with agent:
        run_ctx = AgentRunContext(session_id="smoke-pool-session")
        event_bus = EventBus()
        session = SessionState(session_id="smoke-pool-session", agent_name="smoke-native-pool")
        session._comm_channel = DirectChannel(MemoryJournal())
        handle = RunHandle(
            run_id="smoke-run",
            session_id="smoke-pool-session",
            agent_type="native",
            agent=agent,
            event_bus=event_bus,
            session=session,
            run_ctx=run_ctx,
        )

        gen = handle.start("hello")

        async def _consume() -> None:
            async for _ in gen:
                pass

        consumer_task = _asyncio.create_task(_consume())
        await _asyncio.sleep(0.1)

        # Close to unblock the idle wait after the turn completes
        handle.close()
        await _asyncio.sleep(0.1)
        await consumer_task

    # Force cleanup of suspended async generators so NativeTurn.execute()'s
    # finally block runs and fires post_turn hooks.
    loop = _asyncio.get_running_loop()
    await loop.shutdown_asyncgens()


# ---------------------------------------------------------------------------
# ACP standalone helper
# ---------------------------------------------------------------------------


class _FakeACPClient:
    """Minimal fake ACP client for smoke tests."""

    def __init__(self, updates: list[Any], messages: list[Any]) -> None:
        self._updates = updates
        self._messages = messages

    async def prompt(self, session_id: str, content: list[Any]) -> Any:
        from acp.schema import PromptResponse

        return PromptResponse(stop_reason="end_turn")

    async def stream_events(self, response: Any) -> Any:
        for update in self._updates:
            yield update

    async def get_messages(self, session_id: str) -> list[Any]:
        return list(self._messages)


async def _run_acp_standalone(hook_type: HookType) -> None:
    """Run an ACP agent standalone with the given hook type."""
    _reset_calls()
    hooks = _make_hooks_for(hook_type)

    updates: list[Any] = []
    if hook_type in ("pre_tool_use", "post_tool_use"):
        updates.extend([
            ToolCallStart(tool_call_id="tc-1", title="read_file"),
            ToolCallProgress(
                tool_call_id="tc-1",
                title="read_file",
                status="completed",
                raw_output="result",
            ),
        ])
    updates.append(AgentMessageChunk(content=TextContentBlock(text="hello")))
    updates.append(TurnCompleteUpdate())

    messages = [AgentMessageChunk(content=TextContentBlock(text="hello"))]

    client = _FakeACPClient(updates=updates, messages=messages)
    run_ctx = AgentRunContext(session_id="smoke-acp-session")

    turn = ACPTurn(
        acp_client=client,  # type: ignore[arg-type]
        prompts=["do something"],
        run_ctx=run_ctx,
        session_id="smoke-acp-session",
        agent_name="smoke-acp-agent",
        hooks=hooks,
    )

    _ = [event async for event in turn.execute()]


# ---------------------------------------------------------------------------
# ACP SessionPool helper (skipped)
# ---------------------------------------------------------------------------


async def _run_acp_sessionpool(hook_type: HookType) -> None:
    """Run an ACP agent through SessionPool. Skipped — too complex for smoke test."""
    pytest.skip("ACP SessionPool requires real ACP subprocess — too heavy for smoke test")


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

_MODE_RUNNERS: dict[Mode, Any] = {
    "native_standalone": _run_native_standalone,
    "native_sessionpool": _run_native_sessionpool,
    "acp_standalone": _run_acp_standalone,
    "acp_sessionpool": _run_acp_sessionpool,
}


# ---------------------------------------------------------------------------
# 16-cell smoke test matrix
# ---------------------------------------------------------------------------

_HOOK_TYPES: list[HookType] = ["pre_turn", "post_turn", "pre_tool_use", "post_tool_use"]
_MODES: list[Mode] = [
    "native_standalone",
    "native_sessionpool",
    "acp_standalone",
    "acp_sessionpool",
]


@pytest.mark.integration
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("hook_type", _HOOK_TYPES)
async def test_hook_smoke_matrix(hook_type: HookType, mode: Mode) -> None:
    """Given each hook type x mode combination, the hook fires.

    This is the 16-cell smoke test matrix:
    {pre_turn, post_turn, pre_tool_use, post_tool_use} x
    {native standalone, native SessionPool, ACP standalone, ACP SessionPool}
    """
    runner = _MODE_RUNNERS[mode]
    await runner(hook_type)

    # ACP SessionPool is skipped — no assertions to check
    if mode == "acp_sessionpool":
        return

    assert hook_type in hook_calls, (
        f"Hook '{hook_type}' did not fire in mode '{mode}'. Fired hooks: {hook_calls}"
    )
