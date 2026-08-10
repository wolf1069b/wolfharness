"""Red flag test: Tool-call-only model responses yield no text/thinking events.

When a model decides to call a tool (e.g., ``task``) without emitting any
preceding text, pydantic-ai does NOT yield ``PartDeltaEvent`` (text/thinking)
chunks.  The frontend therefore sees **zero** SSE events between
``RunStartedEvent`` and the tool execution result — the page stays blank.

If the tool then fails (or returns an error string), the only visible outcome
is a ``StreamCompleteEvent`` with the error text, but the user never saw any
progress indicators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import PartDeltaEvent as PyAIPartDeltaEvent
from pydantic_ai.models.test import TestModel
import pytest

from wolfharness import AgentPool, AgentsManifest, NativeAgentConfig
from wolfharness.agents.base_agent import _in_turn_context
from wolfharness.agents.events import (
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.tools import Tool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect_events(stream: AsyncIterator[Any]) -> list[Any]:
    """Drain an async event stream into a list."""
    return [e async for e in stream]


def _failing_tool() -> str:
    """A tool that always returns an error string (simulating a failed delegation)."""
    return "Error: Agent 'general' not found. Available: worker, coder"


# ---------------------------------------------------------------------------
# Red flag tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_tool_call_only_response_has_no_text_deltas() -> None:
    """RED FLAG: Model that calls a tool yields NO text/thinking events.

    Scenario:
    1. User sends a prompt that triggers a tool call (e.g., ``task``).
    2. TestModel is configured to call the tool and produce NO custom text.
    3. Agent runs with streaming enabled.

    Expected (current behaviour):
    - ``RunStartedEvent`` is emitted immediately.
    - ``PartStartEvent`` with ``BaseToolCallPart`` is emitted (tool call start).
    - **NO** ``PartDeltaEvent`` (text or thinking) appears before the tool call.
    - ``FunctionToolCallEvent`` and ``FunctionToolResultEvent`` are emitted.
    - ``ToolCallCompleteEvent`` is emitted.
    - ``StreamCompleteEvent`` closes the stream.
    - **RED FLAG**: ``ToolCallStartEvent`` is NOT emitted when running outside
      SessionPool (run_ctx.event_bus is None). NativeAgent only maps tool call
      events to ToolCallStartEvent when event_bus is present.

    This explains the "blank page" symptom: the frontend has nothing to render
    until the tool result arrives.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent with a task tool",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        agent = pool.manifest.agents["test_agent"].get_agent(pool=pool)

        # Create a tool that simulates a failed task delegation
        failing_tool = Tool.from_callable(_failing_tool, name_override="failing_tool")
        agent._builtin_provider.register_tool(failing_tool)

        # Override model: call the tool, emit NO custom text
        await agent.set_model(
            TestModel(
                call_tools=["failing_tool"],
                custom_output_text=None,
            ),
        )

        # Bypass SessionPool so the shared agent (with our registered tool) runs directly.
        # Without this, run_stream() delegates to SessionPool which creates a per-session
        # agent from the manifest config that lacks our dynamically registered tool.
        _in_turn_context.set(True)
        events = await _collect_events(agent.run_stream("delegate to general"))

    # Categorise events for analysis
    event_types = [type(e).__name__ for e in events]
    # Native agent emits pydantic-ai PartDeltaEvent directly (not wolfharness's subclass)
    text_deltas = [e for e in events if isinstance(e, PyAIPartDeltaEvent)]
    tool_call_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    tool_call_completes = [e for e in events if isinstance(e, ToolCallCompleteEvent)]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    print(f"\nEvents emitted: {event_types}")
    print(f"Text deltas: {len(text_deltas)}")
    print(f"ToolCallStartEvent: {len(tool_call_starts)}")
    print(f"ToolCallCompleteEvent: {len(tool_call_completes)}")
    print(f"StreamCompleteEvent: {len(stream_completes)}")

    # Baseline: stream completes (RunStartedEvent is published by
    # RunHandle.start() to EventBus, not yielded in standalone stream)
    assert len(stream_completes) >= 0, "Stream should produce events"

    # RED FLAG: there are NO text/thinking deltas BEFORE the tool call completes
    # Find index of first ToolCallCompleteEvent
    first_tool_complete_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, ToolCallCompleteEvent)),
        len(events),
    )
    text_deltas_before_tool_complete = [
        e for e in events[:first_tool_complete_idx] if isinstance(e, PyAIPartDeltaEvent)
    ]
    assert len(text_deltas_before_tool_complete) == 0, (
        f"RED FLAG: Expected 0 text/thinking deltas before tool call completes, "
        f"got {len(text_deltas_before_tool_complete)}."
        " Frontend has nothing to render until tool completes."
    )

    # Tool call lifecycle: native agent emits ToolCallStartEvent for
    # FunctionToolCallEvent / PartStartEvent with BaseToolCallPart, even
    # when running outside SessionPool.
    assert len(tool_call_starts) == 1, (
        "ToolCallStartEvent should be emitted for tool-call-only responses."
    )
    assert len(tool_call_completes) >= 1, "ToolCallCompleteEvent should be emitted"

    # Stream closes normally
    assert len(stream_completes) == 1, "Exactly one StreamCompleteEvent expected"

    # The final message contains the tool result (error string)
    final_msg = stream_completes[0].message
    assert final_msg is not None
    assert "Error: Agent 'general' not found" in str(final_msg.content), (
        f"Final message should contain the tool error, got: {final_msg.content!r}"
    )


@pytest.mark.integration
async def test_tool_error_does_not_break_stream() -> None:
    """RED FLAG: Even when the tool returns an error string, the stream must complete.

    This verifies the fix that changed ``raise ToolError`` to ``return`` error
    strings inside ``background_task_provider.task()``.  When a tool returns an
    error string (instead of raising), pydantic-ai treats it as a normal tool
    result and the stream completes cleanly.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        agent = pool.manifest.agents["test_agent"].get_agent(pool=pool)

        # Register a tool that RETURNS an error string (the FIXED behaviour)
        def _broken_tool() -> str:
            return "Error: Agent 'general' not found"

        broken_tool = Tool.from_callable(_broken_tool, name_override="broken_tool")
        agent._builtin_provider.register_tool(broken_tool)

        await agent.set_model(
            TestModel(
                call_tools=["broken_tool"],
                custom_output_text=None,
            ),
        )

        # Bypass SessionPool so the shared agent (with our registered tool) runs directly.
        _in_turn_context.set(True)
        events = await _collect_events(agent.run_stream("trigger broken tool"))

    event_types = [type(e).__name__ for e in events]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    print(f"\nEvents with broken tool: {event_types}")

    # When the tool returns an error string (not raises), the stream completes.
    assert len(stream_completes) == 1, (
        f"Stream must complete when tool returns error string. Got events: {event_types}"
    )


@pytest.mark.integration
async def test_text_response_yields_deltas() -> None:
    """Baseline: a normal text response DOES yield PartDeltaEvents.

    This proves the absence of text deltas in the tool-only case is specific
    to tool-call responses, not a general streaming failure.
    """
    agent_config = NativeAgentConfig(
        name="test_agent",
        model="test",
        system_prompt="You are a test agent",
    )
    manifest = AgentsManifest(agents={"test_agent": agent_config})

    async with AgentPool(manifest) as pool:
        agent = pool.manifest.agents["test_agent"].get_agent(pool=pool)

        # Normal text response — avoid built-in tools so we get immediate text
        await agent.set_model(
            TestModel(call_tools=[], custom_output_text="Hello from model"),
        )

        # Bypass SessionPool so the shared agent (with our TestModel override) runs directly.
        _in_turn_context.set(True)
        events = await _collect_events(agent.run_stream("say hello"))

    event_types = [type(e).__name__ for e in events]
    print(f"\nEvents with text response: {event_types}")

    # Native agent emits pydantic-ai PartDeltaEvent directly (not wolfharness's subclass)
    text_deltas = [e for e in events if isinstance(e, PyAIPartDeltaEvent)]
    stream_completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    assert len(text_deltas) > 0, "Text response should yield PartDeltaEvents"
    assert len(stream_completes) == 1
    assert "Hello from model" in str(stream_completes[0].message.content)
