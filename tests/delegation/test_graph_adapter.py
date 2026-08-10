"""Tests for the MessageNode to pydantic-graph Step adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness.messaging.graph_adapter import AgentPoolState, MessageNodeStep
from wolfharness.messaging.messagenode import MessageNode


pytestmark = pytest.mark.unit


class GraphMessageNode(MessageNode[Any, str]):
    """A concrete MessageNode that uses graph-based execution.

    This node implements :meth:`_execute_node` instead of overriding
    :meth:`run`, so it exercises the pydantic-graph adapter path.
    """

    async def _execute_node(self, *prompts: Any, **kwargs: Any) -> ChatMessage[str]:
        content = " ".join(str(p) for p in prompts) if prompts else "empty"
        return ChatMessage(content=content, role="assistant")

    async def get_stats(self) -> Any:
        pass

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass


class GraphMessageNodeWithEvents(MessageNode[Any, str]):
    """A node that pushes events to the state event queue during execution."""

    async def _execute_node(self, *prompts: Any, **kwargs: Any) -> ChatMessage[str]:
        from pydantic_ai import TextPartDelta

        from wolfharness.agents.events import PartDeltaEvent

        state = kwargs.get("_state")
        if state is not None:
            # Push a fake event to the queue
            event = PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="chunk1"))
            await state.event_queue.put(event)

        content = " ".join(str(p) for p in prompts) if prompts else "empty"
        return ChatMessage(content=content, role="assistant")

    async def get_stats(self) -> Any:
        pass

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        pass


@pytest.mark.asyncio
async def test_message_node_step_builds_graph():
    """MessageNodeStep.build_single_node_graph returns an executable Graph."""
    node = GraphMessageNode(name="test_step")
    step_wrapper = MessageNodeStep(node)
    graph = step_wrapper.build_single_node_graph()

    assert graph is not None


@pytest.mark.asyncio
async def test_message_node_step_runs_via_graph():
    """MessageNodeStep graph execution delegates to node.run()."""
    node = GraphMessageNode(name="test_run")
    step_wrapper = MessageNodeStep(node)
    graph = step_wrapper.build_single_node_graph()

    state = AgentPoolState(node=node, prompts=("hello",))
    result = await graph.run(state=state, deps=None, inputs=None)

    assert isinstance(result, ChatMessage)
    assert result.content == "hello"
    assert state.result is result


@pytest.mark.asyncio
async def test_message_node_run_uses_graph():
    """MessageNode.run() builds a single-node graph and executes via Graph.run()."""
    node = GraphMessageNode(name="test_node_run")
    result = await node.run("world")

    assert isinstance(result, ChatMessage)
    assert result.content == "world"


@pytest.mark.asyncio
async def test_message_node_run_stream_uses_graph_iter():
    """MessageNode.run_stream() drives execution via Graph.iter()."""
    node = GraphMessageNode(name="test_stream")
    events = [event async for event in node.run_stream("stream_test")]

    assert len(events) == 1
    from wolfharness.agents.events import StreamCompleteEvent

    assert isinstance(events[0], StreamCompleteEvent)
    assert events[0].message.content == "stream_test"


@pytest.mark.asyncio
async def test_message_node_run_stream_drains_event_queue():
    """run_stream drains events from AgentPoolState.event_queue."""
    node = GraphMessageNodeWithEvents(name="test_events")
    events = [event async for event in node.run_stream("event_test")]

    from wolfharness.agents.events import PartDeltaEvent, StreamCompleteEvent

    assert len(events) == 2
    assert isinstance(events[0], PartDeltaEvent)
    assert isinstance(events[1], StreamCompleteEvent)
    assert events[1].message.content == "event_test"


@pytest.mark.asyncio
async def test_message_node_signals_emitted():
    """message_received and message_sent signals are emitted during graph run."""
    node = GraphMessageNode(name="test_signals")

    received_handler = AsyncMock()
    sent_handler = AsyncMock()

    node.message_received.connect(received_handler)
    node.message_sent.connect(sent_handler)

    await node.run("signal_test")

    received_handler.assert_awaited_once()
    sent_handler.assert_awaited_once()

    # Verify the sent signal got the result message
    sent_call_args = sent_handler.call_args[0][0]
    assert sent_call_args.content == "signal_test"


@pytest.mark.asyncio
async def test_message_node_run_message():
    """run_message extracts content and passes through the graph."""
    node = GraphMessageNode(name="test_run_message")
    msg = ChatMessage.user_prompt(message="hello via message")
    result = await node.run_message(msg)

    assert result.content == "hello via message"


@pytest.mark.asyncio
async def test_message_node_step_preserves_kwargs():
    """MessageNodeStep passes kwargs through to node.run()."""
    node = GraphMessageNode(name="test_kwargs")
    step_wrapper = MessageNodeStep(node)
    graph = step_wrapper.build_single_node_graph()

    state = AgentPoolState(node=node, prompts=("prompt",), kwargs={"custom": "value"})
    result = await graph.run(state=state, deps=None, inputs=None)

    assert result.content == "prompt"


@pytest.mark.asyncio
async def test_existing_subclass_overrides_run_still_works():
    """Subclasses that override run() are not affected by the graph adapter."""

    class LegacyNode(MessageNode[Any, str]):
        async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[str]:
            return ChatMessage(content="legacy", role="assistant")

        async def get_stats(self) -> Any:
            pass

        def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
            pass

    node = LegacyNode(name="legacy")
    result = await node.run("anything")
    assert result.content == "legacy"
