"""Streaming adapter mapping pydantic-graph ``Graph.iter()`` yields to AgentPool events.

This module bridges pydantic-graph's execution model with AgentPool's existing
``RichAgentStreamEvent`` types. It wraps a ``GraphRun`` iterator and maps:

- ``Sequence[GraphTask]`` yields → ``PartStartEvent`` (step beginning)
- Step-internal streaming chunks → ``PartDeltaEvent`` (via event collector)
- Tool call invocations → ``ToolCallStartEvent`` + ``ToolCallCompleteEvent`` (via event collector)
- ``EndMarker`` → ``StreamCompleteEvent``
- ``ErrorMarker`` → ``RunErrorEvent`` (then re-raised)

Nested streaming (e.g. a step that runs a sub-agent which itself streams) is
handled by wrapping sub-agent events in ``SubAgentEvent`` or flattening them
based on configuration.

Usage:
    async with graph.iter(...) as graph_run:
    adapter = GraphStreamingAdapter(
            graph_run,
            session_id=session_id,
            agent_name="my_agent",
        )
        async for event in adapter:
            match event:
                case PartStartEvent():
                    print(f"Step starting: {event.part}")
                case StreamCompleteEvent():
                    print(f"Done: {event.message.content}")
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from typing import Any, final
from uuid import uuid4

from pydantic_graph.graph_builder import EndMarker, ErrorMarker, GraphRun

from wolfharness.agents.events import (
    PartDeltaEvent,
    PartStartEvent,
    RichAgentStreamEvent,
    RunErrorEvent,
    RunStartedEvent,
    StreamCompleteEvent,
    SubAgentEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.log import get_logger
from wolfharness.messaging.messages import ChatMessage


logger = get_logger(__name__)


@final
class StepEventCollector:
    """Collects step-internal events (streaming chunks, tool calls) during graph execution.

    Steps that stream or call tools emit events to a collector instance. The
    :class:`GraphStreamingAdapter` consumes these events and interleaves them
    with GraphRun-level yields in the correct order.

    Args:
        adapter: The parent adapter that will consume collected events.
        step_name: Human-readable name of the step producing events.
        depth: Nesting depth for sub-agent events (1 = direct child).
    """

    def __init__(
        self,
        adapter: GraphStreamingAdapter,
        *,
        step_name: str,
        depth: int = 0,
    ) -> None:
        self._adapter = adapter
        self.step_name = step_name
        self.depth = depth

    async def emit(self, event: RichAgentStreamEvent[Any]) -> None:
        """Emit a step-internal event into the adapter's event queue.

        When *depth* is greater than 0, events are wrapped in ``SubAgentEvent``
        so consumers can render nested activity appropriately. When *depth* is
        zero, events are yielded directly (flattened).

        Args:
            event: The streaming event to forward.
        """
        if self.depth > 0:
            wrapped = SubAgentEvent(
                source_name=self.step_name,
                source_type="agent",
                event=event,
                depth=self.depth,
                child_session_id=self._adapter.session_id,
                parent_session_id=self._adapter.session_id,
            )
            await self._adapter._event_queue.put(wrapped)
        else:
            await self._adapter._event_queue.put(event)

    async def emit_text_delta(self, index: int, content: str) -> None:
        """Convenience helper to emit a text ``PartDeltaEvent``."""
        await self.emit(PartDeltaEvent.text(index=index, content=content))

    async def emit_tool_start(
        self,
        tool_call_id: str,
        tool_name: str,
        title: str,
        raw_input: dict[str, Any] | None = None,
    ) -> None:
        """Convenience helper to emit a ``ToolCallStartEvent``."""
        await self.emit(
            ToolCallStartEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                title=title,
                raw_input=raw_input or {},
            )
        )

    async def emit_tool_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: Any,
    ) -> None:
        """Convenience helper to emit a ``ToolCallCompleteEvent``."""
        await self.emit(
            ToolCallCompleteEvent(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input,
                tool_result=tool_result,
                agent_name=self._adapter.agent_name,
                message_id=self._adapter.message_id,
            )
        )


@final
class GraphStreamingAdapter:
    """Adapts a ``GraphRun`` iterator to AgentPool ``RichAgentStreamEvent`` types.

    The adapter runs graph iteration in a background task and feeds events
    through an async queue. This ensures that:

    1. Step-internal events (streaming chunks, tool calls) can be emitted
       concurrently with GraphRun yields.
    2. Consumers always see events in the order they occurred.
    3. Cancellation and error propagation follow the same pattern as the
       existing ``Agent._stream_events`` implementation.

    Args:
        graph_run: The pydantic-graph run to adapt.
        session_id: Session identifier injected into all emitted events.
        agent_name: Name of the agent running the graph.
        message_id: Optional message ID (generated if omitted).
        run_id: Optional run ID (generated if omitted).
        user_msg: Optional user message that triggered this run. Used to
            set ``parent_id`` on the final ``StreamCompleteEvent``.
        flatten_depth: Maximum nesting depth to flatten before wrapping in
            ``SubAgentEvent``. ``0`` means never wrap (always flatten).
    """

    def __init__(
        self,
        graph_run: GraphRun,
        *,
        session_id: str,
        agent_name: str,
        message_id: str | None = None,
        run_id: str | None = None,
        user_msg: ChatMessage[Any] | None = None,
        flatten_depth: int = 0,
    ) -> None:
        self.graph_run = graph_run
        self.session_id = session_id
        self.agent_name = agent_name
        self.message_id = message_id or str(uuid4())
        self.run_id = run_id or str(uuid4())
        self.user_msg = user_msg
        self.flatten_depth = flatten_depth

        self._event_queue: asyncio.Queue[RichAgentStreamEvent[Any] | None] = asyncio.Queue()
        self._iteration_done = asyncio.Event()
        self._iteration_error: BaseException | None = None
        self._final_value: Any | None = None

    def create_collector(self, step_name: str, depth: int = 0) -> StepEventCollector:
        """Create an event collector for a step.

        Steps that produce internal events (streaming text, tool calls) should
        create a collector and emit events through it.

        Args:
            step_name: Human-readable name of the step.
            depth: Nesting depth. Use ``> 0`` when the step delegates to a
                sub-agent so events are wrapped in ``SubAgentEvent``.

        Returns:
            A collector bound to this adapter.
        """
        return StepEventCollector(self, step_name=step_name, depth=depth)

    async def _graph_iteration_task(self) -> None:
        """Background task that consumes the GraphRun and enqueues mapped events."""
        try:
            async for yield_item in self.graph_run:
                match yield_item:
                    case Sequence() as tasks:
                        for i, task in enumerate(tasks):
                            await self._event_queue.put(
                                PartStartEvent.text(
                                    index=i,
                                    content=f"Starting step {task.node_id}",  # ty: ignore[unresolved-attribute]
                                )
                            )
                    case EndMarker() as end_marker:
                        self._final_value = end_marker.value
                        break
                    case ErrorMarker() as error_marker:
                        await self._event_queue.put(
                            RunErrorEvent(
                                message=str(error_marker.error),
                                agent_name=self.agent_name,
                                run_id=self.run_id,
                            )
                        )
                        raise error_marker.error  # noqa: TRY301
        except asyncio.CancelledError:
            logger.debug("Graph iteration task cancelled")
        except BaseException as exc:  # noqa: BLE001
            self._iteration_error = exc
        finally:
            await self._event_queue.put(None)

    async def __aiter__(self) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Yield ``RichAgentStreamEvent`` mapped from GraphRun yields.

        Yields:
            Events in the order they occurred:
            1. ``RunStartedEvent``
            2. ``PartStartEvent`` for each ``GraphTask``
            3. Step-internal events (``PartDeltaEvent``, ``ToolCallStartEvent``,
               ``ToolCallCompleteEvent``, etc.)
            4. ``StreamCompleteEvent`` when the graph finishes
        """
        yield RunStartedEvent(
            session_id=self.session_id,
            run_id=self.run_id,
            agent_name=self.agent_name,
        )

        # Producer runs as a background task; consumer yields events
        # in the main coroutine. This avoids cancel-scope boundary
        # issues with generator cleanup (aclose/GC) while still
        # delivering events in real-time (not batched).

        async def _producer() -> None:
            await self._graph_iteration_task()

        producer_task = asyncio.ensure_future(_producer())
        try:
            try:
                while True:
                    event = await self._event_queue.get()
                    if event is None:
                        break
                    yield event
                    if isinstance(event, RunErrorEvent):
                        break
                    if isinstance(event, StreamCompleteEvent):
                        return
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await producer_task
                else:
                    with suppress(asyncio.CancelledError):
                        await producer_task

            if self._iteration_error is not None:
                raise self._iteration_error

            response_msg = ChatMessage(
                content=str(self._final_value) if self._final_value is not None else "",
                role="assistant",
                name=self.agent_name,
                message_id=self.message_id,
                session_id=self.session_id,
                parent_id=self.user_msg.message_id if self.user_msg else None,
            )
            yield StreamCompleteEvent(message=response_msg)
        finally:
            self._iteration_done.set()


async def adapt_graph_run(
    graph_run: GraphRun,
    *,
    session_id: str,
    agent_name: str,
    message_id: str | None = None,
    run_id: str | None = None,
    user_msg: ChatMessage[Any] | None = None,
    flatten_depth: int = 0,
) -> AsyncIterator[RichAgentStreamEvent[Any]]:
    """Convenience function to adapt a GraphRun without managing the adapter lifetime.

    This is a thin wrapper around :class:`GraphStreamingAdapter` that yields
    events directly. Use it when you don't need to create step event collectors
    manually.

    Args:
        graph_run: The pydantic-graph run to adapt.
        session_id: Session identifier.
        agent_name: Name of the agent.
        message_id: Optional message ID.
        run_id: Optional run ID.
        user_msg: Optional triggering user message.
        flatten_depth: Nesting depth threshold for ``SubAgentEvent`` wrapping.

    Yields:
        ``RichAgentStreamEvent`` mapped from GraphRun yields.
    """
    adapter = GraphStreamingAdapter(
        graph_run,
        session_id=session_id,
        agent_name=agent_name,
        message_id=message_id,
        run_id=run_id,
        user_msg=user_msg,
        flatten_depth=flatten_depth,
    )
    async for event in adapter:
        yield event
