"""Stream processors for event pipelines.

This module provides composable processors that can transform, filter, or observe
event streams. Processors wrap AsyncIterators and can be chained together.

Example:
    ```python
    # Simple function processor
    async def log_events(stream):
        async for event in stream:
            print(f"Event: {type(event).__name__}")
            yield event

    # Compose into pipeline
    pipeline = StreamPipeline([log_events])

    async for event in pipeline(raw_events):
        yield event
    ```
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic_ai import (
    PartDeltaEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
)

from wolfharness.agents.events import ToolCallProgressEvent, ToolCallStartEvent


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine

    from wolfharness.agents.events.events import RichAgentStreamEvent


# Type alias for processor callables
type StreamProcessorCallable = Callable[
    [AsyncIterator[RichAgentStreamEvent[Any]]], AsyncIterator[RichAgentStreamEvent[Any]]
]


@runtime_checkable
class StreamProcessor(Protocol):
    """Protocol for stream processors.

    Processors can be:
    - Callables: `(AsyncIterator[RichAgentStreamEvent]) -> AsyncIterator[RichAgentStreamEvent]`
    - Classes with `__call__`: Same signature, but can hold state
    """

    def __call__(
        self, stream: AsyncIterator[RichAgentStreamEvent[Any]]
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Process an event stream.

        Args:
            stream: Input event stream

        Returns:
            Transformed/filtered event stream
        """
        ...


@dataclass
class StreamPipeline:
    """Composable pipeline for processing event streams.

    Chains multiple processors together, passing the output of each
    to the input of the next.

    Example:
        ```python
        tracker = FileTrackingProcessor()
        pipeline = StreamPipeline([
            tracker,
            event_handler_processor(handler),
        ])

        async for event in pipeline(raw_events):
            yield event

        # Access state directly from processor instances
        print(tracker.get_metadata())
        ```
    """

    processors: list[StreamProcessorCallable | StreamProcessor] = field(default_factory=list)

    def __call__(
        self, stream: AsyncIterator[RichAgentStreamEvent[Any]]
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Run events through all processors in sequence.

        Args:
            stream: Input event stream

        Returns:
            Processed event stream
        """
        result = stream
        for processor in self.processors:
            result = processor(result)
        return result

    def add(self, processor: StreamProcessorCallable | StreamProcessor) -> None:
        """Add a processor to the pipeline.

        Args:
            processor: Processor to add
        """
        self.processors.append(processor)


def event_handler_processor(
    handler: Callable[[Any, RichAgentStreamEvent[Any]], Coroutine[Any, Any, None]],
) -> StreamProcessorCallable:
    """Create a processor that calls an event handler for each event.

    The handler is called with (None, event) to match the existing
    MultiEventHandler signature.

    Args:
        handler: Async callable with signature (ctx, event) -> None

    Returns:
        Processor function that calls the handler

    Example:
        ```python
        pipeline = StreamPipeline([
            event_handler_processor(self.event_handler),
        ])
        ```
    """

    async def process(
        stream: AsyncIterator[RichAgentStreamEvent[Any]],
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        async for event in stream:
            await handler(None, event)
            yield event

    return process


def event_to_part(
    event: RichAgentStreamEvent[Any],
) -> TextPart | ThinkingPart | ToolCallPart | ToolReturnPart | None:
    match event:
        case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
            return TextPart(content=delta)
        case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=delta)) if delta:
            return ThinkingPart(content=delta)
        case ToolCallStartEvent(tool_call_id=tc_id, tool_name=tc_name, raw_input=tc_input):
            return ToolCallPart(tool_name=tc_name, args=tc_input, tool_call_id=tc_id)
        case ToolCallProgressEvent(status="failed", tool_call_id=tc_id, title=tc_name, message=msg):
            return ToolReturnPart(
                tool_name=tc_name or "unknown",
                content=msg or "Tool execution failed",
                tool_call_id=tc_id,
                outcome="failed",
            )
        case _:
            return None
