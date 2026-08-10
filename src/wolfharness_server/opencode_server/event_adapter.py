"""Event adapter for OpenCode server.

Provides a clean adapter interface for converting AgentPool events to OpenCode SSE
events. This module wraps the existing event conversion logic from
:mod:`event_processor` and :mod:`stream_adapter`, exposing it through a simple,
discoverable API.

**Event Mapping**

+-----------------------+---------------------------------------------+
| AgentPool Event       | OpenCode Event(s)                           |
+=======================+=============================================+
| ``PartStartEvent``    | ``PartUpdatedEvent`` (TextPart)             |
|                       | ``PartUpdatedEvent`` (ReasoningPart)        |
+-----------------------+---------------------------------------------+
| ``PartDeltaEvent``    | ``PartDeltaEvent`` (text delta)             |
|                       | ``PartDeltaEvent`` (reasoning delta)        |
+-----------------------+---------------------------------------------+
| ``PartEndEvent``      | Completion signal (handled internally)      |
+-----------------------+---------------------------------------------+
| ``ToolCallStartEvent``| ``PartUpdatedEvent`` (ToolPart, running)    |
+-----------------------+---------------------------------------------+
| ``ToolCallComplete``  | ``PartUpdatedEvent`` (ToolPart, completed)  |
|                       | ``PartUpdatedEvent`` (ToolPart, error)      |
+-----------------------+---------------------------------------------+
| ``StreamCompleteEvent``| ``PartUpdatedEvent`` (StepFinishPart)      |
|                       | ``SessionIdleEvent``                        |
+-----------------------+---------------------------------------------+
| ``RunStartedEvent``   | ``SessionStatusEvent`` (busy)               |
+-----------------------+---------------------------------------------+
| ``RunErrorEvent``     | ``SessionErrorEvent``                       |
+-----------------------+---------------------------------------------+

Usage::

    from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter

    adapter = OpenCodeEventAdapter.from_stream_adapter(stream_adapter)
    async for oc_event in adapter.convert_stream(agent_stream):
        ...

    # Or with an existing context:
    adapter = OpenCodeEventAdapter(ctx)
    async for oc_event in adapter.convert_event(agent_event):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,  # noqa: TC001
)
from wolfharness_server.opencode_server.stream_adapter import (  # noqa: TC001
    OpenCodeStreamAdapter,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.agents.events.events import RichAgentStreamEvent
    from wolfharness_server.opencode_server.models.events import Event


class OpenCodeEventAdapter:
    """Clean adapter interface for converting AgentPool events to OpenCode SSE events.

    Wraps :class:`EventProcessor` and :class:`EventProcessorContext` to provide a
    single, discoverable entry point for event conversion.  The adapter does **not**
    duplicate conversion logic; it delegates all heavy lifting to the existing
    processor.

    Args:
        context: The mutable event processor context.  The adapter borrows a
            reference — the caller is responsible for the context lifecycle.
    """

    def __init__(
        self,
        context: EventProcessorContext,
    ) -> None:
        """Initialize the adapter with an existing context.

        Args:
            context: The mutable event processor context.  The adapter borrows a
                reference — the caller is responsible for the context lifecycle.
        """
        self._context = context
        self._processor = EventProcessor()

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_stream_adapter(cls, stream_adapter: OpenCodeStreamAdapter) -> OpenCodeEventAdapter:
        """Create an adapter from an existing :class:`OpenCodeStreamAdapter`.

        This is the preferred constructor when the caller already has a stream
        adapter (e.g. in message routes).

        Args:
            stream_adapter: The stream adapter whose main context will be used.

        Returns:
            A new :class:`OpenCodeEventAdapter` backed by the stream adapter's
            main context.
        """
        return cls(stream_adapter.main_context)

    # ------------------------------------------------------------------
    # Single-event conversion
    # ------------------------------------------------------------------

    async def convert_event(self, event: RichAgentStreamEvent[Any]) -> AsyncIterator[Event]:
        """Convert a single AgentPool event into zero or more OpenCode events.

        Delegates to :meth:`EventProcessor.process` with the adapter's context.
        One AgentPool event may yield multiple OpenCode events (e.g. a
        ``StreamCompleteEvent`` produces a ``StepFinishPart`` and a
        ``SessionIdleEvent``).

        Args:
            event: The AgentPool stream event to convert.

        Yields:
            OpenCode :data:`Event` objects ready for SSE broadcasting.
        """
        async for oc_event in self._processor.process(event, self._context):
            yield oc_event

    # ------------------------------------------------------------------
    # Stream conversion
    # ------------------------------------------------------------------

    async def convert_stream(
        self,
        stream: AsyncIterator[RichAgentStreamEvent[Any]],
    ) -> AsyncIterator[Event]:
        """Convert an entire stream of AgentPool events into OpenCode events.

        This is a convenience wrapper around :meth:`convert_event` that iterates
        over a full async stream.  For production code that needs error handling,
        finalisation, and step-finish tracking, prefer using
        :class:`OpenCodeStreamAdapter` directly.

        Args:
            stream: Async iterator of AgentPool stream events.

        Yields:
            OpenCode :data:`Event` objects ready for SSE broadcasting.
        """
        async for agent_event in stream:
            async for oc_event in self.convert_event(agent_event):
                yield oc_event

    # ------------------------------------------------------------------
    # Context accessors (read-only)
    # ------------------------------------------------------------------

    @property
    def context(self) -> EventProcessorContext:
        """The underlying event processor context."""
        return self._context

    @property
    def response_text(self) -> str:
        """Accumulated response text from the context."""
        return self._context.response_text

    @property
    def input_tokens(self) -> int:
        """Input token count from the context."""
        return self._context.input_tokens

    @property
    def output_tokens(self) -> int:
        """Output token count from the context."""
        return self._context.output_tokens

    @property
    def total_cost(self) -> float:
        """Total cost from the context."""
        return self._context.total_cost
