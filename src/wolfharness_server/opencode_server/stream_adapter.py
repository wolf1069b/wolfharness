"""OpenCode stream adapter.

Translates RichAgentStreamEvent objects from the agent event system
into OpenCode SSE Event objects. The adapter yields events; broadcasting
and persistence are handled by the caller.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import RequestUsage

from wolfharness.log import get_logger
from wolfharness.tasks.exceptions import RunAbortedError
from wolfharness.utils import identifiers as identifier
from wolfharness.utils.time_utils import now_ms
from wolfharness_server.opencode_server.event_processor import EventProcessor
from wolfharness_server.opencode_server.event_processor_context import (
    EventProcessorContext,
)
from wolfharness_server.opencode_server.models import (
    MessageWithParts,
    PartUpdatedEvent,
    SessionErrorEvent,
    TokenCache,
    Tokens,
)
from wolfharness_server.opencode_server.models.parts import (
    StepFinishPart,
    TextPart,
    TimeStartEndOptional,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from wolfharness.agents.events.events import RichAgentStreamEvent
    from wolfharness_server.opencode_server.models import MessageWithParts
    from wolfharness_server.opencode_server.models.events import Event
    from wolfharness_server.opencode_server.state import ServerState

logger = get_logger(__name__)


@dataclass
class OpenCodeStreamAdapter:
    """Translates agent stream events into OpenCode SSE events.

    Owns all mutable tracking state (tool parts, text accumulation, token
    counters). Yields OpenCode ``Event`` objects ready for broadcasting.

    The adapter does NOT own:
    - Broadcasting (caller does ``state.broadcast_event``)
    - Agent invocation (caller provides the async iterator)
    - Message creation (caller sets up user/assistant messages)
    - Storage persistence (caller persists after streaming)
    - LSP warmup (caller provides ``on_file_paths`` callback)

    Args:
        state: The server state for session management and event routing.
        session_id: The OpenCode session ID.
        assistant_msg_id: The assistant message ID.
        assistant_msg: The mutable assistant message to append parts to.
        working_dir: Working directory for path context.
        on_file_paths: Optional callback invoked with file paths discovered during
            tool progress events (used for LSP warmup).
    """

    state: ServerState
    session_id: str
    assistant_msg_id: str
    assistant_msg: MessageWithParts
    working_dir: str
    on_file_paths: Callable[[list[str]], None] | None = None

    # Event processor and context for stream processing
    processor: EventProcessor = field(default_factory=EventProcessor, init=False)
    main_context: EventProcessorContext = field(init=False)
    _cost_info: Any = field(default=None, init=False)
    _step_finish_emitted: bool = field(default=False, init=False)
    """Tracks whether a *final* StepFinishPart was emitted for THIS session.

    Only set when a ``StreamCompleteEvent`` is processed (which emits the
    cumulative ``StepFinishPart``).  Per-step ``StepUsageEvent`` emissions
    do NOT set this flag — they are intermediate markers, and the final
    cumulative ``StepFinishPart`` from ``finalize()`` MUST still be
    emitted.

    Child subagent sessions also emit StepFinishPart events that bubble
    through process_stream(), but those must NOT suppress the parent's
    StepFinishPart — otherwise a stream interrupted after a child
    completes but before the parent's StreamCompleteEvent would leave
    the parent session without a StepFinishPart, causing the TUI to show
    it stuck in "working" state.
    """

    def __post_init__(self) -> None:
        self.main_context = EventProcessorContext(
            session_id=self.session_id,
            assistant_msg_id=self.assistant_msg_id,
            assistant_msg=self.assistant_msg,
            state=self.state,
            working_dir=self.working_dir,
        )

    # --- public read-only accessors ---

    @property
    def response_text(self) -> str:
        return self.main_context.response_text

    @property
    def input_tokens(self) -> int:
        return self.main_context.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.main_context.output_tokens

    @property
    def usage(self) -> RequestUsage:
        """Return usage statistics for the current response."""
        return RequestUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )

    @property
    def total_cost(self) -> float:
        return self.main_context.total_cost

    @property
    def cost_info(self) -> Any:
        """Return cost information for the current response."""

        # Use main_context's cost tracking
        class SimpleCostInfo:
            def __init__(self, total: float) -> None:
                self.total_cost = total

        return (
            SimpleCostInfo(self.main_context.total_cost) if self.main_context.total_cost else None
        )

    @property
    def text_part(self) -> TextPart | None:
        return self.main_context.text_part

    # --- main entry point ---

    async def process_stream(
        self,
        stream: AsyncIterator[RichAgentStreamEvent[Any]],
    ) -> AsyncIterator[Event]:
        """Consume agent events and yield OpenCode SSE events.

        Wraps the entire stream in error handling; on exception yields
        a ``SessionErrorEvent`` so the UI shows the failure.
        """
        try:
            async for event in stream:
                async for oc_event in self.processor.process(event, self.main_context):
                    # Track if a *final* StepFinishPart was emitted for THIS session.
                    # Per-step StepFinishPart (from StepUsageEvent) has step_index set
                    # and must NOT suppress the final cumulative emission.
                    if (
                        isinstance(oc_event, PartUpdatedEvent)
                        and isinstance(oc_event.properties.part, StepFinishPart)
                        and oc_event.properties.part.session_id == self.session_id
                        and oc_event.properties.part.step_index is None
                    ):
                        self._step_finish_emitted = True
                    yield oc_event
        except asyncio.CancelledError:
            # Stream was cancelled by user - this is expected behavior
            # Don't propagate the error, just log it
            logger.debug("Stream cancelled by user", session_id=self.session_id)
            raise  # Re-raise so caller can handle cleanup
        except RunAbortedError:
            # Agent aborted the run (e.g. question_for_user raised RunAbortedError
            # when user cancelled the questionnaire). Re-raise so
            # _wait_and_finalize can finalize the assistant message with
            # aborted state and update the agent's conversation history.
            logger.debug("Run aborted by agent", session_id=self.session_id)
            raise
        except Exception as e:  # noqa: BLE001
            self.main_context.response_text = f"Error calling agent: {e}"
            yield SessionErrorEvent.from_exception(session_id=self.session_id, exception=e)

    async def convert_event(self, event: RichAgentStreamEvent[Any]) -> AsyncIterator[Event]:
        """Convert a single agent event into OpenCode SSE events.

        Processes the event through the adapter's own processor so that
        mutable state (text accumulation, token counters, step-finish
        tracking) is updated on *this* adapter instance.  This is the
        preferred entry point when feeding events one-at-a-time from an
        external consumer (e.g. the EventBus subscriber in
        ``message_routes.py``).

        Args:
            event: The agent stream event to process.

        Yields:
            OpenCode Event objects for broadcasting.
        """
        async for oc_event in self.processor.process(event, self.main_context):
            if (
                isinstance(oc_event, PartUpdatedEvent)
                and isinstance(oc_event.properties.part, StepFinishPart)
                and oc_event.properties.part.session_id == self.session_id
                and oc_event.properties.part.step_index is None
            ):
                self._step_finish_emitted = True
            yield oc_event

    def finalize(self) -> Iterator[Event]:
        """Yield final events after the stream has ended.

        Produces the final text part update (or creates one if text was never
        streamed), the step-finish part (if not already emitted by
        _process_stream_complete), and the final text timing update.
        """
        response_time = now_ms()
        start = self.main_context.stream_start_ms

        # Final text part
        if self.main_context.response_text and self.main_context.text_part is None:
            # Text was never streamed incrementally — create a text part now
            text_part = TextPart(
                id=identifier.ascending("part"),
                message_id=self.assistant_msg_id,
                session_id=self.session_id,
                text=self.main_context.response_text,
                time=TimeStartEndOptional(start=start, end=response_time),
            )
            self.assistant_msg.parts.append(text_part)
            yield PartUpdatedEvent.create(text_part)
        elif self.main_context.text_part is not None:
            # Update streamed text part with final timing
            final_text_part = TextPart(
                id=self.main_context.text_part.id,
                message_id=self.assistant_msg_id,
                session_id=self.session_id,
                text=self.main_context.response_text,
                time=TimeStartEndOptional(start=start, end=response_time),
            )
            self.assistant_msg.update_part(final_text_part)
            yield PartUpdatedEvent.create(final_text_part)

        # Step finish — skip if already emitted by _process_stream_complete
        # (StreamCompleteEvent handler in EventProcessor also emits StepFinishPart)
        if self._step_finish_emitted:
            return

        cache = TokenCache(read=0, write=0)
        tokens = Tokens(
            cache=cache,
            input=self.main_context.input_tokens,
            output=self.main_context.output_tokens,
            reasoning=0,
        )
        step_finish = StepFinishPart(
            id=identifier.ascending("part"),
            message_id=self.assistant_msg_id,
            session_id=self.session_id,
            tokens=tokens,
            cost=self.main_context.total_cost,
        )
        self.assistant_msg.parts.append(step_finish)
        yield PartUpdatedEvent.create(step_finish)
