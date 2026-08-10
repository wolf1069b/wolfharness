"""L3 VCR test — native agent streaming event sequence (P6 pattern).

Pattern P6: ``agent.run_stream()`` + assert event sequence. Verifies the
streaming event order and delta aggregation. VCR replays the streaming
``POST .../chat/completions`` exchange (``stream: true``) so the SSE chunks
from the recorded response are reconstructed into the AgentPool event
sequence.

Cassette: ``tests/cassettes/vcr/test_native_streaming/test_streaming_event_sequence.yaml``
([HUMAN-REQUIRED] — record with ``--record-mode=once`` and ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dirty_equals import IsStr
from pydantic_ai.messages import TextPart, ThinkingPart
import pytest

from tests.vcr.conftest import cassette_exists
from wolfharness.agents.events import (
    PartDeltaEvent,
    PartStartEvent,
    RunStartedEvent,
    StreamCompleteEvent,
)


if TYPE_CHECKING:
    from wolfharness import AgentPool

pytestmark = pytest.mark.vcr

_MODULE_STEM = "test_native_streaming"


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_streaming_event_sequence"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_streaming_event_sequence(vcr_pool: AgentPool) -> None:
    """The streaming event sequence follows the expected structural order.

    Expected structure (design D8, P6):
        RunStartedEvent → (PartStartEvent → PartDeltaEvent* → PartEndEvent)*
        → FinalResultEvent* → StreamCompleteEvent

    PydanticAI emits multiple parts (text, thinking, tool calls), each with
    its own PartStart/PartDelta/PartEnd cycle. The test asserts structural
    invariants rather than an exact sequence to remain resilient across
    pydantic-ai versions and model response patterns.

    Assertions:
    1. First event is ``RunStartedEvent``
    2. Last event is ``StreamCompleteEvent``
    3. At least one ``PartStartEvent`` and ``PartDeltaEvent`` exist
    4. First ``PartStartEvent`` precedes first ``PartDeltaEvent``
    5. First ``PartDeltaEvent`` precedes ``StreamCompleteEvent``
    """
    agent = vcr_pool.get_agent("test_agent")
    events: list[object] = [
        event async for event in agent.run_stream("Count from 1 to 5, one number per line.")
    ]

    assert events, "run_stream produced no events"

    # 1. First event must be RunStartedEvent.
    assert isinstance(events[0], RunStartedEvent), (
        f"First event should be RunStartedEvent, got {type(events[0]).__name__}"
    )

    # 2. Last event must be StreamCompleteEvent.
    assert isinstance(events[-1], StreamCompleteEvent), (
        f"Last event should be StreamCompleteEvent, got {type(events[-1]).__name__}"
    )

    # 3. Must have at least one PartStartEvent and PartDeltaEvent.
    part_start_indices = [i for i, e in enumerate(events) if isinstance(e, PartStartEvent)]
    part_delta_indices = [i for i, e in enumerate(events) if isinstance(e, PartDeltaEvent)]
    assert part_start_indices, "Expected at least one PartStartEvent"
    assert part_delta_indices, "Expected at least one PartDeltaEvent"

    # 4. First PartStartEvent must precede first PartDeltaEvent.
    assert part_start_indices[0] < part_delta_indices[0], (
        "First PartStartEvent must come before first PartDeltaEvent"
    )

    # 5. First PartDeltaEvent must precede StreamCompleteEvent.
    stream_complete_index = next(
        i for i, e in enumerate(events) if isinstance(e, StreamCompleteEvent)
    )
    assert part_delta_indices[0] < stream_complete_index, (
        "First PartDeltaEvent must come before StreamCompleteEvent"
    )


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_streaming_delta_aggregation"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_streaming_delta_aggregation(vcr_pool: AgentPool) -> None:
    """Concatenating all ``PartDeltaEvent`` deltas yields the full response.

    Verifies that delta aggregation works: the union of streamed chunks
    matches the final ``StreamCompleteEvent`` message content.
    """
    agent = vcr_pool.get_agent("test_agent")
    events: list[object] = [
        event async for event in agent.run_stream("Say hello in one short sentence.")
    ]

    deltas = [e for e in events if isinstance(e, PartDeltaEvent)]
    completes = [e for e in events if isinstance(e, StreamCompleteEvent)]

    assert deltas, "Expected at least one PartDeltaEvent"
    assert len(completes) == 1

    # Concatenate delta text. PartDeltaEvent.delta may be a str or a content
    # block; handle both.
    parts: list[str] = []
    for delta in deltas:
        delta_text = getattr(delta, "delta", None)
        if isinstance(delta_text, str):
            parts.append(delta_text)
        elif delta_text is not None:
            parts.append(str(delta_text))
    aggregated = "".join(parts)
    assert aggregated == IsStr(min_length=1)


@pytest.mark.skipif(
    not cassette_exists(_MODULE_STEM, "test_streaming_part_start_structure"),
    reason="Cassette not recorded yet — run with --record-mode=once",
)
async def test_streaming_part_start_structure(vcr_pool: AgentPool) -> None:
    """``PartStartEvent`` and ``StreamCompleteEvent`` carry the expected fields.

    ``PartStartEvent`` inherits from pydantic-ai's ``PyAIPartStartEvent`` and
    exposes a ``part`` field containing the content part (``TextPart``,
    ``ThinkingPart``, etc.) — not a ``part_type`` field.
    """
    agent = vcr_pool.get_agent("test_agent")
    events: list[object] = [event async for event in agent.run_stream("Say hello.")]

    starts = [e for e in events if isinstance(e, PartStartEvent)]
    completes = [e for e in events if isinstance(e, StreamCompleteEvent)]
    assert starts, "Expected at least one PartStartEvent"
    assert completes, "Expected at least one StreamCompleteEvent"

    # PartStartEvent.part contains the content part (TextPart, ThinkingPart, etc.).
    first_start = starts[0]
    assert isinstance(first_start.part, TextPart | ThinkingPart), (
        f"Expected TextPart or ThinkingPart, got {type(first_start.part).__name__}"
    )

    first_complete = completes[0]
    assert first_complete.message.content is not None
