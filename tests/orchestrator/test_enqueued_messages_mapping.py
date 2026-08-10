"""Unit tests for EnqueuedMessagesEvent → UserMessageInsertedEvent mapping.

Tests the ``handle_enqueued_messages`` method on :class:`EventMapper`,
which maps pydantic-ai's ``EnqueuedMessagesEvent`` to AgentPool's
:class:`UserMessageInsertedEvent` with delivery inference based on
the current node type.

Only events with a matching FIFO message_id are mapped — empty FIFO
means the EnqueuedMessagesEvent came from pydantic-ai's internal flow
(not from steer()/followup()) and is dropped.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    EnqueuedMessagesEvent,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
import pytest

from wolfharness.agents.events.events import UserMessageInsertedEvent
from wolfharness.orchestrator.event_mapper import EventMapper


def _make_enqueued_event(content: str = "Hello, steer me!") -> EnqueuedMessagesEvent:
    """Build an EnqueuedMessagesEvent with a single ModelRequest containing a UserPromptPart."""
    return EnqueuedMessagesEvent(
        enqueue_id="eq-001",
        messages=(ModelRequest(parts=[UserPromptPart(content=content)]),),
    )


def _make_mapper(fifo: list[str] | None = None) -> EventMapper:
    """Build an EventMapper with optional FIFO queue."""
    kwargs: dict[str, Any] = {"agent_name": "test-agent", "message_id": "msg-001"}
    if fifo is not None:
        kwargs["_enqueue_message_ids"] = fifo
    return EventMapper(**kwargs)


@pytest.mark.unit
def test_enqueued_messages_steer_delivery_for_model_request_node() -> None:
    """EnqueuedMessagesEvent during ModelRequestNode → delivery='steer'."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = _make_enqueued_event("Steer this conversation!")

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is not None
    assert isinstance(result, UserMessageInsertedEvent)
    assert result.delivery == "steer"
    assert result.source == "processed"
    assert result.content == "Steer this conversation!"
    assert result.message_id == "test-msg-id"
    assert result.session_id == ""
    assert result.meta is None


@pytest.mark.unit
def test_enqueued_messages_followup_delivery_for_call_tools_node() -> None:
    """EnqueuedMessagesEvent during CallToolsNode → delivery='followup'."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = _make_enqueued_event("Followup message")

    result = mapper.map_event(event, current_node_type="CallToolsNode")

    assert result is not None
    assert isinstance(result, UserMessageInsertedEvent)
    assert result.delivery == "followup"
    assert result.source == "processed"
    assert result.content == "Followup message"


@pytest.mark.unit
def test_enqueued_messages_followup_delivery_for_end_node() -> None:
    """EnqueuedMessagesEvent during End node → delivery='followup'."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = _make_enqueued_event("After-turn message")

    result = mapper.map_event(event, current_node_type="End")

    assert result is not None
    assert isinstance(result, UserMessageInsertedEvent)
    assert result.delivery == "followup"
    assert result.content == "After-turn message"


@pytest.mark.unit
def test_enqueued_messages_unknown_node_type_defaults_to_steer() -> None:
    """EnqueuedMessagesEvent with unknown node type defaults to delivery='steer'."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = _make_enqueued_event("Unknown context")

    result = mapper.map_event(event, current_node_type="unknown")

    assert result is not None
    assert isinstance(result, UserMessageInsertedEvent)
    assert result.delivery == "steer"


@pytest.mark.unit
def test_enqueued_messages_empty_messages_returns_none() -> None:
    """EnqueuedMessagesEvent with empty messages tuple returns None."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = EnqueuedMessagesEvent(enqueue_id="eq-empty", messages=())

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is None


@pytest.mark.unit
def test_enqueued_messages_no_user_prompt_part_returns_none() -> None:
    """EnqueuedMessagesEvent containing only ModelResponse (no UserPromptPart) returns None."""
    mapper = _make_mapper(fifo=["test-msg-id"])
    event = EnqueuedMessagesEvent(
        enqueue_id="eq-no-user",
        messages=(ModelResponse(parts=[TextPart(content="Assistant response")]),),
    )

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is None


@pytest.mark.unit
def test_enqueued_messages_empty_fifo_returns_none() -> None:
    """Empty FIFO queue means EnqueuedMessagesEvent came from pydantic-ai
    internal flow, not from steer()/followup(). Drop it.
    """  # noqa: D205
    mapper = _make_mapper(fifo=[])
    event = _make_enqueued_event("Spontaneous internal enqueue")

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is None


@pytest.mark.unit
def test_enqueued_messages_no_fifo_param_returns_none() -> None:
    """No FIFO queue parameter means no steer/followup preceded the enqueue."""
    mapper = _make_mapper()
    event = _make_enqueued_event("No FIFO queue")

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is None


# ---------------------------------------------------------------------------
# FIFO message_id reuse tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_enqueued_messages_reuses_message_id_from_fifo_queue() -> None:
    """handle_enqueued_messages() reuses message_id from _enqueue_message_ids
    FIFO queue instead of generating a new UUID.
    """  # noqa: D205
    fifo: list[str] = ["steer-msg-123"]
    mapper = _make_mapper(fifo=fifo)
    event = _make_enqueued_event("Steer content")

    result = mapper.map_event(event, current_node_type="ModelRequestNode")

    assert result is not None
    assert isinstance(result, UserMessageInsertedEvent)
    assert result.message_id == "steer-msg-123"
    # FIFO should be drained.
    assert len(fifo) == 0


@pytest.mark.unit
def test_enqueued_messages_fifo_pop_order() -> None:
    """Multiple message_ids in FIFO are popped in FIFO order (first in, first out)."""
    fifo: list[str] = ["msg-a", "msg-b", "msg-c"]
    mapper = _make_mapper(fifo=fifo)

    result1 = mapper.map_event(_make_enqueued_event("first"), current_node_type="ModelRequestNode")
    result2 = mapper.map_event(_make_enqueued_event("second"), current_node_type="ModelRequestNode")
    result3 = mapper.map_event(_make_enqueued_event("third"), current_node_type="ModelRequestNode")

    assert result1 is not None
    assert result2 is not None
    assert result3 is not None
    assert result1.message_id == "msg-a"
    assert result2.message_id == "msg-b"
    assert result3.message_id == "msg-c"
    assert len(fifo) == 0
