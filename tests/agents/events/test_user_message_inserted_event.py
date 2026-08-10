"""Tests for UserMessageInsertedEvent construction, defaults, and frozen behavior."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json

import pytest

from wolfharness.agents.events.events import UserMessageInsertedEvent


pytestmark = [pytest.mark.unit]


def test_construction_with_all_fields() -> None:
    """UserMessageInsertedEvent accepts all six fields explicitly."""
    event = UserMessageInsertedEvent(
        session_id="sess-1",
        message_id="msg-1",
        content="hello world",
        delivery="steer",
        source="processed",
        timestamp=1234567890.0,
    )

    assert event.session_id == "sess-1"
    assert event.message_id == "msg-1"
    assert event.content == "hello world"
    assert event.delivery == "steer"
    assert event.source == "processed"
    assert event.timestamp == 1234567890.0


def test_defaults_are_empty_and_initial() -> None:
    """Default values match the spec.

    - session_id='', message_id='', content=''
    - delivery='initial', source='accepted'
    - timestamp auto-generated via time.time
    """
    event = UserMessageInsertedEvent()

    assert event.session_id == ""
    assert event.message_id == ""
    assert event.content == ""
    assert event.delivery == "initial"
    assert event.source == "accepted"
    assert isinstance(event.timestamp, float)


def test_multimodal_content_as_list() -> None:
    """Content accepts a list[Any] for multi-modal prompts."""
    parts: list[dict[str, str]] = [
        {"type": "text", "text": "describe this"},
        {"type": "image", "url": "https://example.com/img.png"},
    ]
    event = UserMessageInsertedEvent(
        session_id="sess-mm",
        message_id="msg-mm",
        content=parts,
    )

    assert isinstance(event.content, list)
    assert event.content == parts
    assert event.content[0] == {"type": "text", "text": "describe this"}
    assert event.content[1] == {"type": "image", "url": "https://example.com/img.png"}


@pytest.mark.parametrize("delivery", ["initial", "steer", "followup"])
def test_all_delivery_values(delivery: str) -> None:
    """UserMessageInsertedEvent accepts all three delivery literals."""
    event = UserMessageInsertedEvent(delivery=delivery)  # type: ignore[arg-type]
    assert event.delivery == delivery


@pytest.mark.parametrize("source", ["accepted", "processed"])
def test_all_source_values(source: str) -> None:
    """UserMessageInsertedEvent accepts both source literals."""
    event = UserMessageInsertedEvent(source=source)  # type: ignore[arg-type]
    assert event.source == source


def test_frozen_dataclass_cannot_modify_fields() -> None:
    """Frozen dataclass raises FrozenInstanceError on field assignment."""
    event = UserMessageInsertedEvent(session_id="sess-frozen")

    with pytest.raises(FrozenInstanceError):
        event.session_id = "changed"  # type: ignore[misc]


def test_frozen_dataclass_cannot_delete_fields() -> None:
    """Frozen dataclass raises FrozenInstanceError on field deletion."""
    event = UserMessageInsertedEvent(session_id="sess-frozen-del")

    with pytest.raises(FrozenInstanceError):
        del event.session_id  # type: ignore[misc]


def test_json_roundtrip_preserves_all_fields() -> None:
    """UserMessageInsertedEvent survives JSON roundtrip via asdict."""
    event = UserMessageInsertedEvent(
        session_id="sess-rt",
        message_id="msg-rt",
        content="roundtrip text",
        delivery="followup",
        source="accepted",
        timestamp=99.0,
    )

    data = json.dumps(asdict(event))
    restored = UserMessageInsertedEvent(**json.loads(data))

    assert restored == event


def test_json_roundtrip_with_multimodal_content() -> None:
    """UserMessageInsertedEvent with list content survives JSON roundtrip."""
    parts = [{"type": "text", "text": "hi"}, {"type": "image", "url": "u"}]
    event = UserMessageInsertedEvent(
        session_id="sess-rt-mm",
        message_id="msg-rt-mm",
        content=parts,
        delivery="steer",
    )

    data = json.dumps(asdict(event))
    restored = UserMessageInsertedEvent(**json.loads(data))

    assert restored.content == parts
    assert restored.delivery == "steer"
