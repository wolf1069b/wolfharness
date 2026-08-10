"""Tests for ThinkingPart/ThinkingPartDelta normalization in EventMapper.

Verifies that raw CoT provider reasoning text (stored in
provider_details['raw_content']) is extracted and populated into
content/content_delta so protocol converters can read it.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent as PyAIPartDeltaEvent,
    PartStartEvent as PyAIPartStartEvent,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    UserPromptPart,
)
from pydantic_ai.models.openai import _make_raw_content_updater
import pytest

from wolfharness.agents.events.events import PartDeltaEvent, PartStartEvent
from wolfharness.orchestrator.event_mapper import (
    EventMapper,
    _normalize_thinking_event,
    normalize_thinking_parts_in_messages,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def mapper() -> EventMapper:
    return EventMapper(agent_name="test_agent", message_id="msg_001")


# ---------------------------------------------------------------------------
# PartStartEvent normalization
# ---------------------------------------------------------------------------


class TestPartStartEventNormalization:
    """Tests for PartStartEvent with empty content from raw CoT providers."""

    def test_empty_content_normalized_from_dict_raw_content(self):
        """PartStartEvent with empty content gets text from provider_details."""
        part = ThinkingPart(
            content="",
            id="resp_001",
            provider_details={"raw_content": ["The user asks"]},
        )
        event = PartStartEvent(index=0, part=part)

        result = _normalize_thinking_event(event)

        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, ThinkingPart)
        assert result.part.content == "The user asks"
        assert result.part.id == "resp_001"
        assert result.part.provider_details == {"raw_content": ["The user asks"]}
        assert result.index == 0

    def test_populated_content_not_modified(self):
        """PartStartEvent with populated content passes through unchanged."""
        part = ThinkingPart(content="reasoning summary", id="resp_002")
        event = PartStartEvent(index=1, part=part)

        result = _normalize_thinking_event(event)

        assert result is event
        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, ThinkingPart)
        assert result.part.content == "reasoning summary"

    def test_no_raw_content_not_modified(self):
        """PartStartEvent without raw_content in provider_details unchanged."""
        part = ThinkingPart(
            content="",
            provider_details={"other_key": "value"},
        )
        event = PartStartEvent(index=0, part=part)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_none_provider_details_not_modified(self):
        """PartStartEvent with None provider_details unchanged."""
        part = ThinkingPart(content="", provider_details=None)
        event = PartStartEvent(index=0, part=part)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_empty_raw_content_list_not_modified(self):
        """PartStartEvent with empty raw_content list unchanged."""
        part = ThinkingPart(
            content="",
            provider_details={"raw_content": []},
        )
        event = PartStartEvent(index=0, part=part)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_retains_event_type_and_structure(self):
        """Normalized event retains PartStartEvent type and index."""
        part = ThinkingPart(
            content="",
            provider_details={"raw_content": ["text"]},
        )
        event = PartStartEvent(index=5, part=part)

        result = _normalize_thinking_event(event)

        assert isinstance(result, PartStartEvent)
        assert result.index == 5
        assert isinstance(result.part, ThinkingPart)


# ---------------------------------------------------------------------------
# PartDeltaEvent normalization (callable provider_details)
# ---------------------------------------------------------------------------


class TestPartDeltaEventCallableNormalization:
    """Tests for PartDeltaEvent with callable provider_details."""

    def test_none_content_delta_normalized_from_callable(self):
        """PartDeltaEvent with callable provider_details gets text via callable(None)."""
        updater = _make_raw_content_updater(" asks about", 0)
        delta = ThinkingPartDelta(content_delta=None, provider_details=updater)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert isinstance(result, PartDeltaEvent)
        assert isinstance(result.delta, ThinkingPartDelta)
        assert result.delta.content_delta == " asks about"

    def test_callable_with_content_index_greater_than_zero(self):
        """Callable with content_index > 0 pads correctly, raw_content[-1] is delta."""
        updater = _make_raw_content_updater("second_seg", 1)
        delta = ThinkingPartDelta(content_delta=None, provider_details=updater)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert isinstance(result, PartDeltaEvent)
        assert isinstance(result.delta, ThinkingPartDelta)
        assert result.delta.content_delta == "second_seg"

    def test_callable_returning_non_dict_not_modified(self):
        """Callable returning non-dict is handled defensively."""

        def bad_callable(_existing: Any) -> Any:
            return "not a dict"

        delta = ThinkingPartDelta(content_delta=None, provider_details=bad_callable)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_callable_raising_exception_not_modified(self):
        """Callable that raises is handled defensively."""

        def raising_callable(_existing: object) -> dict[str, object]:
            raise RuntimeError("boom")

        delta = ThinkingPartDelta(content_delta=None, provider_details=raising_callable)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_callable_with_empty_delta_not_modified(self):
        """Callable with empty string delta returns empty text, skipped."""
        updater = _make_raw_content_updater("", 0)
        delta = ThinkingPartDelta(content_delta=None, provider_details=updater)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event


# ---------------------------------------------------------------------------
# PartDeltaEvent normalization (dict provider_details)
# ---------------------------------------------------------------------------


class TestPartDeltaEventDictNormalization:
    """Tests for PartDeltaEvent with dict provider_details."""

    def test_none_content_delta_normalized_from_dict(self):
        """PartDeltaEvent with dict provider_details extracts raw_content[-1]."""
        delta = ThinkingPartDelta(
            content_delta=None,
            provider_details={"raw_content": ["", "", "delta_text"]},
        )
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert isinstance(result, PartDeltaEvent)
        assert isinstance(result.delta, ThinkingPartDelta)
        assert result.delta.content_delta == "delta_text"

    def test_dict_without_raw_content_not_modified(self):
        """PartDeltaEvent with dict lacking raw_content unchanged."""
        delta = ThinkingPartDelta(
            content_delta=None,
            provider_details={"other": "value"},
        )
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event


# ---------------------------------------------------------------------------
# Non-modification cases
# ---------------------------------------------------------------------------


class TestNoModificationCases:
    """Tests for events that should not be modified by normalization."""

    def test_populated_content_delta_not_modified(self):
        """PartDeltaEvent with populated content_delta unchanged."""
        delta = ThinkingPartDelta(content_delta="reasoning delta")
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event

    def test_none_provider_details_not_modified(self):
        """PartDeltaEvent with None provider_details unchanged."""
        delta = ThinkingPartDelta(content_delta=None, provider_details=None)
        event = PartDeltaEvent(index=0, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event


# ---------------------------------------------------------------------------
# Full stream reconstruction
# ---------------------------------------------------------------------------


class TestFullStreamReconstruction:
    """Test that normalizing a full stream reconstructs complete reasoning text."""

    def test_multi_delta_stream_reconstruction(self):
        """Concatenating normalized content + content_deltas == original deltas."""
        deltas = ["The user", " asks about", " Python", " testing."]

        # Simulate pydantic-ai main branch behavior:
        # PartStartEvent with empty content, provider_details resolved dict
        first_updater = _make_raw_content_updater(deltas[0], 0)
        resolved_pd = first_updater(None)
        part = ThinkingPart(content="", id="resp_001", provider_details=resolved_pd)
        start_event = PartStartEvent(index=0, part=part)

        # PartDeltaEvents with None content_delta, callable provider_details
        delta_events: list[PartDeltaEvent] = []
        for delta_text in deltas[1:]:
            updater = _make_raw_content_updater(delta_text, 0)
            part_delta = ThinkingPartDelta(content_delta=None, provider_details=updater)
            delta_events.append(PartDeltaEvent(index=0, delta=part_delta))

        # Normalize all
        normalized_start = _normalize_thinking_event(start_event)
        normalized_deltas = [_normalize_thinking_event(e) for e in delta_events]

        # Reconstruct
        assert isinstance(normalized_start, PartStartEvent)
        assert isinstance(normalized_start.part, ThinkingPart)
        reconstructed = normalized_start.part.content
        for nd in normalized_deltas:
            assert isinstance(nd, PartDeltaEvent)
            assert isinstance(nd.delta, ThinkingPartDelta)
            assert nd.delta.content_delta is not None
            reconstructed += nd.delta.content_delta

        assert reconstructed == "".join(deltas)


# ---------------------------------------------------------------------------
# Official OpenAI reasoning path
# ---------------------------------------------------------------------------


class TestOfficialOpenAIReasoning:
    """Test that official OpenAI reasoning summaries (content populated) are unchanged."""

    def test_official_part_start_event_unchanged(self):
        """Official OpenAI reasoning PartStartEvent with content passes through."""
        part = ThinkingPart(content="Summary text", id="resp_002")
        event = PartStartEvent(index=1, part=part)

        result = _normalize_thinking_event(event)

        assert result is event
        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, ThinkingPart)
        assert result.part.content == "Summary text"

    def test_official_part_delta_event_unchanged(self):
        """Official OpenAI reasoning PartDeltaEvent with content_delta passes through."""
        delta = ThinkingPartDelta(content_delta=" more summary")
        event = PartDeltaEvent(index=1, delta=delta)

        result = _normalize_thinking_event(event)

        assert result is event
        assert isinstance(result, PartDeltaEvent)
        assert isinstance(result.delta, ThinkingPartDelta)
        assert result.delta.content_delta == " more summary"


# ---------------------------------------------------------------------------
# EventMapper.map_event integration
# ---------------------------------------------------------------------------


class TestEventMapperIntegration:
    """Test that EventMapper.map_event() applies normalization automatically."""

    def test_map_event_normalizes_part_start(self, mapper: EventMapper):
        """map_event normalizes PartStartEvent from pydantic-ai."""
        # Simulate raw CoT: ThinkingPart with empty content
        updater = _make_raw_content_updater("reasoning text", 0)
        resolved_pd = updater(None)
        pyai_event = PyAIPartStartEvent(
            index=0,
            part=ThinkingPart(content="", id="resp_003", provider_details=resolved_pd),
        )

        result = mapper.map_event(pyai_event)

        assert result is not None
        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, ThinkingPart)
        assert result.part.content == "reasoning text"

    def test_map_event_normalizes_part_delta(self, mapper: EventMapper):
        """map_event normalizes PartDeltaEvent from pydantic-ai."""
        updater = _make_raw_content_updater("delta text", 0)
        pyai_event = PyAIPartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(content_delta=None, provider_details=updater),
        )

        result = mapper.map_event(pyai_event)

        assert result is not None
        assert isinstance(result, PartDeltaEvent)
        assert isinstance(result.delta, ThinkingPartDelta)
        assert result.delta.content_delta == "delta text"

    def test_map_event_preserves_populated_content(self, mapper: EventMapper):
        """map_event does not modify events with populated content."""
        pyai_event = PyAIPartStartEvent(
            index=0,
            part=ThinkingPart(content="already populated"),
        )

        result = mapper.map_event(pyai_event)

        assert result is not None
        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, ThinkingPart)
        assert result.part.content == "already populated"

    def test_map_event_passes_text_events_unchanged(self, mapper: EventMapper):
        """map_event does not affect TextPart events."""
        pyai_event = PyAIPartStartEvent(
            index=0,
            part=TextPart(content="hello world"),
        )

        result = mapper.map_event(pyai_event)

        assert result is not None
        assert isinstance(result, PartStartEvent)
        assert isinstance(result.part, TextPart)
        assert result.part.content == "hello world"


# ---------------------------------------------------------------------------
# ModelResponse message_history normalization (issue #155)
# ---------------------------------------------------------------------------


class TestNormalizeThinkingPartsInMessages:
    """Tests for normalize_thinking_parts_in_messages — the post-stream fix.

    After streaming completes, the final ModelResponse stored in
    message_history has ThinkingPart.content="" for raw CoT providers.
    normalize_thinking_parts_in_messages walks the message list and
    copies raw_content text into content.
    """

    def test_empty_content_normalized_from_dict_raw_content(self):
        """ModelResponse with empty ThinkingPart gets content from raw_content."""
        thinking = ThinkingPart(
            content="",
            id="resp_001",
            provider_name="openai",
            provider_details={"raw_content": ["The model reasons about the question."]},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "The model reasons about the question."
        assert messages[0].parts[0].provider_details == {
            "raw_content": ["The model reasons about the question."]
        }

    def test_multiple_thinking_parts_all_normalized(self):
        """Multiple ThinkingParts in one ModelResponse are all normalized."""
        part1 = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["first reasoning"]},
        )
        part2 = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["second reasoning"]},
        )
        response = ModelResponse(parts=[part1, part2], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "first reasoning"
        assert messages[0].parts[1].content == "second reasoning"

    def test_populated_content_not_modified(self):
        """ThinkingPart with non-empty content is left unchanged."""
        thinking = ThinkingPart(content="already has content", id="resp_002")
        response = ModelResponse(parts=[thinking], model_name="openai:gpt-5.2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "already has content"

    def test_no_raw_content_not_modified(self):
        """ThinkingPart without raw_content in provider_details is unchanged."""
        thinking = ThinkingPart(
            content="",
            provider_details={"other_key": "value"},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == ""

    def test_none_provider_details_not_modified(self):
        """ThinkingPart with None provider_details is unchanged."""
        thinking = ThinkingPart(content="", provider_details=None)
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == ""

    def test_non_model_response_messages_skipped(self):
        """ModelRequest messages and other types are not affected."""
        request = ModelRequest(parts=[UserPromptPart(content="Hello")])
        thinking = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["reasoning"]},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [request, response]

        normalize_thinking_parts_in_messages(messages)

        # ModelRequest untouched
        assert isinstance(messages[0], ModelRequest)
        assert messages[0].parts[0].content == "Hello"
        # ModelResponse normalized
        assert messages[1].parts[0].content == "reasoning"

    def test_mixed_part_types_in_model_response(self):
        """TextPart and ThinkingPart coexist; only ThinkingPart is normalized."""
        text_part = TextPart(content="The answer is 42.")
        thinking = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["I need to calculate."]},
        )
        response = ModelResponse(parts=[text_part, thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "The answer is 42."
        assert isinstance(messages[0].parts[0], TextPart)
        assert messages[0].parts[1].content == "I need to calculate."

    def test_empty_message_list_no_error(self):
        """Empty list is a no-op, no error raised."""
        messages: list[Any] = []
        normalize_thinking_parts_in_messages(messages)
        assert messages == []

    def test_multi_raw_content_list_joins_all_entries(self):
        """When raw_content has multiple chunks, all are joined into content."""
        thinking = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["first ", "second ", "third"]},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "first second third"

    def test_multi_raw_content_list_skips_non_str_entries(self):
        """Non-string entries in raw_content are skipped during join."""
        thinking = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["text ", 123, None, " more"]},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)

        assert messages[0].parts[0].content == "text  more"

    def test_idempotent_double_normalization(self):
        """Calling normalize twice does not corrupt content."""
        thinking = ThinkingPart(
            content="",
            provider_name="openai",
            provider_details={"raw_content": ["reasoning text"]},
        )
        response = ModelResponse(parts=[thinking], model_name="svc/kimi-k2")
        messages: list[Any] = [response]

        normalize_thinking_parts_in_messages(messages)
        assert messages[0].parts[0].content == "reasoning text"

        # Second call: content is non-empty, so it's a no-op
        normalize_thinking_parts_in_messages(messages)
        assert messages[0].parts[0].content == "reasoning text"
