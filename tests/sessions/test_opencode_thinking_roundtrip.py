"""Tests for ThinkingPart field preservation through OpenCode storage round-trip.

Verifies that ThinkingPart.id, provider_name, signature, and provider_details
survive the ReasoningPart → JSON → ReasoningPart → ThinkingPart cycle.
See issue #156.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic_ai.messages import ModelResponse, ThinkingPart
import pytest

from wolfharness_server.opencode_server.models.message import (
    AssistantMessage,
    MessagePath,
    MessageTime,
)
from wolfharness_server.opencode_server.models.parts import ReasoningPart
from wolfharness_storage.opencode_provider.helpers import (
    _build_assistant_pydantic_messages,
)


@pytest.fixture
def timestamp() -> datetime:
    return datetime(2025, 7, 14, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def assistant_msg() -> AssistantMessage:
    return AssistantMessage(
        id="msg_test001",
        session_id="ses_test001",
        parent_id="msg_parent001",
        model_id="svc/kimi-k2",
        provider_id="openai",
        path=MessagePath(cwd="/tmp", root="/tmp"),
        time=MessageTime(created=1720958400000, completed=1720958401000),
    )


class TestThinkingPartRoundTrip:
    """Verify ThinkingPart fields survive storage round-trip via metadata."""

    def test_all_fields_preserved(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """All ThinkingPart fields are preserved through ReasoningPart.metadata."""
        reasoning = ReasoningPart(
            id="prt_001",
            session_id="ses_test001",
            message_id="msg_test001",
            text="The user asks about Python testing.",
            metadata={
                "thinking_id": "resp_001",
                "provider_name": "openai",
                "signature": "sig_abc123",
                "provider_details": {"raw_content": ["The user asks about Python testing."]},
            },
            time=None,
        )
        parts = [reasoning]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        assert len(messages) == 1
        assert isinstance(messages[0], ModelResponse)
        assert len(messages[0].parts) == 1
        thinking = messages[0].parts[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == "The user asks about Python testing."
        assert thinking.id == "resp_001"
        assert thinking.provider_name == "openai"
        assert thinking.signature == "sig_abc123"
        assert thinking.provider_details == {"raw_content": ["The user asks about Python testing."]}

    def test_no_metadata_returns_plain_thinkingpart(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """ReasoningPart without metadata produces ThinkingPart with default None fields."""
        reasoning = ReasoningPart(
            id="prt_002",
            session_id="ses_test001",
            message_id="msg_test001",
            text="reasoning without metadata",
        )
        parts = [reasoning]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        thinking = messages[0].parts[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == "reasoning without metadata"
        assert thinking.id is None
        assert thinking.provider_name is None
        assert thinking.signature is None
        assert thinking.provider_details is None

    def test_partial_metadata_preserved(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """Only provided metadata fields are set; others default to None."""
        reasoning = ReasoningPart(
            id="prt_003",
            session_id="ses_test001",
            message_id="msg_test001",
            text="partial reasoning",
            metadata={
                "thinking_id": "resp_003",
                "provider_name": "openai",
            },
        )
        parts = [reasoning]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        thinking = messages[0].parts[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == "partial reasoning"
        assert thinking.id == "resp_003"
        assert thinking.provider_name == "openai"
        assert thinking.signature is None
        assert thinking.provider_details is None

    def test_empty_metadata_dict_defaults_to_none(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """Empty metadata dict produces ThinkingPart with all None extra fields."""
        reasoning = ReasoningPart(
            id="prt_004",
            session_id="ses_test001",
            message_id="msg_test001",
            text="empty meta reasoning",
            metadata={},
        )
        parts = [reasoning]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        thinking = messages[0].parts[0]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == "empty meta reasoning"
        assert thinking.id is None
        assert thinking.provider_name is None

    def test_thinking_and_text_parts_coexist(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """ThinkingPart and TextPart in the same message are both preserved."""
        from wolfharness_server.opencode_server.models.parts import TextPart

        reasoning = ReasoningPart(
            id="prt_005",
            session_id="ses_test001",
            message_id="msg_test001",
            text="thinking here",
            metadata={"thinking_id": "resp_005", "provider_name": "openai"},
        )
        text = TextPart(
            id="prt_006",
            session_id="ses_test001",
            message_id="msg_test001",
            text="answer here",
        )
        parts = [reasoning, text]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        assert len(messages) == 1
        assert isinstance(messages[0], ModelResponse)
        assert len(messages[0].parts) == 2
        thinking = messages[0].parts[0]
        text_part = messages[0].parts[1]
        assert isinstance(thinking, ThinkingPart)
        assert thinking.content == "thinking here"
        assert thinking.id == "resp_005"
        assert thinking.provider_name == "openai"
        assert text_part.content == "answer here"

    def test_empty_text_reasoning_skipped(
        self, assistant_msg: AssistantMessage, timestamp: datetime
    ) -> None:
        """ReasoningPart with empty text is skipped (existing behavior)."""
        reasoning = ReasoningPart(
            id="prt_007",
            session_id="ses_test001",
            message_id="msg_test001",
            text="",
            metadata={"thinking_id": "resp_007"},
        )
        parts = [reasoning]

        messages = _build_assistant_pydantic_messages(assistant_msg, parts, timestamp)

        assert len(messages) == 0
