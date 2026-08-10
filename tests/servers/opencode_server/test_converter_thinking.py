"""Tests for ThinkingPart round-trip through OpenCode converters.

These tests verify that ThinkingPart (LLM reasoning content) is preserved
when converting between ChatMessage and OpenCode MessageWithParts.
This covers the session restore path where messages are loaded from storage
and converted via chat_message_to_opencode().
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic_ai import (
    ModelResponse,
    RequestUsage,
    TextPart as PydanticTextPart,
    ThinkingPart as PydanticThinkingPart,
)
import pytest

from wolfharness.messaging.messages import ChatMessage
from wolfharness_server.opencode_server.converters import (
    chat_message_to_opencode,
    opencode_to_chat_message,
)
from wolfharness_server.opencode_server.models import ReasoningPart


pytestmark = pytest.mark.integration


def _make_assistant_chat_message_with_thinking(
    thinking_content: str = "Let me analyze this step by step.",
    text_content: str = "Here is my answer.",
) -> ChatMessage[str]:
    """Create a ChatMessage with both ThinkingPart and TextPart."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    model_response = ModelResponse(
        parts=[
            PydanticThinkingPart(content=thinking_content),
            PydanticTextPart(content=text_content),
        ],
        usage=RequestUsage(),
        model_name="test-model",
        timestamp=timestamp,
    )
    return ChatMessage(
        content=text_content,
        role="assistant",
        message_id="msg-test-1",
        session_id="session-test-1",
        timestamp=timestamp,
        messages=[model_response],
        usage=RequestUsage(),
        model_name="test-model",
        provider_name="test-provider",
        finish_reason="stop",
    )


def _make_assistant_chat_message_with_dict_thinking(
    thinking_content: str = "Dict thinking content.",
    text_content: str = "Dict text content.",
) -> ChatMessage[str]:
    """Create a ChatMessage where model messages are dicts (as loaded from storage)."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    dict_response = {
        "kind": "response",
        "parts": [
            {"part_kind": "thinking", "content": thinking_content, "id": None},
            {"part_kind": "text", "content": text_content, "id": None},
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "model_name": "test-model",
        "timestamp": timestamp.isoformat(),
    }
    return ChatMessage(
        content=text_content,
        role="assistant",
        message_id="msg-test-dict-1",
        session_id="session-test-dict-1",
        timestamp=timestamp,
        messages=[dict_response],  # type: ignore[list-item]
        usage=RequestUsage(),
        model_name="test-model",
        provider_name="test-provider",
        finish_reason="stop",
    )


# =============================================================================
# chat_message_to_opencode: ThinkingPart -> ReasoningPart
# =============================================================================


def test_chat_message_to_opencode_preserves_thinking_part():
    """Test that ThinkingPart converts to ReasoningPart with correct content."""
    msg = _make_assistant_chat_message_with_thinking(
        thinking_content="I need to consider the trade-offs.",
        text_content="The answer is 42.",
    )
    result = chat_message_to_opencode(msg, session_id="s1")

    reasoning_parts = [p for p in result.parts if isinstance(p, ReasoningPart)]
    assert len(reasoning_parts) == 1, (
        f"Expected 1 ReasoningPart, got {len(reasoning_parts)}. "
        f"All parts: {[type(p).__name__ for p in result.parts]}"
    )
    assert reasoning_parts[0].text == "I need to consider the trade-offs."


def test_chat_message_to_opencode_preserves_text_and_thinking():
    """Test that both ReasoningPart and TextPart are present after conversion."""
    msg = _make_assistant_chat_message_with_thinking()
    result = chat_message_to_opencode(msg, session_id="s1")

    from wolfharness_server.opencode_server.models import TextPart

    reasoning_parts = [p for p in result.parts if isinstance(p, ReasoningPart)]
    text_parts = [p for p in result.parts if isinstance(p, TextPart)]

    assert len(reasoning_parts) == 1
    assert len(text_parts) == 1
    assert reasoning_parts[0].text == "Let me analyze this step by step."
    assert text_parts[0].text == "Here is my answer."


def test_chat_message_to_opencode_dict_path_preserves_thinking():
    """Test that dict path (from storage) preserves thinking content."""
    msg = _make_assistant_chat_message_with_dict_thinking(
        thinking_content="Dict reasoning here.",
        text_content="Dict text here.",
    )
    result = chat_message_to_opencode(msg, session_id="s1")

    reasoning_parts = [p for p in result.parts if isinstance(p, ReasoningPart)]
    assert len(reasoning_parts) == 1, (
        f"Expected 1 ReasoningPart from dict path, got {len(reasoning_parts)}. "
        f"All parts: {[type(p).__name__ for p in result.parts]}"
    )
    assert reasoning_parts[0].text == "Dict reasoning here."


def test_chat_message_to_opencode_no_thinking_no_reasoning_part():
    """Test that no ReasoningPart is created without ThinkingPart."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    model_response = ModelResponse(
        parts=[PydanticTextPart(content="Just text, no thinking.")],
        usage=RequestUsage(),
        model_name="test-model",
        timestamp=timestamp,
    )
    msg = ChatMessage(
        content="Just text, no thinking.",
        role="assistant",
        message_id="msg-no-think",
        session_id="s1",
        timestamp=timestamp,
        messages=[model_response],
        usage=RequestUsage(),
        model_name="test-model",
        provider_name="test-provider",
        finish_reason="stop",
    )
    result = chat_message_to_opencode(msg, session_id="s1")

    reasoning_parts = [p for p in result.parts if isinstance(p, ReasoningPart)]
    assert len(reasoning_parts) == 0


# =============================================================================
# opencode_to_chat_message: ReasoningPart -> ThinkingPart
# =============================================================================


def test_opencode_to_chat_message_preserves_reasoning_part():
    """Test that ReasoningPart converts back to ThinkingPart."""
    msg = _make_assistant_chat_message_with_thinking()
    opencode_msg = chat_message_to_opencode(msg, session_id="s1")
    restored = opencode_to_chat_message(opencode_msg)

    thinking_parts: list[PydanticThinkingPart] = []
    for model_msg in restored.messages:
        if isinstance(model_msg, ModelResponse):
            thinking_parts.extend(p for p in model_msg.parts if isinstance(p, PydanticThinkingPart))

    assert len(thinking_parts) == 1, (
        f"Expected 1 ThinkingPart in restored message, got {len(thinking_parts)}"
    )
    assert thinking_parts[0].content == "Let me analyze this step by step."


# =============================================================================
# Round-trip: ChatMessage -> OpenCode -> ChatMessage
# =============================================================================


def test_thinking_content_survives_round_trip():
    """Test that thinking content survives a full round-trip through converters."""
    original_thinking = "Round trip thinking content."
    original_text = "Round trip text content."

    msg = _make_assistant_chat_message_with_thinking(
        thinking_content=original_thinking,
        text_content=original_text,
    )
    opencode_msg = chat_message_to_opencode(msg, session_id="s1")
    restored = opencode_to_chat_message(opencode_msg)

    restored_thinking: list[str] = []
    for model_msg in restored.messages:
        if isinstance(model_msg, ModelResponse):
            restored_thinking.extend(
                p.content for p in model_msg.parts if isinstance(p, PydanticThinkingPart)
            )

    assert len(restored_thinking) == 1
    assert restored_thinking[0] == original_thinking

    # Check text survived
    assert restored.content == original_text
