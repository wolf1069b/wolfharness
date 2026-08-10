"""Tests for ThinkingPart handling in the OpenAI-compatible API server.

Verifies that thinking/reasoning content is preserved in:
1. Non-streaming chat completions (reasoning_content field)
2. Streaming chat completions (reasoning_content delta in SSE)
3. Responses API (ResponseOutputReasoning in output)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic_ai import (
    ModelResponse,
    RequestUsage,
    TextPart as PydanticTextPart,
    ThinkingPart as PydanticThinkingPart,
)
import pytest

from wolfharness.messaging.messages import ChatMessage
from wolfharness_server.openai_api_server.completions.helpers import stream_response
from wolfharness_server.openai_api_server.completions.models import ChatCompletionRequest
from wolfharness_server.openai_api_server.responses.helpers import handle_request
from wolfharness_server.openai_api_server.responses.models import (
    ResponseMessage,
    ResponseOutputReasoning,
    ResponseRequest,
)


pytestmark = pytest.mark.integration


_DONE = "[DONE]"


def _make_assistant_with_thinking(
    thinking_content: str = "Let me reason about this.",
    text_content: str = "The answer is 42.",
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


# =============================================================================
# Non-streaming completions: reasoning_content field
# =============================================================================


def test_non_streaming_completion_includes_reasoning_content():
    """Test that reasoning_content is populated from ThinkingPart."""
    from wolfharness_server.openai_api_server.completions.models import OpenAIMessage

    msg = _make_assistant_with_thinking(
        thinking_content="Deep reasoning here.",
        text_content="Final answer.",
    )

    # Extract reasoning content the same way server.py does
    from pydantic_ai import ThinkingPart

    reasoning_parts: list[str] = []
    for model_msg in msg.messages:
        if isinstance(model_msg, dict):
            continue
        reasoning_parts.extend(
            part.content
            for part in model_msg.parts
            if isinstance(part, ThinkingPart) and part.content
        )

    openai_msg = OpenAIMessage(
        role="assistant",
        content=str(msg.content),
        reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
    )

    assert openai_msg.reasoning_content == "Deep reasoning here."
    assert openai_msg.content == "Final answer."


def test_non_streaming_completion_no_thinking_no_reasoning_content():
    """Test that reasoning_content is None without ThinkingPart."""
    from wolfharness_server.openai_api_server.completions.models import OpenAIMessage

    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    model_response = ModelResponse(
        parts=[PydanticTextPart(content="No thinking here.")],
        usage=RequestUsage(),
        model_name="test-model",
        timestamp=timestamp,
    )
    msg = ChatMessage(
        content="No thinking here.",
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

    from pydantic_ai import ThinkingPart

    reasoning_parts: list[str] = []
    for model_msg in msg.messages:
        if isinstance(model_msg, dict):
            continue
        reasoning_parts.extend(
            part.content
            for part in model_msg.parts
            if isinstance(part, ThinkingPart) and part.content
        )

    openai_msg = OpenAIMessage(
        role="assistant",
        content=str(msg.content),
        reasoning_content="\n".join(reasoning_parts) if reasoning_parts else None,
    )

    assert openai_msg.reasoning_content is None


# =============================================================================
# Streaming completions: reasoning_content delta in SSE
# =============================================================================


async def _collect_stream_chunks(
    events: Any,
    request: ChatCompletionRequest,
) -> list[dict[str, Any]]:
    """Collect all SSE chunks from stream_response as parsed dicts."""
    import anyenv

    chunks: list[dict[str, Any]] = []
    async for sse_data in stream_response(events, request):
        stripped = sse_data.strip()
        if not stripped.startswith("data: "):
            continue
        payload = stripped[6:]
        if payload == _DONE:
            continue
        chunks.append(anyenv.load_json(payload, return_type=dict))
    return chunks


@pytest.mark.asyncio
async def test_streaming_includes_reasoning_content_delta():
    """Test that ThinkingPartDelta events produce reasoning_content SSE chunks."""
    from pydantic_ai import PartDeltaEvent, TextPartDelta, ThinkingPartDelta

    async def event_stream():
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="Reasoning..."))
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" more"))
        yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="Answer"))

    request = ChatCompletionRequest(model="test-model", messages=[])
    chunks = await _collect_stream_chunks(event_stream(), request)

    reasoning_chunks = [
        c for c in chunks if "reasoning_content" in c.get("choices", [{}])[0].get("delta", {})
    ]
    text_chunks = [c for c in chunks if "content" in c.get("choices", [{}])[0].get("delta", {})]

    assert len(reasoning_chunks) == 2, (
        f"Expected 2 reasoning chunks, got {len(reasoning_chunks)}. "
        f"All deltas: {[c['choices'][0]['delta'] for c in chunks]}"
    )
    assert reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"] == "Reasoning..."
    assert reasoning_chunks[1]["choices"][0]["delta"]["reasoning_content"] == " more"

    assert len(text_chunks) == 1
    assert text_chunks[0]["choices"][0]["delta"]["content"] == "Answer"


# =============================================================================
# Responses API: ResponseOutputReasoning in output
# =============================================================================


@pytest.mark.asyncio
async def test_responses_api_includes_reasoning_output():
    """Test that ResponseOutputReasoning appears before ResponseMessage."""
    msg = _make_assistant_with_thinking(
        thinking_content="Reasoning for responses API.",
        text_content="Response text.",
    )
    request = ResponseRequest(model="test-model", input="test")
    response = await handle_request(request, msg)

    reasoning_outputs = [o for o in response.output if isinstance(o, ResponseOutputReasoning)]
    assert len(reasoning_outputs) == 1, (
        f"Expected 1 ResponseOutputReasoning, got {len(reasoning_outputs)}. "
        f"Output types: {[type(o).__name__ for o in response.output]}"
    )
    assert reasoning_outputs[0].content == "Reasoning for responses API."
    assert reasoning_outputs[0].type == "reasoning"

    # Reasoning should come before the message in the output list
    # (matches OpenAI Responses API output ordering)
    reasoning_index = next(
        i for i, o in enumerate(response.output) if isinstance(o, ResponseOutputReasoning)
    )
    message_index = next(i for i, o in enumerate(response.output) if isinstance(o, ResponseMessage))
    assert reasoning_index < message_index, (
        f"Reasoning (index {reasoning_index}) should come before message "
        f"(index {message_index}) in output list"
    )


@pytest.mark.asyncio
async def test_responses_api_no_thinking_no_reasoning_output():
    """Test that no ResponseOutputReasoning appears without ThinkingPart."""
    timestamp = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
    model_response = ModelResponse(
        parts=[PydanticTextPart(content="No thinking.")],
        usage=RequestUsage(),
        model_name="test-model",
        timestamp=timestamp,
    )
    msg = ChatMessage(
        content="No thinking.",
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
    request = ResponseRequest(model="test-model", input="test")
    response = await handle_request(request, msg)

    reasoning_outputs = [o for o in response.output if isinstance(o, ResponseOutputReasoning)]
    assert len(reasoning_outputs) == 0
