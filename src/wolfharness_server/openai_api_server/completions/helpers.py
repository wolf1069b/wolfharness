"""Helpers for OpenAI-compatible API server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyenv
from pydantic_ai import PartDeltaEvent, TextPartDelta, ThinkingPartDelta

from wolfharness.agents.events import UserMessageInsertedEvent
from wolfharness.agents.events.events import StepUsageEvent
from wolfharness.log import get_logger
from wolfharness.utils.time_utils import now_ms


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from wolfharness_server.openai_api_server.completions.models import ChatCompletionRequest

logger = get_logger(__name__)


async def stream_response(
    events: AsyncIterator[Any],
    request: ChatCompletionRequest,
) -> AsyncGenerator[str]:
    """Generate streaming response chunks from an async event iterator.

    Args:
        events: An async iterator yielding agent stream events
            (e.g. from ``SessionPool.run_stream()``).
        request: The original chat completion request for model metadata.
    """
    ts_ms = now_ms()
    response_id = f"chatcmpl-{ts_ms}"
    created = ts_ms // 1000

    try:
        # First chunk with role
        choice = {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        first_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {anyenv.dump_json(first_chunk)}\n\n"
        async for event in events:
            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=chunk)):
                    # Skip empty chunks
                    if not chunk:
                        continue
                    delta = {"content": chunk}
                    choice = {"index": 0, "delta": delta, "finish_reason": None}
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [choice],
                    }
                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
                case PartDeltaEvent(delta=ThinkingPartDelta(content_delta=chunk)):
                    # Stream thinking/reasoning content as reasoning_content delta
                    if not chunk:
                        continue
                    delta = {"reasoning_content": chunk}
                    choice = {"index": 0, "delta": delta, "finish_reason": None}
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [choice],
                    }
                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
                case UserMessageInsertedEvent():
                    pass  # User message insertions don't produce completion chunks
                case StepUsageEvent(step_usage=su):
                    # Emit per-step token usage as a streaming chunk with usage field
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": su.input_tokens,
                            "completion_tokens": su.output_tokens,
                            "total_tokens": su.total_tokens,
                        },
                    }
                    yield f"data: {anyenv.dump_json(chunk_data)}\n\n"
        final_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {anyenv.dump_json(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error during streaming response")
        delta = {"content": f"Error: {e!s}"}
        choice = {"index": 0, "delta": delta, "finish_reason": "error"}
        error_chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [choice],
        }
        yield f"data: {anyenv.dump_json(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
