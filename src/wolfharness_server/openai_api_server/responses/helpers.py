"""OpenAI-compatible responses endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from wolfharness_server.openai_api_server.responses.models import (
    Response,
    ResponseMessage,
    ResponseOutputReasoning,
    ResponseOutputText,
    ResponseToolCall,
    ResponseUsage,
)


if TYPE_CHECKING:
    from wolfharness.messaging.messages import ChatMessage
    from wolfharness_server.openai_api_server.responses.models import ResponseRequest


async def handle_request(request: ResponseRequest, message: ChatMessage[Any]) -> Response:
    """Build a Response from a completed ChatMessage.

    Extracts both text content and thinking/reasoning content (if present)
    from the model messages. Reasoning output is placed before the message
    to match OpenAI's Responses API output ordering.
    """
    content = message.content if isinstance(message.content, str) else str(message.content)

    # Extract thinking/reasoning content from model messages
    from pydantic_ai import ThinkingPart

    reasoning_parts: list[str] = []
    for model_msg in message.messages:
        if isinstance(model_msg, dict):
            parts = model_msg.get("parts") or []
            reasoning_parts.extend(
                part_dict.get("content") or ""
                for part_dict in parts
                if isinstance(part_dict, dict) and part_dict.get("part_kind") == "thinking"
            )
        else:
            reasoning_parts.extend(
                part.content
                for part in model_msg.parts
                if isinstance(part, ThinkingPart) and part.content
            )

    output: list[ResponseMessage | ResponseToolCall | ResponseOutputReasoning] = []

    # Reasoning comes first (matches OpenAI Responses API output order)
    if reasoning_parts:
        output.append(ResponseOutputReasoning(content="\n".join(reasoning_parts)))

    # Then the message with text content
    text = ResponseOutputText(text=content)
    output_msg = ResponseMessage(id=f"msg_{uuid4().hex}", role="assistant", content=[text])
    output.append(output_msg)

    calls = [
        ResponseToolCall(type=f"{tc.tool_name}_call", id=tc.tool_call_id)
        for tc in message.get_tool_calls()
    ]
    output = calls + output

    usage_info: ResponseUsage | None = None
    if message.cost_info and (token_usage := message.cost_info.token_usage):
        # Map the keys correctly from agent's dict to ResponseUsage TypedDict
        input_tk = token_usage.input_tokens
        output_tk = token_usage.output_tokens
        total_tk = token_usage.total_tokens

        usage_info = ResponseUsage(
            input_tokens=input_tk,
            input_tokens_details={},
            output_tokens=output_tk,
            output_tokens_details={},
            total_tokens=total_tk,
        )

    return Response(
        model=request.model,
        output=output,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        tools=request.tools,
        tool_choice=request.tool_choice,
        usage=usage_info,
        metadata=request.metadata,
    )
