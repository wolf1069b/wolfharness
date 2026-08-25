"""Tool argument sanitizer capability — prevent provider 400s from bad JSON.

Some models (e.g. deepseek-v4-flash) occasionally emit tool call arguments
that are not valid JSON. pydantic-ai tolerates this locally (wrapping the
raw string as ``INVALID_JSON``) so the run does not crash — but the poisoned
``ToolCallPart`` stays in message history, and when the history is
re-serialized back to the provider on a subsequent model request, the
provider rejects it with HTTP 400: "Assistant tool call function.arguments
must be valid JSON."

The existing turn-boundary repairs (``inject_cancelled_tool_results`` in
``wolfharness/orchestrator/run.py`` and ``sanitize_tool_call_args_in_messages``
in ``wolfharness/orchestrator/event_mapper.py``) only sanitize at turn
boundaries / session restore. This capability closes the mid-run gap by
sanitizing message history in ``before_model_request`` — i.e. before EVERY
model request — so invalid arguments never reach the provider.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from wolfharness.utils.pydantic_ai_helpers import has_invalid_json_args


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.models import ModelRequestContext

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ToolArgSanitizeCapability(AbstractCapability[Any]):
    """Replace invalid-JSON tool call arguments before each model request.

    Scans message history for ``ToolCallPart`` instances whose ``args`` is a
    non-empty string that cannot be parsed as JSON, and replaces them with
    ``{}`` so the serialized history is always accepted by the provider.

    Dict args, ``None`` args, and empty strings are left untouched.
    """

    enabled: bool = True
    """Master switch. When ``False``, the history is passed through untouched."""

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Sanitize invalid tool-call args in history before the request."""
        if not self.enabled:
            return request_context
        new_messages: list[ModelMessage] = []
        changed = False
        for message in request_context.messages:
            if isinstance(message, ModelResponse):
                sanitized = self._sanitize_response(message)
                if sanitized is not message:
                    changed = True
                new_messages.append(sanitized)
            else:
                new_messages.append(message)

        if not changed:
            return request_context
        return dataclasses.replace(request_context, messages=new_messages)

    def _sanitize_response(self, response: ModelResponse) -> ModelResponse:
        """Replace invalid-JSON tool call parts in one response message."""
        new_parts: list[Any] = []
        changed = False
        for part in response.parts:
            if isinstance(part, ToolCallPart) and has_invalid_json_args(part):
                logger.warning(
                    "Sanitizing invalid JSON tool call args for tool %r (call %r)",
                    part.tool_name,
                    part.tool_call_id,
                )
                new_parts.append(
                    ToolCallPart(
                        tool_name=part.tool_name,
                        args={},
                        tool_call_id=part.tool_call_id,
                    )
                )
                changed = True
            else:
                new_parts.append(part)

        if not changed:
            return response
        return dataclasses.replace(response, parts=new_parts)
