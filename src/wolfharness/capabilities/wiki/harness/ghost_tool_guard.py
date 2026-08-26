"""Reject textual tool calls and degenerate model reasoning loops.

When a model fails to make a proper structured tool call and instead
emits the call as text (e.g. ``<invoke name="...">`` or
``task_create(...)``), this capability raises ``ModelRetry`` in
``after_model_request`` so the model gets another chance to call the
tool correctly.  The original (ghost) response is preserved in message
history so the model can see what it did wrong.

This prevents the "ghost tool call stall" where a turn ends with only
text output, no real tool call fires, and the pipeline silently deadlocks
because nothing triggers the next turn.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.tools import AgentDepsT


if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

# XML-style patterns that indicate a ghost tool call.  These are the
# tags that models commonly emit when they fail to use the structured
# tool-calling interface and instead write the call as text.
_GHOST_TOOL_RE = re.compile(
    r"<(?:invoke|task_calls|function_call|tool_call|call|parameter)"
    r"[\s>]",
    re.IGNORECASE,
)

_REPETITION_MIN_CHARS = 800
_REPETITION_MIN_OCCURRENCES = 12
_REPETITION_MIN_SHARE = 0.20
_SEGMENT_SPLIT_RE = re.compile(r"[.!?。！？\n\r—]+")
_SEGMENT_NORMALIZE_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)

_RETRY_MESSAGE = (
    "你上一轮把工具调用写成了 XML 文本（如 <invoke>、<task_calls>、<parameter>），"
    "这不是有效的工具调用。你必须通过工具接口调用工具，不要输出 XML 格式的工具调用标签。"
    "请重新执行你刚才意图的工具调用。"
)

_PLAIN_TOOL_RETRY_MESSAGE = (
    "你上一轮把已声明的工具调用写成了普通文本，而没有发出结构化工具调用。"
    "不要描述、模拟或打印调用；请直接通过工具接口执行刚才意图的调用。"
)

_REPETITION_RETRY_MESSAGE = (
    "你上一轮的内部推理陷入了高频重复，持续声明即将行动但没有产生真实工具调用。"
    "请停止复述计划，只选择一个下一步：直接发出一个结构化工具调用，或用一句话明确结束任务。"
)

_EMPTY_RESPONSE_MESSAGE = (
    "你上一轮没有产出任何内容（无文本输出、无工具调用）。请产出有效内容或调用工具，不能返回空响应。"
)

_INSTRUCTIONS = """\
## 工具调用纪律（强制）
- 必须通过工具接口调用工具，绝不要把工具调用写成 XML 文本（如 `<invoke>`、`<task_calls>`、`<function_call>`、`<parameter>`）。
- 每一轮必须以一个真实的工具调用结束，或明确声明任务完成。
- 创建多个任务时使用 `task_create_batch` 工具，不要手写 XML。
"""


def _normalized_segments(content: str) -> list[str]:
    """Return language-neutral sentence/line fingerprints for loop detection."""
    segments: list[str] = []
    for raw_segment in _SEGMENT_SPLIT_RE.split(content.casefold()):
        segment = _SEGMENT_NORMALIZE_RE.sub(" ", raw_segment).strip()
        if len(segment) >= 2:
            segments.append(segment)
    return segments


def _is_degenerate_repetition(content: str) -> bool:
    """Detect sustained exact segment repetition without relying on model phrases."""
    if len(content) < _REPETITION_MIN_CHARS:
        return False

    segments = _normalized_segments(content)
    if not segments:
        return False

    counts = Counter(segments)
    repeated_occurrences = sum(
        count for count in counts.values() if count >= _REPETITION_MIN_OCCURRENCES
    )
    return repeated_occurrences / len(segments) >= _REPETITION_MIN_SHARE


def _contains_plain_tool_call(content: str, tool_names: Iterable[str]) -> bool:
    """Detect a declared tool rendered as ``name(...)`` in assistant text."""
    names = [re.escape(name) for name in tool_names if name]
    if not names:
        return False
    pattern = re.compile(rf"(?<![\w])(?:{'|'.join(names)})\s*\(")
    return pattern.search(content) is not None


class GhostToolCallGuardCapability(AbstractCapability[AgentDepsT]):
    """Detect ghost tool calls in model text output and force a retry.

    When the model emits XML-style tool call patterns as text instead of
    making real structured tool calls, this capability raises
    ``ModelRetry`` so the model gets another chance to call the tool
    correctly.  The original (ghost) response is preserved in message
    history so the model can see what it did wrong.

    Apply this capability to every agent that participates in an
    orchestrated pipeline where a stalled turn (text-only, no tool call)
    causes a silent deadlock.
    """

    def get_instructions(self) -> str | None:
        """Return tool call discipline instructions injected into the system prompt."""
        return _INSTRUCTIONS

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Check model response for ghost tool calls.

        If the response contains text with XML-style tool call patterns
        and no real ``ToolCallPart``s, raise ``ModelRetry`` to force the
        model to retry with proper structured tool calls.

        If the response contains both real tool calls and ghost text, the
        real calls are allowed to proceed — the ghost text is treated as
        narration, not a stalled call.

        Thinking content is checked only for sustained repetition. Merely
        mentioning tool syntax in normal reasoning remains allowed.
        """
        has_real_tool_call = False
        has_ghost_text = False
        has_text_content = False
        text_parts: list[str] = []
        thinking_parts: list[str] = []

        for part in response.parts:
            if isinstance(part, ToolCallPart):
                has_real_tool_call = True
            elif isinstance(part, TextPart):
                text_parts.append(part.content)
                if _GHOST_TOOL_RE.search(part.content):
                    has_ghost_text = True
                if part.content.strip():
                    has_text_content = True
            elif isinstance(part, ThinkingPart):
                thinking_parts.append(part.content)

        if has_ghost_text and not has_real_tool_call:
            raise ModelRetry(_RETRY_MESSAGE)

        response_content = "\n".join((*thinking_parts, *text_parts))
        if _is_degenerate_repetition(response_content):
            raise ModelRetry(_REPETITION_RETRY_MESSAGE)

        if not has_real_tool_call:
            declared_tool_names = (
                tool.name for tool in request_context.model_request_parameters.function_tools
            )
            if _contains_plain_tool_call("\n".join(text_parts), declared_tool_names):
                raise ModelRetry(_PLAIN_TOOL_RETRY_MESSAGE)

        if not has_text_content and not has_real_tool_call:
            raise ModelRetry(_EMPTY_RESPONSE_MESSAGE)

        return response
