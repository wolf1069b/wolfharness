"""Tool output budget capability — limits tool output size.

Truncates tool results via ``wrap_tool_execute`` when they exceed
``max_output_chars``, appending a truncation notice.

Relationship with _ToolInterceptCapability
===========================================

``_ToolInterceptCapability`` (``hook_manager.py``) owns ALL tool interception:
pre-tool hooks, post-tool hooks, error handling, and injection.

``ToolOutputBudgetCapability`` implements ``wrap_tool_execute`` to truncate
tool outputs that exceed the budget.

When both are in the capability chain, ``ToolOutputBudgetCapability``'s
``wrap_tool_execute`` runs AFTER ``_ToolInterceptCapability``'s. This is
because ``CombinedCapability`` chains ``wrap_tool_execute`` in **reverse**
order — the last capability in the list wraps the outermost. The injection
order in ``get_agentlet()`` places:

1. ``_ToolInterceptCapability`` first (innermost — error handling + hooks)
2. ``ToolOutputBudgetCapability`` last (outermost — budget truncation)

This means:
1. ``_ToolInterceptCapability`` wraps the tool first (innermost) — handles
   errors, runs hooks, applies modifications.
2. ``ToolOutputBudgetCapability`` wraps the result (outermost) — truncates
   if over budget.

This ordering is **correct**: budget truncation should happen last, after
all post-tool hooks have had a chance to modify the output. If truncation
ran first, hooks would see an artificially shortened output.

No code change is needed — the ordering is already correct by virtue of
capability injection order in ``get_agentlet()``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturn


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import WrapToolExecuteHandler
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition


@dataclass
class ToolOutputBudgetCapability(AbstractCapability[Any]):
    """Limit tool output size per tool call.

    Wraps ``tool_execute`` and truncates string results that exceed
    ``max_output_chars``. A truncation suffix is appended so the model
    knows the output was cut.

    Set ``max_output_chars`` to 0 or a negative value to disable
    truncation entirely.

    Non-string results (dicts, numbers, etc.) are serialized via
    ``json.dumps`` before the length check, ensuring structured tool
    outputs are also budget-controlled.
    """

    max_output_chars: int = 10_000
    truncation_suffix: str = "\n[Tool output truncated by ToolOutputBudgetCapability]"
    """Suffix appended to truncated output to indicate truncation."""

    def __post_init__(self) -> None:
        # max_output_chars <= 0 disables the capability entirely.
        # No minimum enforcement needed when disabled.
        pass

    @property
    def has_wrap_node_run(self) -> bool:
        return False

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        result = await handler(args)
        match result:
            case str():
                return self._truncate(result)
            case ToolReturn():
                return self._truncate_tool_return(result)
            case list():
                return [self._truncate(item) if isinstance(item, str) else item for item in result]
            case _:
                # Non-string, non-list results: serialize to JSON for length check.
                return self._truncate_non_string(result)
        return result

    def _truncate_tool_return(self, result: ToolReturn[Any]) -> Any:
        """Truncate a ``ToolReturn`` without destroying binary content.

        ``ToolReturn.return_value`` and string ``content`` items are
        budget-checked; binary items (``BinaryImage`` etc.) pass through
        untouched — serializing the whole dataclass via ``default=str``
        would reduce images to bytes repr text and erase them.
        """
        if self.max_output_chars <= 0:
            return result
        from dataclasses import replace

        return_value = result.return_value
        if isinstance(return_value, str):
            return_value = self._truncate(return_value)
        content = result.content
        if isinstance(content, str):
            content = self._truncate(content)
        elif content is not None:
            content = [self._truncate(item) if isinstance(item, str) else item for item in content]
        return replace(result, return_value=return_value, content=content)

    def _truncate(self, text: str) -> str:
        if self.max_output_chars <= 0:
            return text
        if len(text) > self.max_output_chars:
            return text[: self.max_output_chars] + self.truncation_suffix
        return text

    def _truncate_non_string(self, result: Any) -> Any:
        """Serialize a non-string result to JSON and truncate if over budget.

        If the serialized form exceeds ``max_output_chars``, the truncated
        JSON string (with suffix) is returned in place of the original
        object. Otherwise the original object is returned unchanged.
        """
        if self.max_output_chars <= 0:
            return result
        try:
            serialized = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return result
        if len(serialized) > self.max_output_chars:
            return serialized[: self.max_output_chars] + self.truncation_suffix
        return result

    async def for_run(self, ctx: RunContext[Any]) -> ToolOutputBudgetCapability:
        return ToolOutputBudgetCapability(
            max_output_chars=self.max_output_chars,
            truncation_suffix=self.truncation_suffix,
        )
