"""Strict JSON-schema tools capability.

Forces ``strict=True`` on model tool definitions before they reach the
provider. Some routers / non-OpenAI-compatible providers (e.g. LiteLLM
forwarding to SGLang) ignore ``strict`` when it is ``None`` and fall back
to a loose grammar that only guarantees "looks like JSON" rather than
parseable JSON — causing occasional 400 errors on malformed tool-call
arguments.

By setting ``strict=True`` explicitly, providers that honor the flag
emit a constrained grammar, eliminating those malformed calls.

Example YAML::

    capabilities:
      - type: strict_tools
        enabled: true
        apply_to_output_tools: false
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, Any

import logfire
from pydantic_ai.capabilities.abstract import AbstractCapability


if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.tools import ToolDefinition


@dataclass
class StrictToolsCapability(AbstractCapability[Any]):
    """Mark every tool definition as ``strict`` before the model sees it.

    Implements :meth:`prepare_tools` so the ``strict`` flag is baked into
    ``ToolDefinition`` instances before request serialization. Also
    implements :meth:`prepare_output_tools` for structured-output
    definitions, gated behind :attr:`apply_to_output_tools` because some
    providers (notably SGLang-compatible output paths) do not support the
    flag.

    Tools that already carry an explicit ``strict`` value are left
    untouched; only ``strict is None`` definitions are upgraded.
    """

    _: KW_ONLY
    enabled: bool = True
    """Master switch — when ``False``, definitions pass through unchanged."""
    apply_to_output_tools: bool = False
    """Also force ``strict`` on output-tool definitions (structured output)."""

    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Force ``strict=True`` on all function-tool definitions.

        Args:
            ctx: The run context.
            tool_defs: Tool definitions about to be sent to the model.

        Returns:
            Definitions upgraded to ``strict=True`` when enabled.
        """
        if not self.enabled:
            return tool_defs
        result = [self._with_strict(td) for td in tool_defs]
        with logfire.span("strict_tools.prepare"):
            for original, upgraded in zip(tool_defs, result):
                logfire.info(
                    "tool strict status",
                    tool_name=upgraded.name,
                    strict_before=original.strict,
                    strict_after=upgraded.strict,
                    changed=original.strict is None and upgraded.strict is True,
                )
        return result

    async def prepare_output_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Force ``strict=True`` on output-tool definitions when enabled.

        Args:
            ctx: The run context.
            tool_defs: Output-tool definitions about to be sent to the model.

        Returns:
            Definitions unchanged unless :attr:`apply_to_output_tools` is set.
        """
        if not self.enabled or not self.apply_to_output_tools:
            return tool_defs
        return [self._with_strict(td) for td in tool_defs]

    @staticmethod
    def _with_strict(tool_def: ToolDefinition) -> ToolDefinition:
        """Return the definition with ``strict=True`` unless already set.

        Args:
            tool_def: The tool definition to upgrade.

        Returns:
            A copy with ``strict=True`` when the original was ``None``,
            otherwise the original unchanged.
        """
        if tool_def.strict is not None:
            return tool_def
        return replace(tool_def, strict=True)
