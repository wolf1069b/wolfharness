"""QuestionCapability — user interaction question tool with YAML schema overrides.

Provides the ``question`` tool backed by
``wolfharness_toolsets.builtin.question_tools.QuestionTools``.
Accepts optional YAML schema files to override the LLM-facing parameter
description, mirroring the ``BackgroundTaskCapability`` pattern.

Declared via the ``question`` entry point in ``pyproject.toml`` so consumers
can reference it in YAML config as ``type: question``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import logfire
from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    NativeTool,
    ProcessHistory,
)
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.agents.context import AgentContext
from wolfharness.tools.base import ToolResult  # noqa: TC001
from wolfharness.utils.tool_schema import apply_params_schema, load_tool_schema
from wolfharness_config.context import get_config_dir
from wolfharness_toolsets.builtin.question_tools import QuestionTools


if TYPE_CHECKING:
    from schemez.functionschema import OpenAIFunctionDefinition


class QuestionCapability(AbstractCapability[AgentContext]):
    """Capability providing the user interaction ``question`` tool.

    Wraps :class:`~wolfharness_toolsets.builtin.question_tools.QuestionTools`
    and applies optional YAML schema overrides for richer LLM-facing
    parameter descriptions.

    Provides:
    - ``question``: Multi-question XML questionnaire tool with enum/multi/input types

    Schema overrides are controlled by the ``schemas`` dict and ``enabled_tools``
    list, mirroring ``BackgroundTaskCapability``.
    """

    def __init__(
        self,
        schemas: dict[str, str] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize the question capability.

        Args:
            schemas: Optional dictionary mapping tool names to schema file
                paths.  Expected key: ``"question"``.  Paths are resolved
                relative to the config directory using ``CONFIG_DIR``.
            enabled_tools: Optional list of tools to enable.  If ``None``,
                all tools whose schemas are loaded are enabled.
                Expected value: ``"question"``.
        """
        self._schemas = schemas or {}

        # Single canonical QuestionTools instance shared across tool invocations.
        self._question_tools = QuestionTools(name="question_tools")

        # Schema loading
        self._question_schema: OpenAIFunctionDefinition | None = None

        if schemas and (q_path := schemas.get("question")) is not None:
            self._question_schema = self._resolve_and_load_schema(q_path)

        # Determine enabled tools
        available: list[str] = []
        if not self._schemas:
            available = ["question"]
        elif self._question_schema is not None or "question" in self._schemas:
            available.append("question")

        if enabled_tools is not None:
            self._enabled_tools = [t for t in enabled_tools if t in available]
        else:
            self._enabled_tools = available

    @staticmethod
    def _resolve_and_load_schema(schema_path_str: str) -> OpenAIFunctionDefinition:
        """Resolve a schema path relative to ``CONFIG_DIR`` and load it.

        Args:
            schema_path_str: The schema file path (absolute or relative
                to ``CONFIG_DIR``).

        Returns:
            The loaded schema as an ``OpenAIFunctionDefinition``.

        Raises:
            FileNotFoundError: If the schema file doesn't exist.
            ValueError: If the schema file can't be parsed.
        """
        schema_path = Path(schema_path_str)
        if not schema_path.is_absolute():
            config_dir = get_config_dir()
            if config_dir is not None:
                schema_path = Path(str(config_dir)) / schema_path
        result = load_tool_schema(str(schema_path))
        if result is None:
            msg = f"Tool schema at {schema_path} loaded as None"
            raise ValueError(msg)
        return result

    def get_toolset(self) -> AgentToolset[AgentContext] | None:
        """Return ``FunctionToolset`` with the enabled question tool.

        The tool callable is sourced from the wolfharness ``QuestionTools``
        instance, ensuring a single canonical implementation.  YAML schema
        overrides are applied on top for richer LLM-facing descriptions.
        """
        tools: list[Tool[AgentContext]] = []

        if "question" in self._enabled_tools:
            name = (
                self._question_schema.get("name") if self._question_schema else None
            ) or "question"
            description = (
                self._question_schema.get("description") if self._question_schema else None
            ) or "Ask the user one or more structured questions."
            # pydantic_ai can't schema ToolResult (plain dataclass); runtime
            # conversion in _ToolInterceptCapability.wrap_tool_execute handles it.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r"Could not generate return schema for .+",
                )
                tool = Tool(
                    self._question,
                    name=name,
                    description=description,
                    metadata={"category": "other"},
                )
            tools.append(apply_params_schema(tool, self._question_schema))

        if not tools:
            return None
        return FunctionToolset(tools)

    def get_ordering(self) -> CapabilityOrdering | None:
        """Declare middleware chain position."""
        return CapabilityOrdering(wrapped_by=[ProcessHistory, NativeTool])

    # ---- Tool wrapper ----
    # This wraps the canonical tool function from wolfharness's QuestionTools,
    # adapting RunContext[AgentContext] → AgentContext for direct invocation.

    @logfire.instrument("question.capability.question")
    async def _question(
        self,
        ctx: RunContext[AgentContext],
        questions: str,
    ) -> ToolResult:
        """Wrap ``QuestionTools.question`` to accept ``RunContext``."""
        agent_ctx = replace(
            ctx.deps,
            tool_name=ctx.tool_name,
            tool_call_id=ctx.tool_call_id,
            tool_input={"questions": questions},
        )
        return await self._question_tools.question(agent_ctx, questions)


__all__ = ["QuestionCapability"]
