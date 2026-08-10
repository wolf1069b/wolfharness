"""AG-UI skill tools bridge for exposing skills as AG-UI Tools."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from ag_ui.core import Tool
import logfire

from wolfharness.log import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from wolfharness.skills.command import SkillCommand


def _hash_args(args: str) -> str:
    """Hash arguments for privacy in logging.

    Args:
        args: The arguments string to hash.

    Returns:
        A short hash prefix for tracking purposes.
    """
    return hashlib.sha256(args.encode()).hexdigest()[:16]


class AGUISkillToolAdapter:
    """Adapter converting SkillCommand to AG-UI Tool (OpenAI function format)."""

    def __init__(self, skill_cmd: SkillCommand) -> None:
        """Initialize adapter with a SkillCommand.

        Args:
            skill_cmd: The skill command to adapt to AG-UI Tool format.
        """
        self.skill_cmd = skill_cmd

    @logfire.instrument("agui_tool_adapter_convert")
    def to_agui_tool(self) -> Tool:
        """Convert SkillCommand to AG-UI Tool.

        Tool name format: skill__{skill_name} (double underscore)
        Parameters: single arguments: string field

        Returns:
            An AG-UI Tool instance representing the skill.
        """
        tool = Tool(
            name=f"skill__{self.skill_cmd.name}",
            description=self.skill_cmd.description,
            parameters={
                "type": "object",
                "properties": {
                    "arguments": {"type": "string", "description": self.skill_cmd.input_hint}
                },
                "required": ["arguments"],
            },
        )
        logger.debug(
            "Converted skill to AG-UI Tool",
            skill_name=self.skill_cmd.name,
            tool_name=tool.name,
            has_input_hint=bool(self.skill_cmd.input_hint),
        )
        return tool


class AGUISkillBridge:
    """Bridge managing multiple skill tools for AG-UI."""

    def __init__(self) -> None:
        """Initialize the bridge with empty adapter store."""
        self._adapters: dict[str, AGUISkillToolAdapter] = {}

    @logfire.instrument("agui_skill_bridge_handle_change")
    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle skill command add/remove changes.

        Matches CommandChangeHandler signature. When command is None,
        the skill is removed. Otherwise, a new adapter is created.

        Args:
            name: The name of the skill command.
            command: The SkillCommand if adding, None if removing.
        """
        if command is None:
            self._adapters.pop(name, None)
        else:
            logger.debug("Converting skill command %s to AG-UI Tool", name)
            self._adapters[name] = AGUISkillToolAdapter(command)
        logger.debug("AGUISkillBridge has %d tools", len(self._adapters))

    def get_tools(self) -> list[Tool]:
        """Return list of AG-UI Tools from all adapters.

        Returns:
            A list of Tool instances for all registered skills.
        """
        tools = [adapter.to_agui_tool() for adapter in self._adapters.values()]
        logger.debug(
            "Retrieved AG-UI tools",
            tool_count=len(tools),
            tool_names=[tool.name for tool in tools],
        )
        return tools

    @logfire.instrument("agui_skill_bridge_get_handler")
    def get_handler(self, tool_name: str) -> AGUISkillToolAdapter | None:
        """Get adapter for a given tool name.

        Tool name format: skill__{skill_name}

        Args:
            tool_name: The full AG-UI tool name including prefix.

        Returns:
            The adapter if found, None otherwise.
        """
        if not tool_name.startswith("skill__"):
            logger.debug("Invalid tool name format", tool_name=tool_name)
            return None
        skill_name = tool_name.removeprefix("skill__")
        adapter = self._adapters.get(skill_name)
        logger.debug(
            "Retrieved AG-UI tool handler",
            tool_name=tool_name,
            skill_name=skill_name,
            found=adapter is not None,
        )
        return adapter
