"""ACP skill commands bridge for exposing skills as ACP slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import logfire

from wolfharness.log import get_logger
from wolfharness_server.opencode_server.skill_bridge import create_skill_command


logger = get_logger(__name__)

if TYPE_CHECKING:
    from slashed import Command as SlashedCommand

    from wolfharness.skills.command import SkillCommand


class ACPSkillBridge:
    """Bridge class that maps SkillCommand to executable SlashedCommand.

    This class exposes skills as ACP slash commands by converting
    SkillCommand instances to SlashedCommand objects that can be
    registered in a CommandStore and executed via execute_slash_command().

    The executor reuses create_skill_command() from the OpenCode skill
    bridge, which is protocol-agnostic — it loads skill instructions
    and injects them into ctx.data.node.staged_content.

    Attributes:
        _commands: Dictionary mapping command names to SlashedCommand instances.
    """

    def __init__(self) -> None:
        """Initialize the bridge with an empty command store."""
        self._commands: dict[str, SlashedCommand] = {}

    @logfire.instrument("acp_skill_bridge_handle_change")
    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle skill command add/remove changes.

        This method matches the CommandChangeHandler signature and is called
        when skills are added or removed from the SkillsRegistry.

        Args:
            name: The name of the command being changed.
            command: The SkillCommand instance if added, None if removed.
        """
        if command is None:
            self._commands.pop(name, None)
        else:
            logger.debug("Converting skill command %s to SlashedCommand", name)
            self._commands[name] = create_skill_command(command)
        logger.debug("ACPSkillBridge has %d commands", len(self._commands))

    def get_commands(self) -> list[SlashedCommand]:
        """Return list of executable SlashedCommand objects.

        Returns:
            A list of SlashedCommand instances for all stored commands.
        """
        commands = list(self._commands.values())
        logger.debug(
            "Retrieved ACP skill commands",
            command_count=len(commands),
            command_names=[cmd.name for cmd in commands],
        )
        return commands

    def get_command_names(self) -> set[str]:
        """Return the set of currently registered command names.

        Returns:
            A set of command name strings.
        """
        return set(self._commands.keys())
