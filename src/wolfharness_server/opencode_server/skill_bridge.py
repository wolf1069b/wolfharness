"""OpenCode skill bridge for exposing skills as slashed Commands."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

import logfire
from slashed import Command as SlashedCommand, CommandContext

from wolfharness.log import get_logger


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from wolfharness.skills.command import SkillCommand


class SkillCommandWrapper:
    """Wrapper exposing SkillCommand properties for OpenCode integration."""

    def __init__(self, skill_cmd: SkillCommand) -> None:
        self._skill_cmd = skill_cmd
        self.name = skill_cmd.name  # No prefix - matches OpenCode protocol
        self.description = skill_cmd.description
        self.category = skill_cmd.category
        self.skill_uri = skill_cmd.resolved_skill_uri
        """The skill:// URI for this command."""

    @property
    def skill(self) -> Any:
        """Access the underlying Skill object for template/content."""
        return self._skill_cmd.skill


def _hash_args(args: list[str], kwargs: dict[str, str]) -> str:
    """Hash arguments for privacy in logging.

    Args:
        args: The positional arguments.
        kwargs: The keyword arguments.

    Returns:
        A short hash prefix for tracking purposes.
    """
    content = str(args) + str(sorted(kwargs.items()))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_skill_command(
    skill_cmd: SkillCommand, skill_provider: Any | None = None
) -> SlashedCommand:
    """Create a slashed Command from a SkillCommand.

    Args:
        skill_cmd: The skill command to wrap.
        skill_provider: Optional skill provider for fetching instructions
            from virtual/MCP-based skills.

    Returns:
        A slashed Command that loads and executes the skill.
    """
    logger.debug("SkillCommand %s initialized", skill_cmd.name)

    async def execute_skill(
        ctx: CommandContext[Any],
        args: list[str],
        kwargs: dict[str, str],
    ) -> None:
        """Execute the skill command."""
        start_time = time.time()
        args_hash = _hash_args(args, kwargs)
        skill_uri = skill_cmd.resolved_skill_uri

        with logfire.span(
            "skill_command_execute",
            skill_name=skill_cmd.name,
            skill_uri=skill_uri,
            protocol="opencode",
            args_hash=args_hash,
        ):
            logger.info(
                "Executing skill command",
                skill_name=skill_cmd.name,
                skill_uri=skill_uri,
                args_hash=args_hash,
                arg_count=len(args),
                kwarg_count=len(kwargs),
            )

            # Load skill instructions - use provider for virtual skills
            instructions = ""
            if skill_provider is not None:
                try:
                    instructions = await skill_provider.get_skill_instructions(skill_cmd.name)
                except Exception:  # noqa: BLE001
                    # Fall back to local load if provider fetch fails
                    try:
                        instructions = skill_cmd.skill.load_instructions()
                    except ValueError:
                        instructions = ""
            else:
                try:
                    instructions = skill_cmd.skill.load_instructions()
                except ValueError:
                    instructions = ""
            duration_ms = (time.time() - start_time) * 1000

            if instructions:
                await ctx.print(f"Loading skill: {skill_cmd.name} ({skill_uri})")
                logger.info(
                    "Skill command executed successfully",
                    skill_name=skill_cmd.name,
                    skill_uri=skill_uri,
                    duration_ms=round(duration_ms, 2),
                    has_instructions=True,
                )
                # Build complete prompt with instructions AND user request (args)
                # This matches OpenCode's pattern: skill-instruction + user-request
                user_request = " ".join(args)
                full_prompt = f"""<skill-instruction>
{instructions}
</skill-instruction>

<user-request>
{user_request}
</user-request>"""
                # Inject complete prompt into staged_content for agent processing
                if (
                    hasattr(ctx, "data")
                    and ctx.data is not None
                    and hasattr(ctx.data, "node")
                    and ctx.data.node is not None
                    and hasattr(ctx.data.node, "staged_content")
                ):
                    ctx.data.node.staged_content.add_text(full_prompt)
                    logger.debug(
                        "Injected skill instructions into staged_content",
                        skill_name=skill_cmd.name,
                        instruction_length=len(instructions),
                    )
            else:
                await ctx.print(f"Skill {skill_cmd.name} has no instructions ({skill_uri})")
                logger.warning(
                    "Skill command executed but no instructions found",
                    skill_name=skill_cmd.name,
                    skill_uri=skill_uri,
                    duration_ms=round(duration_ms, 2),
                )

    return SlashedCommand.from_raw(
        execute_skill,
        name=skill_cmd.name,  # No prefix - matches OpenCode protocol
        description=skill_cmd.description,
        category="skill",
        usage=skill_cmd.input_hint,
    )


class OpenCodeSkillBridge:
    """Bridge managing skill commands for OpenCode's slashed CommandStore."""

    def __init__(self, skill_provider: Any | None = None) -> None:
        """Initialize the skill bridge.

        Args:
            skill_provider: Optional skill provider for fetching instructions
                from virtual/MCP-based skills.
        """
        self._commands: dict[str, SlashedCommand] = {}
        self._skill_cmds: dict[str, SkillCommand] = {}
        self._on_change_callbacks: list[Callable[[], None]] = []
        self._skill_provider = skill_provider

    def on_commands_changed(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when commands change.

        Args:
            callback: Function to call when commands are added or removed.
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self) -> None:
        """Notify all registered callbacks of a command change."""
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Error notifying command change callback")

    @logfire.instrument("opencode_skill_bridge_handle_change")
    def handle_change(self, name: str, command: SkillCommand | None) -> None:
        """Handle skill command add/remove changes.

        Matches the CommandChangeHandler signature from SkillCommandRegistry.

        Args:
            name: The name of the skill command.
            command: The SkillCommand if adding, None if removing.
        """
        if command is None:
            self._commands.pop(name, None)
            self._skill_cmds.pop(name, None)
            logger.info(
                "Skill command removed from OpenCode bridge",
                skill_name=name,
                total_commands=len(self._commands),
            )
        else:
            self._commands[name] = create_skill_command(command, self._skill_provider)
            self._skill_cmds[name] = command
            logger.info(
                "Skill command wrapped for OpenCode",
                skill_name=name,
                skill_uri=command.resolved_skill_uri,
                total_commands=len(self._commands),
            )
        # Notify registered callbacks of the change
        self._notify_change()

    def get_commands(self) -> list[SlashedCommand]:
        """Return all commands as slashed Commands."""
        commands = list(self._commands.values())
        logger.debug(
            "Retrieved OpenCode skill commands",
            command_count=len(commands),
            command_names=[cmd.name for cmd in commands],
        )
        return commands

    def get_skill_commands(self) -> list[SkillCommand]:
        """Return all skill commands with full skill info (for template access)."""
        return list(self._skill_cmds.values())

    @logfire.instrument("opencode_skill_bridge_get_command")
    def get_command(self, name: str) -> SlashedCommand | None:
        """Get command by name (with or without 'skill:' prefix).

        Args:
            name: The command name to look up.

        Returns:
            The command if found, None otherwise.
        """
        skill_name = name.removeprefix("skill:") if name.startswith("skill:") else name
        command = self._commands.get(skill_name)
        logger.debug(
            "Retrieved OpenCode skill command",
            requested_name=name,
            skill_name=skill_name,
            found=command is not None,
        )
        return command
