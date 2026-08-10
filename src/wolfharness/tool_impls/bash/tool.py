"""Bash command execution tool."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
import uuid

from exxec.events import (
    OutputEvent,
    ProcessCompletedEvent,
    ProcessErrorEvent,
    ProcessStartedEvent,
)

from wolfharness.agents.context import AgentContext  # noqa: TC001
from wolfharness.log import get_logger
from wolfharness.tool_impls.bash.helpers import format_output, truncate_output
from wolfharness.tools.base import Tool, ToolResult


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from exxec import ExecutionEnvironment


logger = get_logger(__name__)

_MAX_COMMAND_LOG_LENGTH = 100


DANGEROUS_COMMANDS = {
    "rm",
    "rmdir",
    "dd",
    "mkfs",
    "format",
    "fdisk",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
    "su",
    "sudo",
    "chmod",
    "chown",
    "chroot",
    "passwd",
    "usermod",
    "userdel",
    "groupmod",
    "killall",
    "kill",
}

DANGEROUS_PATTERNS = [
    r"&&",
    r"\|\|",
    r";",
    r"\|",
    r"`",
    r"\$\(.*\)",  # command substitution $(...)
    r">\s*/dev/",
    r"<\s*/dev/",
    r">\s*/",  # output redirection to root
]


def validate_command(command: str) -> tuple[bool, str | None]:
    """Validate shell command for dangerous patterns.

    Args:
        command: Command string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    cmd_stripped = command.strip()

    # Check for dangerous command operators
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_stripped):
            return False, f"Dangerous command operator detected: {pattern}"

    # Check for dangerous base commands
    parts = cmd_stripped.split()
    if not parts:
        return False, "Empty command"

    base_cmd = parts[0].lower()

    # Check if base command is in dangerous list
    for dangerous in DANGEROUS_COMMANDS:
        if base_cmd == dangerous or base_cmd.startswith(f"{dangerous} "):
            return False, f"Dangerous command detected: {dangerous}"

    return True, None


@dataclass
class BashTool(Tool[ToolResult]):
    """Execute shell commands and return the output.

    A standalone, configurable tool for running bash commands. Can be configured
    with a specific execution environment and output limits.

    Use create_bash_tool() factory for convenient instantiation with defaults.
    """

    # Tool-specific configuration
    env: ExecutionEnvironment | None = None
    """Execution environment to use. Falls back to agent.env if not set."""

    output_limit: int | None = None
    """Default maximum bytes of output to return."""

    timeout: float | None = None
    """Default command timeout in seconds. None means no timeout."""

    def get_callable(self) -> Callable[..., Awaitable[ToolResult]]:
        """Return the execute method as the callable."""
        return self._execute

    def _get_env(self, ctx: AgentContext) -> ExecutionEnvironment:
        """Get execution environment, falling back to agent's env."""
        if self.env is not None:
            return self.env
        return ctx.agent.env

    async def _execute(  # noqa: PLR0915
        self,
        ctx: AgentContext,
        command: str,
        output_limit: int | None = None,
        timeout: float | None = None,
        filter_lines: str | None = None,
    ) -> ToolResult:
        """Execute a shell command and return the output.

        This is the primary tool for running shell commands. It executes the command,
        waits for completion, and returns the full output in a single call.

        Args:
            ctx: Agent context for event emission and environment access
            command: Shell command to execute
            output_limit: Maximum bytes of output to return (overrides default)
            timeout: Command timeout in seconds (overrides default)
            filter_lines: Optional regex pattern to filter output lines
                          (only matching lines returned)
        """
        effective_limit = output_limit or self.output_limit
        effective_timeout = timeout if timeout is not None else self.timeout
        process_id: str | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int | None = None
        error_msg: str | None = None
        env = self._get_env(ctx)

        # Validate command before execution
        is_valid, validation_error = validate_command(command)
        if not is_valid:
            logger.warning(
                "Bash command validation failed",
                command=command[:_MAX_COMMAND_LOG_LENGTH]
                if len(command) > _MAX_COMMAND_LOG_LENGTH
                else command,
                error=validation_error,
            )
            return ToolResult(
                content=f"Command rejected: {validation_error}",
                metadata={"output": "", "exit": None, "description": command},
            )

        # Check if we're running in ACP - terminal streams client-side
        from exxec.acp_provider import ACPExecutionEnvironment

        is_acp = isinstance(env, ACPExecutionEnvironment)

        try:
            async for event in env.stream_command(command, timeout=effective_timeout):
                match event:
                    case ProcessStartedEvent(process_id=pid, command=cmd):
                        process_id = pid
                        await ctx.events.process_started(pid, cmd, success=True)
                    case OutputEvent(process_id=pid, data=data, stream=stream):
                        if stream == "stderr":
                            stderr_parts.append(data)
                        else:
                            stdout_parts.append(data)
                        # Skip progress events for ACP - terminal streams client-side
                        if not is_acp:
                            await ctx.events.process_output(pid, data)
                    case ProcessCompletedEvent(process_id=pid, exit_code=code_):
                        exit_code = code_
                        # Skip exit event for ACP - completion handled by FunctionToolResultEvent
                        if not is_acp:
                            combined = "".join(stdout_parts) + "".join(stderr_parts)
                            await ctx.events.process_exit(pid, exit_code, final_output=combined)
                    case ProcessErrorEvent(error=err, exit_code=code_):
                        error_msg = err
                        exit_code = code_

            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)

            # Apply regex filter if specified
            if filter_lines:
                try:
                    pattern = re.compile(filter_lines)
                    stdout_lines = [
                        line for line in stdout.splitlines(keepends=True) if pattern.search(line)
                    ]
                    stderr_lines = [
                        line for line in stderr.splitlines(keepends=True) if pattern.search(line)
                    ]
                    stdout = "".join(stdout_lines)
                    stderr = "".join(stderr_lines)
                except re.error as regex_err:
                    error_msg = f"Invalid filter regex: {regex_err}"
                    return ToolResult(
                        content=error_msg,
                        metadata={"output": "", "exit": None, "description": command},
                    )

            # Apply output limit if specified
            truncated = False
            if effective_limit:
                stdout, stdout_truncated = truncate_output(stdout, effective_limit)
                stderr, stderr_truncated = truncate_output(stderr, effective_limit)
                truncated = stdout_truncated or stderr_truncated

            # Format error response
            if error_msg:
                output = stdout + stderr if stdout or stderr else ""
                result_output = f"{output}\n\nError: {error_msg}\nExit code: {exit_code}"
                return ToolResult(
                    content=result_output,
                    metadata={"output": output, "exit": exit_code, "description": command},
                )

        except Exception as e:  # noqa: BLE001
            error_id = process_id or f"cmd_{uuid.uuid4().hex[:8]}"
            await ctx.events.process_started(error_id, command, success=False, error=str(e))
            error_msg = f"Error executing command: {e}"
            return ToolResult(
                content=error_msg,
                metadata={"output": "", "exit": None, "description": command},
            )

        # Format success response
        formatted_output = format_output(stdout, stderr, exit_code, truncated)
        combined_output = stdout + stderr
        return ToolResult(
            content=formatted_output,
            metadata={"output": combined_output, "exit": exit_code, "description": command},
        )
