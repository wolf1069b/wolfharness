"""ACP (Agent Client Protocol) session management for wolfharness.

This module provides session lifecycle management, state tracking, and coordination
between agents and ACP clients through the JSON-RPC protocol.

The implementation is split across mixin modules:
- :mod:`session_lifecycle` — initialization, MCP setup, prompt processing, close
- :mod:`session_events` — state update handling, commands update
- :mod:`session_agent_mgmt` — agent switching, command registration, slash command execution
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING, Any, Literal, cast

from exxec.acp_provider import ACPExecutionEnvironment
from slashed import CommandStore

from acp.agent.acp_requests import ACPRequests
from acp.agent.notifications import ACPNotifications
from acp.filesystem import ACPFileSystem
from acp.schema import AvailableCommand, ClientCapabilities
from wolfharness import Agent
from wolfharness.agents.acp_agent import ACPAgent
from wolfharness.commands.base import NodeCommand
from wolfharness.log import get_logger
from wolfharness_server.acp_server.input_provider import ACPInputProvider
from wolfharness_server.acp_server.session_agent_mgmt import ACPSessionAgentMgmtMixin
from wolfharness_server.acp_server.session_events import ACPSessionEventsMixin
from wolfharness_server.acp_server.session_lifecycle import ACPSessionLifecycleMixin


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pydantic_ai import UserContent
    from slashed import BaseCommand

    from acp import Client, RequestPermissionRequest, RequestPermissionResponse
    from acp.schema import (
        Implementation,
        McpServer,
        StopReason,
        Usage,
    )
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.common_types import PathReference
    from wolfharness.host.context import HostContext
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
    from wolfharness_server.acp_server.event_converter import ACPEventConverter
    from wolfharness_server.acp_server.session_manager import ACPSessionManager

logger = get_logger(__name__)

# Maximum length for internal provider name keys (kept as local constant
# after removing from uri_resolver — provider segment concept was removed in D9).
_MAX_PROVIDER_NAME_LENGTH = 63
SLASH_PATTERN = re.compile(r"^/([\w-]+)(?:\s+(.*))?$")

# Zed-specific instructions for code references
ZED_CLIENT_PROMPT = """\
## Code References

When referencing code locations in responses, use markdown links with `file://` URLs:

- **File**: `[filename](file:///absolute/path/to/file.py)`
- **Line range**: `[filename#L10-25](file:///absolute/path/to/file.py#L10:25)`
- **Single line**: `[filename#L10](file:///absolute/path/to/file.py#L10:10)`
- **Directory**: `[dirname/](file:///absolute/path/to/dir/)`

Line range format is `#L<start>:<end>` (1-based, inclusive).

Use these clickable references instead of inline code blocks when pointing to specific \
code locations. For showing actual code content, still use fenced code blocks.

## Zed-specific URLs

In addition to `file://` URLs, these `zed://` URLs work in the agent context:

- **File reference**: `[text](zed:///agent/file?path=/absolute/path/to/file.py)`
- **Selection**: `[text](zed:///agent/selection?path=/absolute/path/to/file.py#L10:25)`
- **Symbol**: `[text](zed:///agent/symbol/function_name?path=/absolute/path/to/file.py#L10:25)`
- **Directory**: `[text](zed:///agent/directory?path=/absolute/path/to/dir)`

Query params must be URL-encoded (spaces → `%20`). Paths must be absolute.
"""


def get_all_commands() -> Sequence[BaseCommand]:
    """Return empty command list to align with OpenCode behavior.

    All built-in framework commands are hidden to keep ACP consistent
    with OpenCode, which does not register wolfharness_commands at all.
    """
    return []


def _is_slash_command(text: str) -> bool:
    """Check if text starts with a slash command."""
    return bool(SLASH_PATTERN.match(text.strip()))


def split_commands(
    contents: Sequence[UserContent | PathReference],
    command_store: CommandStore,
) -> tuple[list[str], list[UserContent | PathReference]]:
    """Split content into local slash commands and pass-through content.

    Only commands that exist in the local command_store are extracted.
    Remote commands (from nested ACP agents) stay in non_command_content
    so they flow through to the agent and reach the nested server.
    """
    commands: list[str] = []
    non_command_content: list[UserContent | PathReference] = []
    for item in contents:
        # Check if this is a LOCAL command we handle
        if (
            isinstance(item, str)
            and _is_slash_command(item)
            and (match := SLASH_PATTERN.match(item.strip()))
            and command_store.get_command(match.group(1))
        ):
            commands.append(item.strip())
        else:
            # Not a local command - pass through (may be remote command or regular text)
            non_command_content.append(item)
    return commands, non_command_content


def infer_stop_reason(error_msg: str) -> StopReason:
    """Infers the reason for stopping the session based on the error message."""
    if "request_limit" in error_msg:
        return "max_turn_requests"
    if any(limit in error_msg for limit in ["tokens_limit", "token_limit"]):
        return "max_tokens"
    # Tool call limits don't have a direct ACP stop reason, treat as refusal
    if "tool_calls_limit" in error_msg or "tool call" in error_msg:
        return "refusal"
    return "max_tokens"  # Default to max_tokens for other usage limits


@dataclass
class ACPSession(
    ACPSessionAgentMgmtMixin,
    ACPSessionEventsMixin,
    ACPSessionLifecycleMixin,
):
    """Individual ACP session state and management.

    Manages the lifecycle and state of a single ACP session, including:
    - Agent instance and conversation state
    - Working directory and environment
    - MCP server connections
    - File system bridge for client operations
    - Tool execution and streaming updates
    """

    session_id: str
    """Unique session identifier"""

    agent: BaseAgent[Any, Any]
    """Currently active agent instance.

    The agent carries its own pool reference via agent.host_context,
    which is used for agent switching and pool-level operations.
    """

    cwd: str
    """Working directory for the session"""

    client: Client
    """External library Client interface for operations"""

    acp_agent: AgentPoolACPAgent
    """ACP agent instance for capability tools"""

    mcp_servers: Sequence[McpServer] | None = None
    """Optional MCP server configurations"""

    client_capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)
    """Client capabilities for tool registration"""

    client_info: Implementation | None = None
    """Client implementation info (name, version, title)"""

    manager: ACPSessionManager | None = None
    """Session manager for managing sessions. Used for session management commands."""

    subagent_display_mode: Literal["legacy", "zed", "qwen"] = "legacy"
    """How to display subagent output:
    - 'legacy': Default display mode using tool_box semantics
    - 'zed': Zed-compatible display mode
    """

    raw_input_mode: Literal["dict", "skip", "json_str"] = "dict"
    """How to emit tool call raw_input:
    - 'dict': Parse args as dict (default)
    - 'skip': Omit raw_input until tool call is complete
    - 'json_str': Emit raw_input as a JSON string
    """

    checkpoint_enabled: bool = False
    """Whether durable elicitation checkpointing is enabled for this session.

    When True, the ACPInputProvider.supports_durable_elicitation property
    returns True, enabling the two-level interception path that checkpoints
    elicitation requests for crash recovery.
    """

    def __post_init__(self) -> None:
        """Initialize session state and set up providers."""
        self.mcp_servers = self.mcp_servers or []
        self.log = logger.bind(session_id=self.session_id)
        self._task_lock = asyncio.Lock()
        self._cancelled = False
        self._current_converter: ACPEventConverter | None = None
        self.last_usage: Usage | None = None
        self.fs = ACPFileSystem(
            self.client,
            session_id=self.session_id,
            client_capabilities=self.client_capabilities,
        )
        self.command_store = CommandStore(commands=get_all_commands())
        self.command_store._initialize_sync()
        self._update_callbacks: list[Callable[[], None]] = []
        self._remote_commands: list[AvailableCommand] = []

        # Skill bridge: converts SkillCommand → SlashedCommand for command_store
        from wolfharness_server.acp_server.commands.skill_commands import ACPSkillBridge

        self._skill_bridge: ACPSkillBridge = ACPSkillBridge()
        self._skill_change_task: asyncio.Task[None] | None = None
        self._skill_register_lock = asyncio.Lock()

        # CRITICAL: Initialize requests and acp_env BEFORE agent mutation
        self.notifications = ACPNotifications(client=self.client, session_id=self.session_id)
        self.requests = ACPRequests(client=self.client, session_id=self.session_id)
        self.input_provider = ACPInputProvider(self)
        self.acp_env = ACPExecutionEnvironment(fs=self.fs, requests=self.requests, cwd=self.cwd)

        # Inject Zed-specific instructions if client is Zed
        if self.client_info and self.client_info.name and "zed" in self.client_info.name.lower():
            self.agent.staged_content.add_text(ZED_CLIENT_PROMPT)

        # Only mutate THIS session's agent, not all pool agents
        self.agent.env = self.acp_env
        # CRITICAL: Set the real input provider (overrides temp None from creation)
        self.agent._input_provider = self.input_provider
        if isinstance(self.agent, Agent):
            self.agent.sys_prompts.prompts.append(self.get_cwd_context)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
            # Wire ACP MCP connection manager for per-session cleanup tracking.
            # Without this, MCPManager.cleanup_session() can never delegate
            # to AcpMcpConnectionManager.cleanup_session(), leaking per-session
            # ACP stream pairs and reverse-index entries.
            self.agent.mcp._acp_mcp_manager = self.acp_agent._mcp_manager
        if isinstance(self.agent, ACPAgent):

            async def permission_callback(
                params: RequestPermissionRequest,
            ) -> RequestPermissionResponse:
                forwarded = params.model_copy(update={"session_id": self.session_id})
                return cast(
                    "RequestPermissionResponse",
                    await self.requests.client.request_permission(forwarded),
                )

            self.agent.acp_permission_callback = permission_callback

        # Subscribe to state changes for THIS agent only
        # Defense: disconnect first (idempotent) to prevent duplicate connections
        with suppress(Exception):
            self.agent.state_updated.disconnect(self._on_state_updated)
        self.agent.state_updated.connect(self._on_state_updated)
        # Register global commands from manifest.commands (e.g., static commands like start_eval)
        self._register_manifest_commands()

        # Register pool-level skills as slash commands
        self._register_skill_commands()

        # Subscribe to dynamic skill changes from ExtensionRegistry
        self._start_skill_change_watcher()

        self.log.info("Created ACP session", current_agent=self.agent.name)

    @property
    def is_busy(self) -> bool:
        """Whether the session is currently processing a prompt.

        Returns:
            True if the task lock is held (active prompt processing).
        """
        return cast("bool", self._task_lock.locked())

    @property
    def host_context(self) -> HostContext:
        """Get the host context from the current agent."""
        ctx = self.agent.host_context
        if ctx is None:
            msg = "Agent has no associated pool"
            raise RuntimeError(msg)
        return ctx

    def get_cwd_context(self) -> str:
        """Get current working directory context for prompts."""
        return f"Working directory: {self.cwd}" if self.cwd else ""

    def _notify_command_update(self) -> None:
        """Notify all registered callbacks about command updates."""
        for callback in self._update_callbacks:
            try:
                callback()
            except Exception:
                logger.exception("Command update callback failed")

    def get_acp_commands(self) -> list[AvailableCommand]:
        """Convert all slashed commands to ACP format."""
        # Filter commands by node compatibility
        cmds = []
        for cmd in self.command_store.list_commands():
            # Check if command supports current node type
            if isinstance(cmd, NodeCommand) and not cmd.supports_node(self.agent):
                continue
            available_cmd = AvailableCommand.create(
                name=cmd.name,
                description=cmd.description,
                input_hint=cmd.usage,
            )
            cmds.append(available_cmd)
        return cmds

    def register_update_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for command updates."""
        self._update_callbacks.append(callback)
