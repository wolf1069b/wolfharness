"""Lifecycle mixin for ACPSession.

Extracted from session.py as part of the session-debt-cleanup file split.
Contains session initialization, MCP server setup, prompt processing,
cancellation, and close methods.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import anyio
from pydantic_ai import UsageLimitExceeded

from wolfharness import Agent
from wolfharness.log import get_logger
from wolfharness_server.acp_server.converters import from_acp_content
from wolfharness_server.acp_server.event_converter import ACPEventConverter


if TYPE_CHECKING:
    from collections.abc import Sequence

    from acp.schema import ContentBlock, McpServer, StopReason
    from wolfharness.mcp_server.config_snapshot import McpConfigEntry


logger = get_logger(__name__)


class ACPSessionLifecycleMixin:
    """Mixin providing session lifecycle methods for ACPSession.

    Contains initialization, MCP server setup, prompt processing,
    cancellation, and close methods.

    All attributes are provided by the main :class:`ACPSession` dataclass.
    Type annotations are declared under ``TYPE_CHECKING`` to avoid being
    treated as dataclass fields.
    """

    if TYPE_CHECKING:
        session_id: str
        cwd: str
        agent: Any  # BaseAgent[Any, Any]
        acp_agent: Any  # AgentPoolACPAgent
        mcp_servers: Any  # Sequence[McpServer] | None
        client_capabilities: Any  # ClientCapabilities
        command_store: Any  # CommandStore
        subagent_display_mode: Any  # Literal["legacy", "zed", "qwen"]
        raw_input_mode: Any  # Literal["dict", "skip", "json_str"]
        _task_lock: Any  # asyncio.Lock
        _cancelled: bool
        _current_converter: ACPEventConverter | None
        _skill_change_task: Any  # asyncio.Task[None] | None
        acp_env: Any  # ACPExecutionEnvironment
        input_provider: Any  # ACPInputProvider
        notifications: Any  # ACPNotifications
        requests: Any  # ACPRequests
        log: Any
        manager: Any  # ACPSessionManager | None

        @property
        def host_context(self) -> Any: ...
        def get_cwd_context(self) -> str: ...
        async def _register_mcp_prompts_as_commands(self) -> None: ...
        async def execute_slash_command(self, command_text: str) -> None: ...
        async def _on_state_updated(self, state: Any) -> None: ...

    async def initialize(self) -> None:
        """Initialize async resources. Must be called after construction."""
        # Prevent _detect_os_type() from sending terminal/create requests
        # when the client does not support terminal capability.
        # _detect_os_type() runs uname -s / ver via terminal, which fails
        # for clients declaring terminal=false in their capabilities.
        if not self.client_capabilities.terminal:
            import platform

            self.acp_env._os_type = platform.system()
        await self.acp_env.__aenter__()

    def _make_provider_name(self, display_name: str) -> str:
        """Build a provider name that fits within the 63-char DNS-label limit.

        Truncates the session_id (via SHA-256 prefix) when the full name
        would exceed ``MAX_PROVIDER_NAME_LENGTH``.

        Args:
            display_name: The MCP server display name to embed.

        Returns:
            A provider name guaranteed to pass ``_validate_provider_name``.
        """
        import hashlib

        from wolfharness_server.acp_server.session import _MAX_PROVIDER_NAME_LENGTH

        prefix = "session_"
        suffix = f"_{display_name}"
        budget = _MAX_PROVIDER_NAME_LENGTH - len(prefix) - len(suffix)
        if budget >= len(self.session_id):
            return f"{prefix}{self.session_id}{suffix}"
        # Truncate session_id to fit — use SHA-256 prefix for collision resistance
        safe_budget = max(0, budget)
        truncated = hashlib.sha256(self.session_id.encode()).hexdigest()[:safe_budget]
        return f"{prefix}{truncated}{suffix}"

    async def initialize_mcp_servers(self) -> None:
        """Initialize MCP servers if any are configured.

        Session-level MCP servers are converted to :class:`McpConfigEntry`
        objects and merged into the session's MCP config snapshot via
        :meth:`MCPManager.update_session_snapshot`.  For ACP-transport
        servers, the transport is registered via
        :meth:`MCPManager.add_acp_transport` so that snapshot-aware
        capability building can reuse it.
        """
        from acp.schema.mcp import AcpMcpServer
        from wolfharness import Agent
        from wolfharness.mcp_server.config_snapshot import (
            McpConfigEntry,
            McpConfigSnapshot,
        )
        from wolfharness_server.acp_server.converters import (
            convert_acp_mcp_server_to_config,
        )

        if not self.mcp_servers:
            return
        self.log.info("Initializing MCP servers", server_count=len(self.mcp_servers))

        entries: list[McpConfigEntry] = []

        async def _init_server(server: McpServer) -> None:
            try:
                with anyio.fail_after(30):
                    cfg = convert_acp_mcp_server_to_config(server)

                    # ACP-transport MCP servers need a live connection to the
                    # client before the transport can be created.
                    if isinstance(server, AcpMcpServer):
                        self.log.info(
                            "Connecting ACP MCP server via mcp/connect",
                            server_name=server.name,
                        )
                        connection_id, session_key = await self.acp_agent.connect_acp_mcp_server(
                            server, self.session_id
                        )
                        conn = self.acp_agent._mcp_manager.get_connection(connection_id)
                        if conn is None:
                            raise RuntimeError(  # noqa: TRY301
                                f"AcpMcpConnection not found for {connection_id}"
                            )
                        from wolfharness_server.acp_server.acp_mcp_transport import (
                            AcpMcpTransport,
                        )

                        transport = AcpMcpTransport(conn, timeout=600.0)
                        if isinstance(self.agent, Agent):
                            # Register the ACP transport on the MCPManager's
                            # session context so that get_capabilities() can
                            # find it and child sessions can inherit it via
                            # copy_pre_created_transports().
                            await self.agent.mcp.add_acp_transport(
                                self.session_id,
                                cfg.client_id,
                                transport,
                                connection_id,
                                session_key,
                            )
                        self.log.info(
                            "Added session ACP MCP server",
                            server_name=cfg.name,
                            session_id=self.session_id,
                        )
                    else:
                        self.log.info(
                            "Added session MCP server",
                            server_name=cfg.name,
                            session_id=self.session_id,
                        )

                    entries.append(McpConfigEntry(server_config=cfg, source="session"))
            except TimeoutError:
                self.log.warning(
                    "MCP server initialization timed out",
                    server_name=server.name,
                )
            except Exception:
                self.log.exception(
                    "Failed to setup MCP server",
                    server_name=server.name,
                )

        await asyncio.gather(*[_init_server(s) for s in self.mcp_servers])

        # Merge new session configs into the agent's MCP snapshot, deduplicating
        # by client_id so that re-initialisation does not duplicate entries.
        if entries and isinstance(self.agent, Agent):
            ctx = self.agent.mcp.get_session_context(self.session_id)
            existing = ctx.snapshot if ctx is not None else None
            existing_session = existing.session_configs if existing is not None else ()
            seen_ids: set[str] = {e.server_config.client_id for e in existing_session}
            merged: list[McpConfigEntry] = list(existing_session)
            for entry in entries:
                if entry.server_config.client_id not in seen_ids:
                    merged.append(entry)
                    seen_ids.add(entry.server_config.client_id)
            new_snapshot = (existing or McpConfigSnapshot()).with_session_configs(tuple(merged))
            # Sync the updated snapshot to the MCPManager's session context
            # so that get_capabilities(session_id) can discover ACP MCP configs
            # and child sessions can inherit them via copy_pre_created_transports().
            self.agent.mcp.update_session_snapshot(self.session_id, new_snapshot)
            self.log.info(
                "Updated agent MCP snapshot with session configs",
                session_config_count=len(merged),
                session_id=self.session_id,
            )

        # Register MCP prompts as commands after all servers are added
        try:
            await self._register_mcp_prompts_as_commands()
        except Exception:
            self.log.exception("Failed to register MCP prompts as commands")

    async def cancel(self) -> None:
        """Cancel the current prompt turn.

        This actively interrupts the running agent by calling its interrupt() method,
        which handles protocol-specific cancellation (e.g., sending CancelNotification
        for ACP agents, etc.).

        Note:
            Tool call cleanup is handled in process_prompt() to avoid race conditions
            with the converter state being modified from multiple async contexts.
        """
        self._cancelled = True
        self.log.info("Session cancelled, interrupting agent")
        try:  # Actively interrupt the agent's stream
            await self.agent.interrupt()
        except Exception:
            self.log.exception("Failed to interrupt agent")

    def is_cancelled(self) -> bool:
        """Check if the session is cancelled."""
        return self._cancelled

    async def process_prompt(self, content_blocks: Sequence[ContentBlock]) -> StopReason:  # noqa: PLR0911, PLR0915
        """Process a prompt request and stream responses.

        Args:
            content_blocks: List of content blocks from the prompt request

        Returns:
            Stop reason
        """
        from wolfharness_server.acp_server.session import infer_stop_reason, split_commands

        self._cancelled = False
        fs = self.agent.env.get_fs()
        contents = [from_acp_content(i, fs=fs) for i in content_blocks]
        self.log.debug("Converted content", content=contents)
        if not contents:
            self.log.warning("Empty prompt received")
            return "refusal"
        commands, non_command_content = split_commands(contents, self.command_store)
        async with self._task_lock:
            if commands:  # Process commands if found
                for command in commands:
                    self.log.info("Processing slash command", command=command)
                    await self.execute_slash_command(command)

                # If only commands and no staged content, end turn
                if not non_command_content and len(self.agent.staged_content) == 0:
                    return "end_turn"

            self.log.debug("Processing prompt", content_items=len(non_command_content))
            event_count = 0
            # Derive turn-complete support from client capabilities
            client_supports_turn_complete = (
                bool(self.client_capabilities.turn_complete)
                if self.client_capabilities is not None
                else False
            )
            # Create a new event converter for this prompt
            converter = ACPEventConverter(
                subagent_display_mode=self.subagent_display_mode,
                raw_input_mode=self.raw_input_mode,
                client_supports_turn_complete=client_supports_turn_complete,
            )
            self._current_converter = converter  # Track for cancellation

            # Route through SessionPool for unified session management.
            # MCP tools are handled via McpConfigSnapshot → get_capabilities() →
            # MCPToolset, not through agent.tools.providers.
            agent_ctx = self.agent.host_context
            session_pool = agent_ctx.session_pool if agent_ctx is not None else None
            try:
                if session_pool is not None:
                    stream = session_pool.run_stream(
                        self.session_id,
                        *non_command_content,
                        input_provider=self.input_provider,
                        deps=self,
                    )
                else:
                    raise RuntimeError(  # noqa: TRY301
                        f"SessionPool is required for prompt processing "
                        f"in session {self.session_id}"
                    )

                async for event in stream:
                    if self._cancelled:
                        self.log.info("Cancelled during event loop, cleaning up tool calls")
                        # Send cancellation notifications for any pending tool calls
                        # This happens in the same async context as the converter
                        async for cancel_update in converter.cancel_pending_tools():
                            await self.notifications.send_update(cancel_update)
                        # CRITICAL: Allow time for client to process tool completion notifications
                        # before sending PromptResponse. Without this delay, the client may receive
                        # and process the PromptResponse before the tool notifications, causing UI
                        # state desync where subsequent prompts appear stuck/unresponsive.
                        # This is needed because even though send() awaits the write, the client
                        # may process messages asynchronously or out of order.
                        await anyio.sleep(0.05)
                        self._current_converter = None
                        return "cancelled"

                    event_count += 1
                    async for update in converter.convert(event):
                        await self.notifications.send_update(update)
                    # Yield control to allow notifications to be sent immediately
                    await anyio.sleep(0.01)
                self.log.info("Streaming finished", events_processed=event_count)
            except asyncio.CancelledError:
                # Task was cancelled (e.g., via interrupt()) - return proper stop reason
                # This is critical: CancelledError doesn't inherit from Exception,
                # so we must catch it explicitly to send the PromptResponse
                self.log.info("Stream cancelled via CancelledError, cleaning up tool calls")
                # Send cancellation notifications for any pending tool calls
                async for cancel_update in converter.cancel_pending_tools():
                    await self.notifications.send_update(cancel_update)
                # CRITICAL: Allow time for client to process tool completion notifications
                # before sending PromptResponse. See comment in cancellation branch above.
                await anyio.sleep(0.05)
                self._current_converter = None
                return "cancelled"
            except UsageLimitExceeded as e:
                self.log.info("Usage limit exceeded", error=str(e))
                return infer_stop_reason(str(e))
            except Exception as e:
                self._current_converter = None  # Clear converter reference
                self.log.exception("Error during streaming")
                # Send error as toast notification instead of polluting chat history
                await self._send_toast(
                    message=f"Agent error: {e}",
                    level="error",
                )
                await anyio.sleep(0.05)  # Allow network buffers to flush
                return "end_turn"
            else:
                # Title generation is now handled automatically by log_session
                self.last_usage = converter.last_usage
                self._current_converter = None  # Clear converter reference
                return "end_turn"

    async def _send_toast(
        self,
        message: str,
        level: str = "error",
        *,
        duration: int | None = None,
        action: dict[str, str] | None = None,
    ) -> None:
        """Send a toast notification via ExtNotification.

        Uses _wolfharness/toast ext notification instead of polluting chat
        history with error messages disguised as agent text.

        Args:
            message: Toast message text.
            level: Severity level (error, warning, info, success).
            duration: Display duration in ms; None for persistent.
            action: Optional action button {label, command}.
        """
        if self._cancelled:
            return
        try:
            await self.notifications.send_ext_notification(
                method="_wolfharness/toast",
                params={
                    "message": message,
                    "level": level,
                    "duration": duration,
                    "action": action,
                },
            )
        except Exception:
            self.log.exception("Failed to send toast notification")

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        try:
            # Cancel skill change watcher
            if self._skill_change_task is not None:
                self._skill_change_task.cancel()
                try:
                    await self._skill_change_task
                except asyncio.CancelledError:
                    if not self._skill_change_task.cancelled():
                        raise
                except Exception:
                    self.log.exception("Error awaiting cancelled skill change task")
                self._skill_change_task = None

            # Cleanup MCP session-scoped resources (toolset cache, connection
            # pool, ACP connection manager) before tearing down the agent env.
            # Must run BEFORE acp_env.__aexit__ so the agent context is still live.
            try:
                await self.agent.mcp.cleanup_session(self.session_id)
            except Exception:
                self.log.exception("Failed to cleanup MCP session", session_id=self.session_id)

            await self.acp_env.__aexit__(None, None, None)

            # Disconnect state_updated signal to prevent stale callbacks
            with suppress(Exception):
                self.agent.state_updated.disconnect(self._on_state_updated)

            # Clean up sys_prompts from THIS session's agent only
            if isinstance(self.agent, Agent) and (
                self.get_cwd_context in self.agent.sys_prompts.prompts
            ):
                self.agent.sys_prompts.prompts.remove(self.get_cwd_context)  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]

            # Note: Individual agents are managed by the pool's lifecycle
            # The pool will handle agent cleanup when it's closed
            self.log.info("Closed ACP session")
        except Exception:
            self.log.exception("Error closing session")
