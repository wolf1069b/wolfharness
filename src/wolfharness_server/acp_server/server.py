"""ACP (Agent Client Protocol) server implementation for wolfharness.

This module provides the main server class for exposing AgentPool via
the Agent Client Protocol.
"""

from __future__ import annotations

import asyncio
import functools
import sys
from typing import TYPE_CHECKING, Any, Literal, Self

from acp import ACPWebSocketTransport, StdioTransport, serve
from wolfharness import AgentPool
from wolfharness.log import get_logger
from wolfharness.models.manifest import AgentsManifest
from wolfharness_config.context import ConfigContextManager
from wolfharness_config.pool_server import ACPPoolServerConfig
from wolfharness_server import BaseServer
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from collections.abc import Callable

    from upathtools import JoinablePathLike

    from acp import Transport
    from acp.agent.connection import AgentSideConnection
    from wolfharness.agents.base_agent import BaseAgent


logger = get_logger(__name__)

SubagentDisplayMode = Literal["legacy", "zed", "qwen"]
RawInputMode = Literal["dict", "skip", "json_str"]


def _coerce_subagent_display_mode(value: str) -> SubagentDisplayMode:
    """Normalize config strings to the ACP literal union."""
    if value == "legacy":
        return "legacy"
    if value in ("inline", "tool_box"):
        logger.warning(
            "Subagent display mode '%s' is deprecated, use 'legacy' instead",
            value,
        )
        return "legacy"
    if value == "zed":
        logger.info("Subagent display mode set to 'zed'")
        return "zed"
    if value == "qwen":
        logger.info("Subagent display mode set to 'qwen'")
        return "qwen"
    logger.warning("Unknown subagent display mode '%s', falling back to 'legacy'", value)
    return "legacy"


def _acp_event_observer(show_detailed: bool = False) -> Callable[[Any], None]:
    """Create an ACP stream observer that prints JSON-RPC messages to stderr.

    Args:
        show_detailed: Whether to print full message content or just summary.

    Returns:
        StreamObserver callable for Connection._observers.
    """

    def observer(event: Any) -> None:
        direction_icon = "→" if event.direction == "outgoing" else "←"
        method = event.message.get("method", "response")
        if show_detailed:
            import json

            msg_str = json.dumps(
                event.message, ensure_ascii=False, separators=(",", ":"), default=str
            )
            print(f"[ACP {direction_icon}] {method}: {msg_str}", flush=True, file=sys.stderr)
        else:
            print(f"[ACP {direction_icon}] {method}", flush=True, file=sys.stderr)

    return observer


class ACPServer(BaseServer):
    """ACP (Agent Client Protocol) server for wolfharness using external library.

    Provides a bridge between wolfharness's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.

    The actual client communication happens via the AgentSideConnection created
    when start() is called, which communicates with the external process over stdio.
    """

    def __init__(
        self,
        pool: AgentPool[Any],
        *,
        name: str | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
        debug_commands: bool = False,
        agent: str | None = None,
        load_skills: bool | None = None,
        config_path: str | None = None,
        transport: Transport = "stdio",
        subagent_display_mode: SubagentDisplayMode = "legacy",
        raw_input_mode: RawInputMode = "dict",
        show_events: bool = False,
        show_events_detailed: bool = False,
    ) -> None:
        """Initialize ACP server with configuration.

        Args:
            pool: AgentPool containing available agents
            name: Optional Server name (auto-generated if None)
            debug_messages: Whether to enable debug message logging
            debug_file: File path for debug message logging
            debug_commands: Whether to enable debug slash commands for testing
            agent: Optional specific agent name to use (defaults to first agent)
            load_skills: Whether to load client-side skills from .claude/skills.
                If None (default), uses the manifest's skills.include_default setting.
            config_path: Path to the configuration file (for tracking/hot-switching)
            transport: Transport configuration ("stdio", "websocket", or transport object)
            subagent_display_mode: How to display nested agent output in ACP clients
            raw_input_mode: How to emit tool call raw_input ("dict", "skip", or "json_str")
            show_events: Whether to print agent stream events to stderr
            show_events_detailed: Whether to print detailed agent stream events to stderr
        """
        super().__init__(pool, name=name, raise_exceptions=True)
        self.debug_messages = debug_messages
        self.debug_file = debug_file
        self.debug_commands = debug_commands
        self.agent = agent
        self.load_skills = load_skills
        self.config_path = config_path
        self.transport: Transport = transport
        self.subagent_display_mode: SubagentDisplayMode = subagent_display_mode
        self.raw_input_mode: RawInputMode = raw_input_mode
        self.show_events = show_events
        self.show_events_detailed = show_events_detailed

    @classmethod
    def from_config(
        cls,
        config: JoinablePathLike | AgentsManifest,
        *,
        debug_messages: bool = False,
        debug_file: str | None = None,
        debug_commands: bool = False,
        agent: str | None = None,
        load_skills: bool | None = None,
        transport: Transport = "stdio",
        subagent_display_mode: SubagentDisplayMode | None = None,
        raw_input_mode: RawInputMode | None = None,
        show_events: bool = False,
        show_events_detailed: bool = False,
    ) -> Self:
        """Create ACP server from configuration path or manifest.

        Args:
            config: Path to YAML config file or pre-loaded AgentsManifest
            debug_messages: Enable saving JSON messages to file
            debug_file: Path to debug file
            debug_commands: Enable debug slash commands for testing
            agent: Optional specific agent name to use (defaults to first agent)
            load_skills: Whether to load client-side skills from .claude/skills.
                If None (default), uses the manifest's skills.include_default setting.
            transport: Transport configuration ("stdio", "websocket", or transport object)
            subagent_display_mode: Override for subagent display mode (argument > config > default)
            raw_input_mode: Override for raw input mode (argument > config > default)
            show_events: Whether to print agent stream events to stderr
            show_events_detailed: Whether to print detailed agent stream events to stderr

        Returns:
            Configured ACP server instance with agent pool
        """
        # AgentPool handles both path and manifest
        pool = AgentPool(
            manifest=config,
            main_agent_name=agent,
        )

        # Determine config_path for tracking
        config_path = config.config_file_path if isinstance(config, AgentsManifest) else str(config)

        # Resolve subagent_display_mode with priority: argument > config > default
        resolved_display_mode: SubagentDisplayMode
        if subagent_display_mode is not None:
            resolved_display_mode = subagent_display_mode
        # Fall back to config value
        elif isinstance(config, AgentsManifest):
            config_mode: str = getattr(config.pool_server, "subagent_display_mode", "legacy")
            resolved_display_mode = _coerce_subagent_display_mode(config_mode)
        else:
            resolved_display_mode = "legacy"

        # Resolve raw_input_mode with priority: argument > config > default
        resolved_raw_input_mode: RawInputMode
        if raw_input_mode is not None:
            resolved_raw_input_mode = raw_input_mode
        elif isinstance(config, AgentsManifest):
            resolved_raw_input_mode = getattr(config.pool_server, "raw_input_mode", "dict")
        else:
            resolved_raw_input_mode = "dict"

        # Resolve transport with priority: argument > config > default
        resolved_transport: Transport
        if transport != "stdio":
            # Explicit transport argument overrides config
            resolved_transport = transport
        elif isinstance(config, AgentsManifest) and isinstance(
            config.pool_server, ACPPoolServerConfig
        ):
            if config.pool_server.transport == "streamable-http":
                resolved_transport = ACPWebSocketTransport(
                    host=config.pool_server.host, port=config.pool_server.port
                )
            else:
                resolved_transport = StdioTransport()
        else:
            resolved_transport = StdioTransport()

        # Resolve load_skills with priority: argument > manifest > default (True)
        resolved_load_skills: bool
        if load_skills is not None:
            # Explicit argument overrides manifest
            resolved_load_skills = load_skills
        elif pool.manifest.skills is not None:
            # Fall back to manifest's skills.include_default setting
            resolved_load_skills = pool.manifest.skills.include_default
        else:
            resolved_load_skills = True

        server = cls(
            pool,
            debug_messages=debug_messages,
            debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
            debug_commands=debug_commands,
            agent=agent,
            load_skills=resolved_load_skills,
            config_path=config_path,
            transport=resolved_transport,
            subagent_display_mode=resolved_display_mode,
            raw_input_mode=resolved_raw_input_mode,
            show_events=show_events,
            show_events_detailed=show_events_detailed,
        )
        agent_names = list(server.pool.manifest.agents.keys())

        # Validate specified agent exists if provided
        if agent and agent not in pool.manifest.agents:
            msg = f"Specified agent {agent!r} not found in config. Available agents: {agent_names}"
            raise ValueError(msg)

        server.log.info("Created ACP server", agent_names=agent_names, config_path=config_path)
        if agent:
            server.log.info("ACP session agent", agent=agent)
        return server

    async def _resolve_default_agent(self) -> BaseAgent[Any, Any]:
        """Resolve the default agent from name or get pool's default agent.

        Creates a bare agent instance with pool reference via
        ``config.get_agent(pool=pool)`` — this does NOT create a session
        in the SessionController. Previously this called
        ``get_or_create_session_agent("acp-default", ...)`` which created
        a phantom ``"acp-default"`` session that polluted logs and
        expired after 1 hour.

        Returns:
            The resolved agent instance

        Raises:
            RuntimeError: If no agents are available or SessionPool not available
            ValueError: If specified agent doesn't exist
        """
        session_pool = self.pool.session_pool
        if session_pool is None:
            msg = "SessionPool not available"
            raise RuntimeError(msg)

        agent_name = self.agent if self.agent else self.pool.main_agent_name
        if self.agent and self.agent not in self.pool.manifest.agents:
            raise ValueError(f"Agent {self.agent!r} not found in pool")

        # Create a bare agent with pool reference, WITHOUT creating a session.
        cfg = self.pool.manifest.agents[agent_name]
        agent: BaseAgent[Any, Any] = cfg.get_agent(pool=self.pool)
        await agent.__aenter__()
        return agent

    async def _start_async(self) -> None:
        """Start the ACP server (blocking async - runs until stopped)."""
        transport_name = (
            type(self.transport).__name__ if not isinstance(self.transport, str) else self.transport
        )
        self.log.info("Starting ACP server", transport=transport_name)
        # Resolve agent instance from name
        default_agent = await self._resolve_default_agent()
        self.log.info("Using default agent", agent=default_agent.name)
        # Create a shared session manager so WebSocket disconnects can clean up
        # all sessions for the dropped connection via close_all_sessions_for_connection().
        session_manager = ACPSessionManager(pool=self.pool)
        create_acp_agent = functools.partial(
            AgentPoolACPAgent,
            default_agent=default_agent,
            debug_commands=self.debug_commands,
            load_skills=self.load_skills,
            server=self,
            subagent_display_mode=self.subagent_display_mode,
            raw_input_mode=self.raw_input_mode,
            session_manager=session_manager,
        )

        async def on_disconnect(conn: AgentSideConnection) -> None:
            """Clean up sessions when a WebSocket client disconnects."""
            connection_id = conn.connection_id
            if connection_id is not None:
                await session_manager.close_all_sessions_for_connection(connection_id)

        debug_file = self.debug_file if self.debug_messages else None
        observers = None
        if self.show_events or self.show_events_detailed:
            observers = [_acp_event_observer(show_detailed=self.show_events_detailed)]
        self.log.info("ACP server started")
        try:
            await serve(
                create_acp_agent,
                transport=self.transport,
                shutdown_event=self._shutdown_event,
                debug_file=debug_file,
                on_disconnect=on_disconnect,
                observers=observers,
            )
        except asyncio.CancelledError:
            self.log.info("ACP server shutdown requested")
            raise
        except KeyboardInterrupt:
            self.log.info("ACP server shutdown requested")
        except Exception:
            self.log.exception("ACP server error")

    def stop(self) -> None:
        """Stop the ACP server.

        Sets the shutdown event before cancelling the server task
        so the serving coroutine can observe it.
        """
        self._shutdown_event.set()
        super().stop()

    async def swap_pool(
        self, config_path: str, agent_name: str | None = None
    ) -> BaseAgent[Any, Any]:
        """Swap the current pool with a new one from config.

        This method handles the full lifecycle of swapping pools:
        1. Validates the new configuration
        2. Creates and initializes the new pool
        3. Cleans up the old pool
        4. Updates internal references

        Args:
            config_path: Path to the new agent configuration file
            agent_name: Optional specific agent name to use as default

        Returns:
            The resolved default agent instance from the new pool

        Raises:
            ValueError: If config is invalid or specified agent not found
            FileNotFoundError: If config file doesn't exist
        """
        # 1. Parse and validate new config before touching current pool
        self.log.info("Loading new pool configuration", config_path=config_path)
        with ConfigContextManager(config_path):
            new_manifest = AgentsManifest.from_file(config_path)
            new_pool = AgentPool(
                manifest=new_manifest,
            )
        # 2. Validate agent exists in new pool if specified
        agent_names = list(new_pool.manifest.agents.keys())
        if not agent_names:
            msg = "New configuration contains no agents"
            raise ValueError(msg)
        if agent_name and agent_name not in agent_names:
            msg = f"Agent {agent_name!r} not found in new config. Available: {agent_names}"
            raise ValueError(msg)
        # 3. Enter new pool context first (so we can roll back if it fails)
        try:
            await new_pool.__aenter__()
        except Exception as e:
            self.log.exception("Failed to initialize new pool")
            msg = f"Failed to initialize new pool: {e}"
            raise ValueError(msg) from e
        # 4. Exit old pool context
        old_pool = self.pool
        try:
            await old_pool.__aexit__(None, None, None)
        except Exception:
            self.log.exception("Error closing old pool (continuing with swap)")
        # 5. Update references
        self.pool = new_pool
        self.agent = agent_name
        self.config_path = config_path
        # 6. Resolve and return the default agent instance
        default_agent = await self._resolve_default_agent()
        self.log.info(
            "Pool swapped successfully", agent_names=agent_names, default_agent=default_agent.name
        )
        return default_agent
