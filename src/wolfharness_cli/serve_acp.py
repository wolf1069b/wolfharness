"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.

Configuration is resolved from multiple layers (in precedence order):
1. Global config (~/.config/wolfharness/wolfharness.yml)
2. Custom config (AGENTPOOL_CONFIG env var)
3. Project config (wolfharness.yml in project/git root)
4. Explicit CLI argument (highest precedence)
5. Built-in fallback (only if no agents defined elsewhere)
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated, Any, Literal
import warnings

import anyenv
from platformdirs import user_log_path
import typer as t

from wolfharness_cli import log


if TYPE_CHECKING:
    from acp import Transport


logger = log.get_logger(__name__)


def acp_command(  # noqa: PLR0915
    # Too many statements - complex CLI command with many options
    config: Annotated[str | None, t.Argument(help="Path to agent configuration (optional)")] = None,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity in logs")
    ] = False,
    show_events: Annotated[
        bool,
        t.Option(
            "--show-events",
            help="Print agent stream events to stderr for debugging",
        ),
    ] = False,
    show_events_detailed: Annotated[
        bool,
        t.Option(
            "--show-events-detailed",
            help="Print detailed agent stream events to stderr (includes tool progress)",
        ),
    ] = False,
    debug_messages: Annotated[
        bool, t.Option("--debug-messages", help="Save raw JSON-RPC messages to debug file")
    ] = False,
    debug_file: Annotated[
        str | None,
        t.Option(
            "--debug-file",
            help="File to save JSON-RPC debug messages (default: acp-debug.jsonl)",
        ),
    ] = None,
    debug_commands: Annotated[
        bool,
        t.Option(
            "--debug-commands",
            help="Enable debug slash commands for testing ACP notifications",
        ),
    ] = False,
    agent: Annotated[
        str | None,
        t.Option(
            "--agent",
            help="Name of specific agent to use (defaults to pool's default agent)",
        ),
    ] = None,
    load_skills: Annotated[
        bool | None,
        t.Option(
            "--skills/--no-skills",
            help="Load client-side skills from .claude/skills directory. "
            "Defaults to the manifest's skills.include_default setting.",
        ),
    ] = None,
    transport: Annotated[
        Literal["stdio", "websocket", "streamable-http"],
        t.Option(
            "--transport",
            "-t",
            help="Transport type: stdio (default), streamable-http, or websocket (deprecated)",
        ),
    ] = "stdio",
    host: Annotated[
        str,
        t.Option(
            "--host",
            "-h",
            help="Host to bind the server to (only used with --transport streamable-http)",
        ),
    ] = "localhost",
    port: Annotated[
        int,
        t.Option(
            "-p",
            "--port",
            help="Port for the server to listen on (only used with --transport streamable-http)",
        ),
    ] = 8080,
    ws_host: Annotated[
        str,
        t.Option(
            "--ws-host",
            help="WebSocket host (only used with --transport websocket, deprecated)",
        ),
    ] = "localhost",
    ws_port: Annotated[
        int,
        t.Option(
            "--ws-port",
            help="WebSocket port (only used with --transport websocket, deprecated)",
        ),
    ] = 8765,
    ws_ping_interval: Annotated[
        float,
        t.Option(
            "--ws-ping-interval",
            help="Seconds between WebSocket ping frames for WebSocket transports",
        ),
    ] = 60.0,
    ws_pong_timeout: Annotated[
        float,
        t.Option(
            "--ws-pong-timeout",
            help="Seconds to wait for each WebSocket pong",
        ),
    ] = 30.0,
    ws_max_missed_pongs: Annotated[
        int,
        t.Option(
            "--ws-max-missed-pongs",
            help="Consecutive missed WebSocket pongs before disconnecting",
        ),
    ] = 3,
    mcp_config: Annotated[
        str | None,
        t.Option(
            "--mcp-config",
            help='MCP servers configuration as JSON (format: {"mcpServers": {...}})',
        ),
    ] = None,
    subagent_display_mode: Annotated[
        Literal["legacy", "zed", "qwen"] | None,
        t.Option(
            "--subagent-display-mode",
            help="Display subagent: 'legacy', 'zed', or 'qwen'",
        ),
    ] = None,
    raw_input_mode: Annotated[
        Literal["dict", "skip", "json_str"] | None,
        t.Option(
            "--raw-input-mode",
            help="Tool call raw_input mode: 'dict' (default), 'skip', or 'json_str'",
        ),
    ] = None,
) -> None:
    r"""Run agents as an ACP (Agent Client Protocol) server.

    This creates an ACP-compatible JSON-RPC 2.0 server that communicates over stdio
    streams or streamable HTTP/WebSocket, enabling your agents to work with desktop
    applications that support the Agent Client Protocol.

    Configuration Layers (merged in order, later overrides earlier):

    1. Global config - ~/.config/wolfharness/wolfharness.yml for user preferences
    2. Custom config - AGENTPOOL_CONFIG env var for CI/deployment overrides
    3. Project config - wolfharness.yml in project/git root for project-specific settings
    4. Explicit config - CLI argument (highest precedence)
    5. Built-in fallback - Only used if no agents defined in any layer

    Agent Selection:
    Use --agent to specify which agent to use by name. Without this option,
    the pool's default agent is used (set via 'default_agent' in config,
    or falls back to the first agent).

    Transport Options:
        # Stdio transport (default, for IDE integration)
        wolfharness serve-acp agents.yml

        # Streamable HTTP transport (recommended for Toad, web clients)
        wolfharness serve-acp agents.yml --transport streamable-http --port 8080

        # Legacy WebSocket transport (deprecated)
        wolfharness serve-acp agents.yml --transport websocket --ws-port 8765
    """
    from acp import ACPWebSocketTransport, StdioTransport, WebSocketTransport
    from wolfharness import log
    from wolfharness.config_resources import ACP_ASSISTANT
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_config.context import ConfigContextManager
    from wolfharness_config.resolution import resolve_config
    from wolfharness_server.acp_server import ACPServer

    # Build transport config
    if transport == "streamable-http":
        transport_config: Transport = ACPWebSocketTransport(
            host=host,
            port=port,
            ping_interval=ws_ping_interval,
            pong_timeout=ws_pong_timeout,
            max_missed_pongs=ws_max_missed_pongs,
        )
    elif transport == "websocket":
        warnings.warn(
            "--transport websocket is deprecated; use --transport streamable-http instead",
            DeprecationWarning,
            stacklevel=2,
        )
        transport_config = WebSocketTransport(
            host=ws_host,
            port=ws_port,
            ping_interval=ws_ping_interval,
            pong_timeout=ws_pong_timeout,
            max_missed_pongs=ws_max_missed_pongs,
        )
    elif transport == "stdio":
        transport_config = StdioTransport()

    # Always log to file with rollover
    log_dir = user_log_path("wolfharness", appauthor=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "acp.log"
    log.configure_logging(force=True, log_file=str(log_file))
    logger.info("Configured file logging with rollover", log_file=str(log_file))

    # Resolve configuration from all layers
    # fallback_config is only used if no agents are defined in any layer
    try:
        resolved = resolve_config(
            explicit_path=config,
            fallback_config=ACP_ASSISTANT,
        )
    except ValueError as e:
        raise t.BadParameter(str(e)) from e

    # Log which config layers were used
    if resolved.layers:
        sources = [f"{layer.source}:{layer.path}" for layer in resolved.layers if layer.path]
        logger.info("Config layers loaded", sources=sources, transport=transport)
    else:
        logger.info("Starting ACP server with built-in defaults only", transport=transport)

    # Load manifest from merged config data with config context for path resolution
    try:
        with ConfigContextManager(resolved.primary_path):
            manifest = AgentsManifest.model_validate(resolved.data)
            if resolved.primary_path:
                # Update manifest and all agent/team config_file_path for runtime
                # path resolution (per-session agent creation needs this)
                def update_with_path(nodes: dict[str, Any]) -> dict[str, Any]:
                    return {
                        name: config.model_copy(update={"config_file_path": resolved.primary_path})
                        for name, config in nodes.items()
                    }

                manifest = manifest.model_copy(
                    update={
                        "config_file_path": resolved.primary_path,
                        "agents": update_with_path(manifest.agents),
                        "teams": update_with_path(manifest.teams),
                    }
                )

            acp_server = ACPServer.from_config(
                manifest,
                debug_messages=debug_messages,
                debug_file=debug_file or "acp-debug.jsonl" if debug_messages else None,
                debug_commands=debug_commands,
                agent=agent,
                load_skills=load_skills,
                transport=transport_config,
                subagent_display_mode=subagent_display_mode,
                raw_input_mode=raw_input_mode,
                show_events=show_events,
                show_events_detailed=show_events_detailed,
            )
    except Exception as e:
        raise t.BadParameter(f"Invalid merged configuration: {e}") from e

    # Inject MCP servers from --mcp-config if provided
    # TODO: Consider adding to specific agent's MCP manager instead of pool-level
    # for better isolation (currently all agents in pool share these servers)
    if mcp_config:
        from wolfharness_config.mcp_server import parse_mcp_servers_json

        try:
            mcp_data = anyenv.load_json(mcp_config)
        except anyenv.JsonLoadError as e:
            raise t.BadParameter(f"Invalid JSON in --mcp-config: {e}") from e

        try:
            servers = parse_mcp_servers_json(mcp_data)
        except ValueError as e:
            raise t.BadParameter(str(e)) from e

        for server in servers:
            acp_server.pool.mcp.add_server_config(server)
            logger.info("Added MCP server from --mcp-config", server_name=server.name)

    if show_messages:
        logger.info("Message activity logging enabled")
    if debug_messages:
        debug_path = debug_file or "acp-debug.jsonl"
        logger.info("Raw JSON-RPC message debugging enabled", path=debug_path)
    if debug_commands:
        logger.info("Debug slash commands enabled")
    logger.info("Server PID", pid=os.getpid())

    async def run_acp_server() -> None:
        try:
            async with acp_server:
                await acp_server.start()
        except KeyboardInterrupt:
            logger.info("ACP server shutdown requested")
        except Exception as e:
            logger.exception("ACP server error")
            raise t.Exit(1) from e

    asyncio.run(run_acp_server())


if __name__ == "__main__":
    t.run(acp_command)
