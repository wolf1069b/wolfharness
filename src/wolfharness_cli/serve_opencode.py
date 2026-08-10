"""Command for running agents as an OpenCode-compatible server.

This creates an HTTP server that implements the OpenCode API protocol,
allowing OpenCode TUI and SDK clients to interact with AgentPool agents.

Configuration is resolved from multiple layers (in precedence order):
1. Global config (~/.config/wolfharness/wolfharness.yml)
2. Custom config (AGENTPOOL_CONFIG env var)
3. Project config (wolfharness.yml in project/git root)
4. Explicit CLI argument (highest precedence)
5. Built-in fallback (only if no agents defined elsewhere)
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from platformdirs import user_log_path
import typer as t

from wolfharness_cli import log
from wolfharness_cli.cli_types import LogLevel  # noqa: TC001


logger = log.get_logger(__name__)


def _apply_config_file_path(manifest: Any, primary_path: str) -> Any:
    """Set config_file_path on manifest and each agent/team for relative path resolution."""

    def update_with_path(nodes: dict[str, Any]) -> dict[str, Any]:
        return {
            name: cfg.model_copy(update={"config_file_path": primary_path})
            for name, cfg in nodes.items()
        }

    return manifest.model_copy(
        update={
            "config_file_path": primary_path,
            "agents": update_with_path(manifest.agents),
            "teams": update_with_path(manifest.teams),
        }
    )


def _configure_observability_and_logging(
    manifest: Any, host: str, port: int, resolved: Any, log_level: str = "info"
) -> None:
    """Initialize observability and configure file logging."""
    from wolfharness import log as ap_log
    from wolfharness.observability import registry

    registry.configure_observability(manifest.observability)

    log_dir = user_log_path("wolfharness", appauthor=False)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "opencode.log"
    ap_log.configure_logging(level=log_level.upper(), force=True, log_file=str(log_file))
    logger.info("Configured file logging with rollover", log_file=str(log_file))

    # Log which config layers were used
    if resolved.layers:
        sources = [f"{layer.source}:{layer.path}" for layer in resolved.layers if layer.path]
        logger.info("Config layers loaded", sources=sources, host=host, port=port)
    else:
        logger.info("Starting OpenCode server with built-in defaults only", host=host, port=port)


def opencode_command(
    config: Annotated[str | None, t.Argument(help="Path to agent configuration (optional)")] = None,
    host: Annotated[
        str,
        t.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        t.Option("--port", "-p", help="Port to listen on"),
    ] = 4096,
    agent: Annotated[
        str | None,
        t.Option(
            "--agent",
            help="Name of specific agent to use (defaults to pool's default agent)",
        ),
    ] = None,
    working_dir: Annotated[
        str | None,
        t.Option(
            "--working-dir",
            "-w",
            help="Working directory for file operations (defaults to current directory)",
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        t.Option("--log-level", "-l", help="Log level"),
    ] = "info",
) -> None:
    """Run agents as an OpenCode-compatible HTTP server.

    This creates an HTTP server implementing the OpenCode API protocol,
    enabling your AgentPool agents to work with OpenCode TUI and SDK clients.

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
    """
    from wolfharness import AgentPool
    from wolfharness.config_resources import ACP_ASSISTANT
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_config.context import ConfigContextManager
    from wolfharness_config.resolution import resolve_config
    from wolfharness_server.opencode_server.server import OpenCodeServer

    # Resolve configuration from all layers FIRST (before logging)
    # fallback_config is only used if no agents defined elsewhere
    try:
        resolved = resolve_config(
            explicit_path=config,
            fallback_config=ACP_ASSISTANT,
        )
    except ValueError as e:
        raise t.BadParameter(str(e)) from e

    # Load manifest from merged config data with config context for path resolution
    # Config context must be maintained for AgentPool initialization (for relative path resolution)
    try:
        with ConfigContextManager(resolved.primary_path):
            manifest = AgentsManifest.model_validate(resolved.data)
            if resolved.primary_path:
                manifest = _apply_config_file_path(manifest, resolved.primary_path)

            _configure_observability_and_logging(manifest, host, port, resolved, log_level)

            # Load agent from merged manifest (needs config context for path resolution)
            pool = AgentPool(manifest, main_agent_name=agent)
    except Exception as e:
        raise t.BadParameter(f"Invalid merged configuration: {e}") from e

    async def run_server() -> None:
        async with pool:
            # Get main agent instance via SessionPool for OpenCode server
            assert pool.session_pool is not None
            agent = await pool.session_pool.sessions.get_or_create_session_agent(
                session_id="__opencode_bootstrap__",
                agent_name=pool.main_agent_name,
            )

            # Load agent rules from global and project locations
            await agent.load_rules(working_dir)

            server = OpenCodeServer(
                agent,
                host=host,
                port=port,
                working_dir=working_dir,
            )
            logger.info("Server starting", url=f"http://{host}:{port}")
            await server.run_async()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("OpenCode server shutdown requested")
    except Exception as e:
        logger.exception("OpenCode server error")
        raise t.Exit(1) from e


if __name__ == "__main__":
    t.run(opencode_command)
