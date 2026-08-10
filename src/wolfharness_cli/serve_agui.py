"""Command for running agents as an AG-UI server."""

from __future__ import annotations

import os
from typing import Annotated

import typer as t

from wolfharness_cli import resolve_agent_config
from wolfharness_cli.log import get_logger


logger = get_logger(__name__)


def agui_command(
    ctx: t.Context,
    config: Annotated[str | None, t.Argument(help="Path to agent configuration")] = None,
    host: Annotated[str, t.Option(help="Host to bind server to")] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on")] = 8002,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity (deprecated, no-op)")
    ] = False,
) -> None:
    """Run agents as an AG-UI server.

    This creates an AG-UI protocol server that makes your agents available
    through the AG-UI interface, compatible with AG-UI clients like Toad.

    Each agent is accessible at /{agent_name} route.
    """
    import anyio

    from wolfharness import AgentPool, AgentsManifest
    from wolfharness_config.context import ConfigContextManager
    from wolfharness_server.agui_server import AGUIServer

    logger.info("Server PID", pid=os.getpid())

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e

    with ConfigContextManager(config_path):
        manifest = AgentsManifest.from_file(config_path)

    async def run_server() -> None:
        async with AgentPool(manifest) as pool:
            # show_messages is disabled: agent instances are no longer created at pool level.
            # Session-level event monitoring is available via EventBus instead.

            server = AGUIServer(pool, host=host, port=port)
            async with server:
                logger.info(
                    "AG-UI server started",
                    host=host,
                    port=port,
                    agents=list(pool.manifest.agents.keys()),
                )
                # List agent routes
                for name, url in server.list_agent_routes().items():
                    logger.info("Agent route", agent=name, url=url)

                async with server.run_context():
                    # Keep running until interrupted
                    try:
                        while True:
                            await anyio.sleep(1)
                    except KeyboardInterrupt:
                        logger.info("Shutting down AG-UI server")

    anyio.run(run_server)


if __name__ == "__main__":
    import typer

    typer.run(agui_command)
