"""Command for running agents as a completions API server."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

import typer as t

from wolfharness_cli import resolve_agent_config
from wolfharness_cli.log import get_logger


logger = get_logger(__name__)


def api_command(
    ctx: t.Context,
    config: Annotated[str | None, t.Argument(help="Path to agent configuration")] = None,
    host: Annotated[str, t.Option(help="Host to bind server to")] = "localhost",
    port: Annotated[int, t.Option(help="Port to listen on")] = 8000,
    cors: Annotated[bool, t.Option(help="Enable CORS")] = True,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show message activity (deprecated, no-op)")
    ] = False,
    docs: Annotated[bool, t.Option(help="Enable API documentation")] = True,
) -> None:
    """Run agents as a completions API server.

    This creates an OpenAI-compatible API server that makes your agents available
    through a standard completions API interface.
    """
    import uvicorn

    from wolfharness import AgentPool, AgentsManifest
    from wolfharness_config.context import ConfigContextManager
    from wolfharness_server.openai_api_server.server import OpenAIAPIServer

    logger.info("Server PID", pid=os.getpid())

    try:
        config_path = resolve_agent_config(config)
    except ValueError as e:
        msg = str(e)
        raise t.BadParameter(msg) from e
    with ConfigContextManager(config_path):
        manifest = AgentsManifest.from_file(config_path)
        if config_path:

            def update_with_path(nodes: dict[str, Any]) -> dict[str, Any]:
                return {
                    name: node_config.model_copy(update={"config_file_path": config_path})
                    for name, node_config in nodes.items()
                }

            manifest = manifest.model_copy(
                update={
                    "config_file_path": config_path,
                    "agents": update_with_path(manifest.agents),
                    "teams": update_with_path(manifest.teams),
                }
            )

        # Keep AgentPool initialization inside the config context so custom
        # providers can resolve relative schema/prompt paths against the YAML directory.
        pool = AgentPool(manifest)

    # show_messages is disabled: agent instances are no longer created at pool level.
    # Session-level event monitoring is available via EventBus instead.

    # Get log level from the global context
    log_level = ctx.obj.get("log_level", "info") if ctx.obj else "info"

    async def run_server() -> None:
        async with pool:
            server = OpenAIAPIServer(pool, cors=cors, docs=docs)
            config = uvicorn.Config(server.app, host=host, port=port, log_level=log_level.lower())
            uv_server = uvicorn.Server(config)
            await uv_server.serve()

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("API server shutdown requested")
    except Exception as e:
        logger.exception("API server error")
        raise t.Exit(1) from e


if __name__ == "__main__":
    import typer

    typer.run(api_command)
