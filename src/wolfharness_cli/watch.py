"""Command for watching agents and displaying messages."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer as t

from wolfharness_cli import log


logger = log.get_logger(__name__)


def watch_command(
    config: Annotated[str, t.Argument(help="Path to agent configuration")],
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show all messages (deprecated, no-op)")
    ] = True,
    detail_level: Annotated[
        str | None, t.Option("-d", "--detail", help="Output detail level (deprecated, no-op)")
    ] = None,
    show_metadata: Annotated[
        bool, t.Option("--metadata", help="Show message metadata (deprecated, no-op)")
    ] = False,
    show_costs: Annotated[
        bool, t.Option("--costs", help="Show token usage and costs (deprecated, no-op)")
    ] = False,
) -> None:
    """Run agents in event-watching mode."""

    async def run_watch() -> None:
        from wolfharness import AgentPool, AgentsManifest
        from wolfharness_config.context import ConfigContextManager

        with ConfigContextManager(config):
            manifest = AgentsManifest.from_file(config)
        async with AgentPool(manifest) as pool:
            # show_messages is disabled: agent instances are no longer created at pool level.
            # Session-level event monitoring is available via EventBus instead.

            await pool.run_event_loop()

    asyncio.run(run_watch())
