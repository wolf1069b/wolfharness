"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import contextlib
import traceback
from typing import Annotated
import uuid

from pydantic_ai import TextPartDelta
import typer as t

from wolfharness.agents.events import PartDeltaEvent, StreamCompleteEvent
from wolfharness_cli import resolve_agent_config
from wolfharness_cli.cli_types import DetailLevel  # noqa: TC001
from wolfharness_cli.common import verbose_opt


def run_command(
    node_name: Annotated[str, t.Argument(help="Agent / Team name to run")],
    prompts: Annotated[list[str] | None, t.Argument(help="Additional prompts to send")] = None,
    config_path: Annotated[
        str | None, t.Option("-c", "--config", help="Override config path")
    ] = None,
    show_messages: Annotated[
        bool, t.Option("--show-messages", help="Show all messages (not just final responses)")
    ] = True,
    detail_level: Annotated[
        DetailLevel, t.Option("-d", "--detail", help="Output detail level")
    ] = "simple",
    show_metadata: Annotated[bool, t.Option("--metadata", help="Show message metadata")] = False,
    show_costs: Annotated[bool, t.Option("--costs", help="Show token usage and costs")] = False,
    verbose: bool = verbose_opt,
) -> None:
    """Single-shot run a node (agent/team) with prompts."""
    try:
        # Resolve configuration path
        try:
            config_path = resolve_agent_config(config_path)
        except ValueError as e:
            error_msg = str(e)
            raise t.BadParameter(error_msg) from e

        async def run() -> None:
            from wolfharness import AgentPool

            async with AgentPool(config_path) as pool:
                sp = pool.session_pool
                if sp is None:
                    msg = "SessionPool not available"
                    raise RuntimeError(msg)  # noqa: TRY301

                # Validate agent exists
                if node_name not in pool.agent_configs:
                    available = list(pool.agent_configs.keys())
                    msg = f"Agent '{node_name}' not found. Available agents: {', '.join(available)}"
                    raise t.BadParameter(msg)  # noqa: TRY301

                session_id = f"run-{node_name}-{uuid.uuid4().hex[:8]}"
                await sp.create_session(session_id, agent_name=node_name)

                try:
                    for prompt in prompts or []:
                        final_message = None
                        async for event in sp.run_stream(session_id, prompt, scope="session"):
                            if isinstance(event, StreamCompleteEvent):
                                final_message = event.message
                            elif (
                                isinstance(event, PartDeltaEvent)
                                and show_messages
                                and isinstance(event.delta, TextPartDelta)
                            ):
                                print(event.delta.content_delta, end="", flush=True)
                        if show_messages:
                            print()

                        if final_message and not show_messages:
                            print(
                                final_message.format(
                                    style=detail_level,
                                    show_metadata=show_metadata,
                                    show_costs=show_costs,
                                )
                            )
                finally:
                    with contextlib.suppress(Exception):
                        await sp.close_session(session_id)

        # Run the async code in the sync command
        asyncio.run(run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
