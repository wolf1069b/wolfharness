"""Task execution command."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Annotated, cast
import uuid

import typer as t

from wolfharness.agents.events import StreamCompleteEvent
from wolfharness_cli import log, resolve_agent_config


if TYPE_CHECKING:
    from wolfharness import AgentsManifest


TASK_HELP = """
Execute a defined task with the specified agent.

Example:
    wolfharness task docs write_api_docs --prompt "Include code examples"
    wolfharness task docs write_api_docs --log-level DEBUG
"""


async def execute_job(
    agent_name: str,
    task_name: str,
    config: AgentsManifest,
    *,
    prompt: str | None = None,
) -> str:
    """Execute task with agent."""
    from wolfharness import AgentPool

    async with AgentPool(config) as pool:
        sp = pool.session_pool
        if sp is None:
            msg = "SessionPool not available"
            raise RuntimeError(msg)

        # Validate agent exists
        if agent_name not in pool.agent_configs:
            available = list(pool.agent_configs.keys())
            msg = f"Agent '{agent_name}' not found in config. Available: {', '.join(available)}"
            raise ValueError(msg)

        # Get task config (still available via TaskRegistry)
        task = pool.get_job(task_name)

        # Create final prompt from task and additional input
        task_prompt = await task.get_prompt()
        if prompt:
            task_prompt = f"{task_prompt}\n\nAdditional instructions:\n{prompt}"

        # Run through SessionPool
        session_id = f"task-{agent_name}-{uuid.uuid4().hex[:8]}"
        await sp.create_session(session_id, agent_name=agent_name)

        try:
            final_message = None
            async for event in sp.run_stream(session_id, task_prompt, scope="session"):
                if isinstance(event, StreamCompleteEvent):
                    final_message = event.message

            if final_message is None:
                msg = "No response received from agent"
                raise RuntimeError(msg)
            return cast(str, final_message.data)
        finally:
            with contextlib.suppress(Exception):
                await sp.close_session(session_id)


def task_command(
    agent_name: Annotated[str, t.Argument(help="Name of agent to run task with")],
    task_name: Annotated[str, t.Argument(help="Name of task to execute")],
    config: Annotated[
        str | None, t.Option("--config", "-c", help="Agent configuration file")
    ] = None,
    prompt: Annotated[str | None, t.Option("--prompt", "-p", help="Additional prompt")] = None,
) -> None:
    """Execute a task with the specified agent."""
    from wolfharness import AgentsManifest
    from wolfharness_config.context import ConfigContextManager

    logger = log.get_logger(__name__)
    try:
        logger.debug("Starting task execution", name=task_name)
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        with ConfigContextManager(config_path):
            manifest = AgentsManifest.from_file(config_path)
        result = asyncio.run(execute_job(agent_name, task_name, manifest, prompt=prompt))
        print(result)

    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        logger.debug("Exception details", exc_info=True)
        raise t.Exit(1) from e


if __name__ == "__main__":
    t.run(task_command)
