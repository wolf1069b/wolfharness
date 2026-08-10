"""Provider for subagent/task tools with streaming support.

Business-layer event routing is intentionally minimal. All agent stream events
flow through the SessionPool, which publishes them to the EventBus.
The protocol layer (OpenCode, ACP, etc.) subscribes to the parent session with
``scope="descendants"`` and receives child session events automatically — no
manual forwarding from the business layer is required.
"""

from __future__ import annotations

import asyncio
import datetime
import re
from typing import Any, Literal

import logfire
from pydantic_ai import ModelRetry

from wolfharness.agents.context import AgentContext  # noqa: TC001
from wolfharness.agents.events import StreamCompleteEvent
from wolfharness.agents.exceptions import MAX_DELEGATION_DEPTH, DelegationDepthError
from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.log import get_logger
from wolfharness.tools.exceptions import ToolError


logger = get_logger(__name__)

# Set to hold references to background tasks, preventing GC while running
_background_tasks: set[asyncio.Task[Any]] = set()


def _serialize_content(content: Any) -> str:
    """Serialize subagent output content to a string."""
    if not content:
        return ""
    if isinstance(content, str):
        return content

    from pydantic import BaseModel

    if isinstance(content, BaseModel):
        return content.model_dump_json()
    return str(content)


def _generate_task_id(description: str) -> str:
    """Generate a unique, sortable task ID from timestamp and description.

    Args:
        description: Short task description to include in the ID

    Returns:
        Task ID in format: YYYYMMDD-HHMMSS-description
    """
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    # Sanitize description: lowercase, replace spaces/special chars with dashes
    slug = re.sub(r"[^a-z0-9]+", "-", description.lower()).strip("-")[:30]
    return f"{timestamp}-{slug}"


class SubagentTools(FunctionToolsetCapability):
    """Provider for task delegation tools with streaming progress."""

    def __init__(
        self,
        name: str = "subagent_tools",
    ) -> None:
        super().__init__(name=name)
        # create_tool already adds to _tools, no need to call add_tool
        self.create_tool(self.task, category="other")

    async def task(  # noqa: D417
        self,
        ctx: AgentContext,
        agent_or_team: str,
        prompt: str,
        description: str,
        async_mode: bool = False,
    ) -> dict[str, Any]:
        """Execute a task on an agent or team.

        Launch a task to be executed by a specialized agent or team.

        In synchronous mode (default), the task runs with streaming progress events
        and returns the result when complete.

        In async mode, the task starts in the background and returns immediately
        with a task ID. The output is written to /tasks/{task_id}/output.md in
        the internal filesystem after the run completes.

        Args:
            agent_or_team: The agent or team to execute the task
            prompt: The task instructions for the agent or team
            description: A short (3-5 words) description of the task
            async_mode: If True, run in background and return task ID immediately

        Returns:
            Structured output containing result and metadata
        """
        from wolfharness.common_types import SupportsRunStream

        _ = description  # Used for logging/tracking in future

        session_pool = self._require_session_pool(ctx)
        agent_cfg, team_cfg = self._resolve_node_config(ctx, agent_or_team)

        # Register agent config in runtime registry so that
        # get_or_create_session_agent() (called internally by
        # create_child_session) can find it without pool-level storage.
        if agent_cfg is not None:
            session_pool.sessions.runtime_registry.register(agent_or_team, agent_cfg)

        source_type, agent_type_str = self._determine_source_type(agent_cfg, team_cfg)

        logger.info(
            "Executing task",
            agent_or_team=agent_or_team,
            description=description,
            async_mode=async_mode,
        )

        # Compute current delegation depth
        current_depth = ctx.run_ctx.depth if ctx.run_ctx is not None else 0
        child_depth = current_depth + 1

        # Guard against excessive nesting before creating any resources
        if child_depth > MAX_DELEGATION_DEPTH:
            raise DelegationDepthError(child_depth)

        parent_session_id = getattr(ctx.node, "session_id", None) or (
            ctx.run_ctx.session_id if ctx.run_ctx else ""
        )

        node_model_id = self._extract_model_id(agent_cfg)
        input_provider = self._resolve_input_provider(ctx, agent_or_team)

        is_team_node = team_cfg is not None
        child_session_id = await ctx.create_child_session(
            agent_name=agent_or_team,
            agent_type=agent_type_str,
            parent_session_id=parent_session_id,
            spawn_mechanism="task",
            description=f"Run {agent_or_team} task",
            tool_call_id=ctx.tool_call_id,
            source_name=agent_or_team,
            source_type=source_type,
            depth=child_depth,
            model_id=node_model_id,
            input_provider=input_provider,
            skip_agent_registration=is_team_node,
        )

        # For teams, we still need to create the team node directly
        node: SupportsRunStream[Any] | None = None
        if is_team_node:
            assert team_cfg is not None
            node = await session_pool.create_team_from_config(agent_or_team, team_cfg)
            if not isinstance(node, SupportsRunStream):
                msg = f"Team {agent_or_team} does not support streaming"
                raise ToolError(msg)

        if async_mode:
            return await self._start_async_task(
                ctx,
                session_pool,
                node,
                child_session_id,
                agent_or_team,
                prompt,
                description,
                input_provider,
                is_team_node,
            )

        # Synchronous mode — block until completion and return final result
        final_content = await self._run_sync(
            session_pool,
            node,
            child_session_id,
            prompt,
            input_provider,
            is_team_node,
        )
        return {
            "output": final_content,
            "metadata": {"sessionId": child_session_id},
        }

    @staticmethod
    def _require_session_pool(ctx: AgentContext) -> Any:
        """Validate pool and session_pool are available, return session_pool."""
        if ctx.pool is None:
            msg = "Agent needs to be in a pool to execute tasks"
            raise ToolError(msg)
        session_pool = ctx.pool.session_pool
        if session_pool is None:
            msg = "SessionPool is required for subagent task execution"
            raise ToolError(msg)
        return session_pool

    @staticmethod
    def _resolve_node_config(ctx: AgentContext, agent_or_team: str) -> tuple[Any, Any]:
        """Resolve agent or team config from manifest, raising ModelRetry if not found."""
        agent_cfg = ctx.pool.manifest.agents.get(agent_or_team) if ctx.pool else None
        team_cfg = ctx.pool.manifest.teams.get(agent_or_team) if ctx.pool else None
        if agent_cfg is None and team_cfg is None:
            assert ctx.pool is not None
            available = list(ctx.pool.manifest.agents.keys()) + list(ctx.pool.manifest.teams.keys())
            msg = (
                f"No agent or team found with name: {agent_or_team}. "
                f"Available nodes: {', '.join(available)}"
            )
            raise ModelRetry(msg)
        return agent_cfg, team_cfg

    @staticmethod
    def _determine_source_type(
        agent_cfg: Any, team_cfg: Any
    ) -> tuple[Literal["team_parallel", "team_sequential", "agent"], str]:
        """Determine source_type and agent_type_str from configs."""
        if agent_cfg is not None:
            return "agent", agent_cfg.type
        assert team_cfg is not None
        agent_type_str = "team"
        source_type: Literal["team_parallel", "team_sequential", "agent"] = (
            "team_parallel" if team_cfg.mode == "parallel" else "team_sequential"
        )
        return source_type, agent_type_str

    @staticmethod
    def _extract_model_id(agent_cfg: Any) -> str | None:
        """Extract model_id from agent config if it's a NativeAgentConfig."""
        if agent_cfg is None:
            return None
        from wolfharness.models.agents import NativeAgentConfig

        if isinstance(agent_cfg, NativeAgentConfig):
            raw_model = agent_cfg.model
            return str(raw_model) if raw_model else None
        return None

    @staticmethod
    def _resolve_input_provider(ctx: AgentContext, agent_or_team: str) -> Any:
        """Resolve input_provider, returning None if unavailable."""
        try:
            return ctx.get_input_provider()
        except RuntimeError:
            logger.warning(
                "No input_provider available in parent context; "
                "subagent will not support elicitation",
                agent=agent_or_team,
            )
            return None

    @staticmethod
    async def _run_sync(
        session_pool: Any,
        node: Any,
        child_session_id: str,
        prompt: str,
        input_provider: Any,
        is_team_node: bool,
    ) -> str:
        """Run task synchronously and return final content."""
        from wolfharness.messaging.message_history import MessageHistory

        if is_team_node:
            result = await node.run(prompt, message_history=MessageHistory())
            return _serialize_content(result.content)

        final_content = ""
        async for event in session_pool.run_stream(
            child_session_id,
            prompt,
            input_provider=input_provider,
            message_history=MessageHistory(),
        ):
            if isinstance(event, StreamCompleteEvent):
                content = event.message.content
                final_content = _serialize_content(content)
        return final_content

    @staticmethod
    async def _start_async_task(
        ctx: AgentContext,
        session_pool: Any,
        node: Any,
        child_session_id: str,
        agent_or_team: str,
        prompt: str,
        description: str,
        input_provider: Any,
        is_team_node: bool,
    ) -> dict[str, Any]:
        """Start a background async task and return task metadata."""
        task_id = _generate_task_id(description)
        output_path = f"/tasks/{task_id}/output.md"
        fs = ctx.internal_fs
        fs.mkdirs(f"/tasks/{task_id}", exist_ok=True)

        async def _background_run() -> None:
            """Run task through SessionPool and write final result to filesystem."""
            final_content = await SubagentTools._run_sync(
                session_pool,
                node,
                child_session_id,
                prompt,
                input_provider,
                is_team_node,
            )
            fs.pipe(output_path, final_content.encode("utf-8"))
            logger.info(
                "Async task completed",
                task_id=task_id,
                agent=agent_or_team,
                output_path=output_path,
            )

        # Wrap with error handling
        async def _safe_background_run() -> None:
            try:
                with logfire.span("subagent.background_task", task_id=task_id):
                    await _background_run()
            except Exception:
                logger.exception("Async task failed", task_id=task_id, agent=agent_or_team)
                error_content = (
                    f"# Task Failed\n\nTask {task_id} ({agent_or_team}) failed with an error."
                )
                fs.pipe(output_path, error_content.encode("utf-8"))

        task = asyncio.create_task(_safe_background_run(), name=f"async_task_{task_id}")
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

        return {
            "output": (
                f"Task started in background.\n"
                f"Task ID: {task_id}\n"
                f"Output will be written to: {output_path}\n"
                f"Use the read tool to check the output file for results."
            ),
            "metadata": {
                "taskId": task_id,
                "sessionId": child_session_id,
                "outputFile": output_path,
            },
        }
