"""Base class for teams — concrete team implementation with mode parameter.

This replaces the previous abstract ``BaseTeam`` / concrete ``Team`` /
``TeamRun`` three-class hierarchy with a single concrete ``BaseTeam`` that
dispatches execution based on ``mode`` ("parallel" or "sequential").
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import uuid4
from xml.sax.saxutils import escape

from anyenv.async_run import as_generated
import anyio
from jinja2 import BaseLoader, Environment
import logfire

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.events import SpawnSessionStart, SubAgentEvent
from wolfharness.agents.exceptions import MAX_DELEGATION_DEPTH, DelegationDepthError
from wolfharness.log import get_logger
from wolfharness.messaging import AgentResponse, ChatMessage, MessageNode, TeamResponse
from wolfharness.messaging.context import NodeContext
from wolfharness.messaging.messagenode import get_source_type
from wolfharness.messaging.processing import finalize_message, prepare_prompts
from wolfharness.talk.stats import AggregatedMessageStats
from wolfharness.talk.talk import Talk
from wolfharness.utils.time_utils import get_now


logger = get_logger(__name__)
_PROMPT_TEMPLATE_ENV = Environment(loader=BaseLoader(), autoescape=False)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from evented_config import EventConfig
    from toprompt import AnyPromptType

    from wolfharness import Agent, AgentPool
    from wolfharness.agents.events import RichAgentStreamEvent
    from wolfharness.common_types import (
        ProcessorCallback,
        PromptCompatible,
    )
    from wolfharness.delegation.graph_team import ExtendedTeamTalk
    from wolfharness.talk.stats import AggregatedTalkStats
    from wolfharness.ui.base import InputProvider
    from wolfharness_config.mcp_server import MCPServerConfig
    from wolfharness_config.teams import ExecutionMode, TeamMemberConfig


@dataclass(kw_only=True)
class TeamContext[TDeps = object](NodeContext[TDeps]):
    """Context for team nodes."""

    pool: AgentPool | None = None
    """Pool the team is part of."""


async def _timeout_stream[T](
    stream: AsyncIterator[T],
    timeout: float | None,
    member_name: str,
    team_name: str,
) -> AsyncIterator[T]:
    """Wrap an async iterator with an overall deadline.

    Each ``__anext__`` is bounded by the remaining budget so a single hung
    call cannot block forever.  If ``timeout`` is ``None`` the stream is
    yielded through unchanged.
    """
    if timeout is None:
        async for item in stream:
            yield item
        return

    deadline = perf_counter() + timeout
    it = stream.__aiter__()
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            logger.warning(
                "Team member stream timed out",
                member=member_name,
                team=team_name,
                timeout=timeout,
            )
            return
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            return
        except TimeoutError:
            logger.warning(
                "Team member stream timed out",
                member=member_name,
                team=team_name,
                timeout=timeout,
            )
            return
        yield item


class BaseTeam[TDeps, TResult](MessageNode[TDeps, TResult]):
    """A concrete team of agents that executes in parallel or sequential mode.

    ``BaseTeam`` replaces the earlier ``Team`` (parallel) / ``TeamRun``
    (sequential) hierarchy.  The ``mode`` parameter determines execution
    strategy.
    """

    _error_mode: Literal["fail_all", "collect_exceptions"] = "collect_exceptions"

    def __init__(
        self,
        agents: Sequence[MessageNode[TDeps, TResult]],
        *,
        mode: ExecutionMode = "parallel",
        name: str | None = None,
        description: str | None = None,
        display_name: str | None = None,
        shared_prompt: str | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        event_configs: Sequence[EventConfig] | None = None,
        agent_pool: AgentPool | None = None,
        member_prompt_templates: dict[str, TeamMemberConfig] | None = None,
        member_timeout: float | None = None,
        validator: MessageNode[Any, TResult] | None = None,
    ) -> None:
        """Initialize a team of agents.

        Args:
            agents: The agents/nodes that are part of this team.
            mode: Execution mode — ``"parallel"`` (default) or ``"sequential"``.
            name: Optional name for the team.
            description: Optional description.
            display_name: Optional human-readable name.
            shared_prompt: Prompt prepended to all member inputs.
            mcp_servers: MCP servers to make available to members.
            event_configs: Event configuration.
            agent_pool: The pool this team belongs to.
            member_prompt_templates: Per-member Jinja2 prompt templates.
            member_timeout: Per-member execution timeout in seconds.
            validator: An optional validator node (sequential mode only).
        """
        from wolfharness.delegation.graph_team import ExtendedTeamTalk

        self._name = name or " & ".join([i.name for i in agents])
        self._nodes: list[MessageNode[Any, Any]] = []
        self.mode: ExecutionMode = mode
        super().__init__(
            name=self._name,
            display_name=display_name,
            mcp_servers=mcp_servers,
            description=description,
            event_configs=event_configs,
            agent_pool=agent_pool,
        )
        for agent in agents:
            self.add_node(agent)
        self._team_talk = ExtendedTeamTalk()
        self.shared_prompt = shared_prompt
        self.member_prompt_templates = member_prompt_templates or {}
        self.member_timeout = member_timeout
        self.validator = validator
        self.result_mode: Literal["last", "concat"] = "last"
        self._main_task: asyncio.Task[ChatMessage[Any] | None] | None = None
        self._infinite = False

    @property
    def nodes(self) -> list[MessageNode[Any, Any]]:
        return self._nodes

    def add_node(self, node: MessageNode[Any, Any]) -> None:
        """Handler for adding new nodes to the team."""
        from wolfharness.agents import Agent

        self._nodes.append(node)
        if isinstance(node, Agent):
            aggregating_provider = self.mcp.get_aggregating_provider()
            node._external_capabilities.append(aggregating_provider)

    def remove_node(self, node: MessageNode[Any, Any]) -> None:
        """Handler for removing nodes from the team."""
        from wolfharness.agents import Agent

        self._nodes.remove(node)
        if isinstance(node, Agent):
            aggregating_provider = self.mcp.get_aggregating_provider()
            with contextlib.suppress(ValueError):
                node._external_capabilities.remove(aggregating_provider)

    def __repr__(self) -> str:
        """Create readable representation."""
        members = ", ".join(node.name for node in self.nodes)
        name = f" ({self.name})" if self.name else ""
        mode_label = "par" if self.mode == "parallel" else "seq"
        return f"BaseTeam[{mode_label},{len(self.nodes)}]{name}: {members}"

    def __len__(self) -> int:
        """Get number of team members."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[MessageNode[TDeps, TResult]]:
        """Iterate over team members."""
        return iter(self.nodes)

    def __getitem__(self, index_or_name: int | str) -> MessageNode[TDeps, TResult]:
        """Get team member by index or name."""
        if isinstance(index_or_name, str):
            return next(node for node in self.nodes if node.name == index_or_name)
        return self.nodes[index_or_name]

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def __or__(
        self,
        other: Agent[Any, Any] | ProcessorCallback[Any] | BaseTeam[Any, Any],
    ) -> BaseTeam[Any, Any]:
        """Create a sequential pipeline."""
        from wolfharness.agents import Agent

        if callable(other):
            other = Agent.from_callback(other, agent_pool=self._agent_pool)

        # If we're already a sequential team with no validator, extend it
        if self.mode == "sequential" and not self.validator:
            self._nodes.append(other)
            return self
        # Otherwise create new sequential team
        return BaseTeam([self, other], mode="sequential")

    @overload
    def __and__(self, other: BaseTeam[None, Any]) -> BaseTeam[None, Any]: ...
    @overload
    def __and__(self, other: BaseTeam[TDeps, Any]) -> BaseTeam[TDeps, Any]: ...
    @overload
    def __and__(self, other: BaseTeam[Any, Any]) -> BaseTeam[Any, Any]: ...
    @overload
    def __and__(self, other: Agent[TDeps, Any]) -> BaseTeam[TDeps, Any]: ...
    @overload
    def __and__(self, other: Agent[Any, Any]) -> BaseTeam[Any, Any]: ...

    def __and__(
        self,
        other: BaseTeam[Any, Any] | Agent[Any, Any] | ProcessorCallback[Any],
    ) -> BaseTeam[Any, Any]:
        """Combine teams, preserving type safety for same types."""
        from wolfharness.agents import Agent

        if callable(other):
            other = Agent.from_callback(other, agent_pool=self._agent_pool)

        match other:
            case BaseTeam() if other.mode == "parallel":
                # Flatten when combining parallel teams
                return BaseTeam([*self.nodes, *other.nodes], mode="parallel")
            case _:
                # Everything else just becomes a member
                return BaseTeam([*self.nodes, other], mode="parallel")

    # ------------------------------------------------------------------
    # Stats & lifecycle
    # ------------------------------------------------------------------

    async def get_stats(self) -> AggregatedMessageStats:
        """Get aggregated stats from all team members."""
        stats = [await node.get_stats() for node in self.nodes]
        return AggregatedMessageStats(stats=stats)

    @property
    def is_running(self) -> bool:
        """Whether execution is currently running."""
        return bool(self._main_task and not self._main_task.done())

    def is_busy(self) -> bool:
        """Check if team is processing any tasks."""
        return bool(self._pending_tasks or self._main_task)

    async def stop(self) -> None:
        """Stop background execution if running."""
        try:
            if self._main_task and not self._main_task.done():
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.log.warning("Main task raised during shutdown", exc_info=e)
        finally:
            self._main_task = None
            await self._cleanup_pending_tasks()

    async def wait(self) -> ChatMessage[Any] | None:
        """Wait for background execution to complete and return last message."""
        if not self._main_task:
            raise RuntimeError("No execution running")
        if self._infinite:
            raise RuntimeError("Cannot wait on infinite execution")
        try:
            return await self._main_task
        finally:
            await self._cleanup_pending_tasks()
            self._main_task = None

    async def run_in_background(
        self,
        *prompts: PromptCompatible | None,
        max_count: int | None = 1,
        interval: float = 1.0,
        **kwargs: Any,
    ) -> ExtendedTeamTalk:
        """Start execution in background.

        Args:
            prompts: Prompts to execute
            max_count: Maximum number of executions (None = run indefinitely)
            interval: Seconds between executions
            **kwargs: Additional args for execute()
        """
        if self._main_task:
            raise RuntimeError("Execution already running")
        self._infinite = max_count is None

        async def _continuous() -> ChatMessage[Any] | None:
            count = 0
            last_message = None
            while max_count is None or count < max_count:
                try:
                    result = await self.execute(*prompts, **kwargs)
                    last_message = result[-1].message if result else None
                    count += 1
                    if max_count is None or count < max_count:
                        await anyio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("Background execution cancelled")
                    break
            return last_message

        self._main_task = asyncio.create_task(_continuous(), name="main_execution")
        return self._team_talk

    @property
    def execution_stats(self) -> AggregatedTalkStats:
        """Get current execution statistics."""
        return self._team_talk.stats

    @property
    def talk(self) -> ExtendedTeamTalk:
        """Get current connection."""
        return self._team_talk

    @property
    async def cancel(self) -> None:
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self._cleanup_pending_tasks()

    async def _cleanup_pending_tasks(self) -> None:
        """Wait for all pending background tasks to complete, then clear."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

    def get_structure_diagram(self) -> str:
        """Generate mermaid flowchart of node hierarchy."""
        lines = ["flowchart TD"]

        def add_node(node: MessageNode[Any, Any], parent: str | None = None) -> None:
            node_id = f"node_{id(node)}"
            lines.append(f"    {node_id}[{node.name}]")
            if parent:
                lines.append(f"    {parent} --> {node_id}")
            if isinstance(node, BaseTeam):
                for member in node.nodes:
                    add_node(member, node_id)

        for node in self.nodes:
            add_node(node)
        return "\n".join(lines)

    def iter_agents(self) -> Iterator[BaseAgent[Any, Any]]:
        """Recursively iterate over all child agents."""
        for node in self.nodes:
            match node:
                case BaseTeam():
                    yield from node.iter_agents()
                case BaseAgent():
                    yield node
                case _:
                    raise ValueError(f"Invalid node type: {type(node)}")

    def get_context(
        self,
        data: Any = None,
        input_provider: InputProvider | None = None,
    ) -> TeamContext:
        """Create a new context for this team.

        Args:
            data: Optional custom data to attach to the context
            input_provider: Optional input provider override

        Returns:
            A new TeamContext instance

        Raises:
            ValueError: If team members belong to different pools
        """
        pool_ids: set[int] = set()
        shared_pool: AgentPool | None = None

        for agent in self.iter_agents():
            # TODO(m4): Migrate to `agent.host_context` once NodeContext.pool
            # is replaced with NodeContext.host: HostContext | None.
            # Requires adding `input_provider` to HostContext (see M4 task 14.8).
            pool = agent._agent_pool
            if pool:
                pool_id = id(pool)
                if pool_id not in pool_ids:
                    pool_ids.add(pool_id)
                    shared_pool = pool

        if not pool_ids:
            logger.debug("No pool found for team.", team=self.name)
            return TeamContext(
                node=self,
                pool=shared_pool,
                input_provider=input_provider,
                data=data,
            )

        if len(pool_ids) > 1:
            raise ValueError(f"Team members in {self.name} belong to different pools")
        return TeamContext(
            node=self,
            pool=shared_pool,
            input_provider=input_provider,
            data=data,
        )

    # ------------------------------------------------------------------
    # Core execution — dispatch by mode
    # ------------------------------------------------------------------

    def __prompt__(self) -> str:
        """Format team info for prompts."""
        if self.mode == "parallel":
            members = ", ".join(a.name for a in self.nodes)
            desc = f" - {self.description}" if self.description else ""
            return f"Parallel Team {self.name!r}{desc}\nMembers: {members}"
        members = " -> ".join(a.name for a in self.nodes)
        desc = f" - {self.description}" if self.description else ""
        return f"Sequential Team {self.name!r}{desc}\nPipeline: {members}"

    async def execute(
        self,
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Execute the team, dispatching by ``self.mode``.

        Args:
            *prompts: Prompt-compatible inputs to pass to team members.
            **kwargs: Extra keyword arguments.

        Returns:
            A :class:`TeamResponse` with member results.
        """
        if self.mode == "parallel":
            return await self._execute_parallel(*prompts, **kwargs)
        return await self._execute_sequential(*prompts, **kwargs)

    async def execute_iter(
        self,
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> AsyncIterator[Talk[Any] | AgentResponse[Any]]:
        """Yield ``AgentResponse`` and ``Talk`` objects in execution order.

        Only available in sequential mode — builds a chained graph and
        yields results as each step completes.
        """
        from pydantic_graph import GraphBuilder, Step
        from pydantic_graph.id_types import NodeID

        from wolfharness.delegation.graph_team import _make_sequential_step, _TeamRunGraphState

        all_nodes = list(self.nodes)
        if self.validator:
            all_nodes.append(self.validator)

        connections: list[Talk[Any]] = []
        for source, target in pairwise(all_nodes):
            talk = Talk[Any](
                source=source,
                targets=[target],
                connection_type="run",
                queued=True,
            )
            connections.append(talk)
            self._team_talk.append(talk)

        state = _TeamRunGraphState(
            prompts=tuple(prompts),
            kwargs=kwargs,
            connections=connections,
        )

        steps: list[Any] = []
        for i, node in enumerate(all_nodes):
            step_fn = _make_sequential_step(node, i)
            step = Step(
                id=NodeID(f"{node.name}_{i}"),
                call=step_fn,
                label=node.description or node.name,
            )
            steps.append(step)

        builder = GraphBuilder(
            state_type=_TeamRunGraphState,
            input_type=Any,
            output_type=ChatMessage[Any],
        )
        builder.add_edge(builder.start_node, steps[0])
        for s, t in pairwise(steps):
            builder.add_edge(s, t)
        builder.add_edge(steps[-1], builder.end_node)
        graph = builder.build()

        try:
            await graph.run(state=state, deps=None, inputs=None)
        except Exception as exc:  # noqa: BLE001
            unwrapped: BaseException = exc
            if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                unwrapped = exc.exceptions[0]
            for i, response in enumerate(state.responses):
                yield response
                if i < len(connections):
                    yield connections[i]
            raise unwrapped from None

        if len(state.responses) == len(all_nodes) and len(all_nodes) > 1:
            last_response = state.responses[-1]
            last_talk = Talk[Any](all_nodes[-1], [], connection_type="run")
            if last_response.message:
                last_talk._stats.messages.append(last_response.message)
            self._team_talk.append(last_talk)

        for i, response in enumerate(state.responses):
            yield response
            if i < len(connections):
                yield connections[i]

    async def run(
        self,
        *prompts: PromptCompatible | None,
        wait_for_connections: bool | None = None,
        store_history: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[Any]:
        """Run the team and return a single :class:`ChatMessage`.

        In parallel mode, content is a list of all member outputs.
        In sequential mode, content is determined by ``self.result_mode``
        (last member output by default).
        """
        user_msg, processed_prompts = await prepare_prompts(*prompts)
        await self.message_received.emit(user_msg)
        message_id = str(uuid4())

        result = await self.execute(*processed_prompts, **kwargs)

        if self.mode == "parallel":
            message = ChatMessage(
                content=[r.message.content for r in result if r.message],
                messages=[m for r in result if r.message for m in r.message.messages],
                role="assistant",
                name=self.name,
                message_id=message_id,
                session_id=user_msg.session_id,
                parent_id=user_msg.message_id,
                metadata={
                    "agent_names": [r.agent_name for r in result],
                    "errors": {name: str(error) for name, error in result.errors.items()},
                    "start_time": result.start_time.isoformat(),
                    "child_session_ids": dict[str, bool | int | float | str](
                        result.child_session_ids
                    ),
                },
            )
        else:
            all_messages = [r.message for r in result if r.message]
            assert all_messages, "Error during execution, returned None for team"
            match self.result_mode:
                case "last":
                    content = all_messages[-1].content
                case "concat":
                    content = "\n".join(str(msg.content) for msg in all_messages)
                case _:
                    raise ValueError(f"Invalid result mode: {self.result_mode}")

            message = ChatMessage(
                content=content,
                messages=[m for chat_message in all_messages for m in chat_message.messages],
                role="assistant",
                name=self.name,
                associated_messages=all_messages,
                message_id=message_id,
                session_id=user_msg.session_id,
                parent_id=user_msg.message_id,
                metadata={
                    "execution_order": [r.agent_name for r in result],
                    "start_time": result.start_time.isoformat(),
                    "errors": {name: str(error) for name, error in result.errors.items()},
                },
            )

        if store_history:
            pass
        return await finalize_message(
            message,
            user_msg,
            self,
            self.connections,
            wait_for_connections,
        )

    async def run_iter(
        self,
        *prompts: AnyPromptType,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages as they arrive.

        In parallel mode, yields messages from all members as they
        complete.  In sequential mode, yields messages from the chain.
        """
        if self.mode == "parallel":
            async for msg in self._run_iter_parallel(*prompts, **kwargs):
                yield msg
        else:
            async for msg in self._run_iter_sequential(*prompts, **kwargs):
                yield msg

    async def run_stream(
        self,
        *prompts: PromptCompatible,
        depth: int = 0,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Stream responses from team members.

        In parallel mode, streams all members concurrently.
        In sequential mode, streams members one after another,
        passing each member's output as input to the next.
        """
        if self.mode == "parallel":
            async for event in self._run_stream_parallel(*prompts, depth=depth, **kwargs):
                yield event
        else:
            async for event in self._run_stream_sequential(*prompts, depth=depth, **kwargs):
                yield event

    # ------------------------------------------------------------------
    # Parallel execution (was ``Team``)
    # ------------------------------------------------------------------

    @staticmethod
    def _active_parent_session_id() -> str | None:
        """Return the active SessionPool session id when running inside a turn."""
        from wolfharness.agents.base_agent import _current_run_ctx_var

        run_ctx = _current_run_ctx_var.get()
        session_id = getattr(run_ctx, "session_id", None)
        return str(session_id) if session_id else None

    async def _resolve_scoped_team_nodes(
        self,
        nodes: list[MessageNode[Any, Any]],
        parent_session_id: str | None,
        team_run_id: str,
    ) -> tuple[list[MessageNode[Any, Any]], dict[str, str]]:
        """Resolve team members to child-session agents for a parent session."""
        ctx = self.host_context
        if not parent_session_id or ctx is None or ctx.session_pool is None:
            return nodes, {}

        from wolfharness.agents.base_agent import BaseAgent
        from wolfharness.utils.identifiers import generate_session_id

        session_pool = ctx.session_pool
        pool_agents = ctx.manifest.agents
        pool_teams = ctx.manifest.teams
        scoped_nodes: list[MessageNode[Any, Any]] = []
        child_session_ids: dict[str, str] = {}

        await self._save_scoped_storage_session(parent_session_id)
        for node in nodes:
            if node.name not in pool_agents and node.name not in pool_teams:
                scoped_nodes.append(node)
                continue
            child_state = await session_pool.create_session(
                session_id=generate_session_id(),
                parent_session_id=parent_session_id,
                lifecycle_policy="cascade",
                agent_name=node.name,
                agent_type=getattr(node, "agent_type", type(node).__name__),
                team_name=self.name,
                team_run_id=team_run_id,
                generate_title=False,
            )
            child_session_id = child_state.session_id
            child_node: MessageNode[Any, Any]
            if isinstance(node, BaseAgent) and node.name in pool_agents:
                child_node = await session_pool.sessions.get_or_create_session_agent(
                    child_session_id,
                    agent_name=node.name,
                )
            else:
                child_node = node
            scoped_nodes.append(child_node)
            child_session_ids[node.name] = child_session_id
            await self._save_scoped_storage_session(child_session_id)

        return scoped_nodes, child_session_ids

    async def _save_scoped_storage_session(self, session_id: str | None) -> None:
        """Persist SessionPool session data to protocol storage for lineage checks."""
        ctx = self.host_context
        if not session_id or ctx is None or ctx.session_pool is None:
            return
        storage = ctx.storage
        if storage is None:
            return
        session_controller = ctx.session_pool.sessions
        session = session_controller.get_session(session_id)
        if session is None:
            return
        await storage.save_session(session_controller._state_to_data(session))

    async def _close_scoped_team_nodes(self, child_session_ids: dict[str, str]) -> None:
        """Close and delete round-scoped child sessions created for a team run."""
        ctx = self.host_context
        if not child_session_ids or ctx is None or ctx.session_pool is None:
            return
        for session_id in reversed(list(child_session_ids.values())):
            await ctx.session_pool.close_session(session_id)
            await self._delete_scoped_storage_session(session_id)

    async def _delete_scoped_storage_session(self, session_id: str) -> None:
        """Remove protocol storage written for a round-scoped child session."""
        ctx = self.host_context
        if ctx is None:
            return
        storage = ctx.storage
        if storage is None:
            return
        await storage.delete_session_messages(session_id)
        await storage.delete_session(session_id)

    @staticmethod
    def _normalize_member_skills(value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, list[str]] = {}
        for member_name, raw_names in value.items():
            name = str(member_name).strip()
            if not name:
                continue
            raw_list = raw_names if isinstance(raw_names, list) else [raw_names]
            names = [str(item).strip() for item in raw_list if str(item).strip()]
            if names:
                result[name] = list(dict.fromkeys(names))
        return result

    async def _load_member_skill_instructions(
        self,
        member_skills: dict[str, list[str]],
    ) -> dict[str, str]:
        if not member_skills or self._agent_pool is None:
            return {}

        result: dict[str, str] = {}
        for member_name, skill_names in member_skills.items():
            loaded_sections: list[str] = []
            for skill_name in skill_names:
                instructions = await self._load_skill_instructions(skill_name, member_name)
                if instructions:
                    loaded_sections.append(self._format_skill_instruction(skill_name, instructions))
            if loaded_sections:
                result[member_name] = "\n\n".join(loaded_sections)
        return result

    async def _load_skill_instructions(self, skill_name: str, member_name: str) -> str:
        pool = self._agent_pool
        if pool is None or pool.skill_resolver is None:
            from wolfharness.skills.exceptions import SkillNotFoundError

            raise SkillNotFoundError(skill_name)

        return await pool.get_skill_instructions_for_node(skill_name, member_name)

    @staticmethod
    def _format_skill_instruction(skill_name: str, instructions: str) -> str:
        return (
            f'<skill-instruction name="{escape(skill_name)}">\n'
            f"{instructions.strip()}\n"
            "</skill-instruction>"
        )

    @staticmethod
    def _inject_member_skill_instructions(
        member_name: str,
        prompts: list[PromptCompatible | None],
        member_skill_instructions: dict[str, str],
    ) -> list[PromptCompatible | None]:
        instructions = member_skill_instructions.get(member_name, "").strip()
        if not instructions:
            return prompts
        prompt_text = "\n\n".join(str(prompt) for prompt in prompts if prompt is not None).strip()
        combined = f"{instructions}\n\n{prompt_text}" if prompt_text else instructions
        return [combined]

    def _resolve_member_prompt(
        self,
        member_name: str,
        default_prompt: list[PromptCompatible | None],
        raw_prompts: tuple[PromptCompatible | None, ...],
        template_vars: dict[str, Any],
    ) -> list[PromptCompatible | None]:
        """Return per-member prompt if a template is configured, else the default."""
        cfg = self.member_prompt_templates.get(member_name)
        if cfg is None or cfg.prompt_template is None:
            return default_prompt

        prompt_str = " ".join(str(p) for p in raw_prompts if p is not None)
        rendered = _PROMPT_TEMPLATE_ENV.from_string(cfg.prompt_template).render(
            prompt=prompt_str,
            shared_prompt=self.shared_prompt or "",
            extra=template_vars,
        )
        return [rendered]

    @logfire.instrument("team.execute_parallel")
    async def _execute_parallel(
        self,
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Run all agents in parallel via pydantic-graph Fork + Join."""
        from wolfharness.delegation.graph_team import _TeamGraphState, run_team_graph

        self._team_talk.clear()
        default_prompt = list(prompts)
        if self.shared_prompt:
            default_prompt.insert(0, self.shared_prompt)

        session_id_kwarg = str(kwargs.pop("session_id", "") or "")
        parent_session_id_kwarg = str(kwargs.pop("parent_session_id", "") or "")
        if session_id_kwarg:
            kwargs["session_id"] = session_id_kwarg
        if parent_session_id_kwarg:
            kwargs["parent_session_id"] = parent_session_id_kwarg
        parent_session_id = (
            parent_session_id_kwarg or session_id_kwarg or self._active_parent_session_id()
        )
        team_run_id = uuid4().hex
        template_vars = kwargs.pop("template_vars", {})
        if not isinstance(template_vars, dict):
            template_vars = {}
        timeout = self.member_timeout
        member_retry_attempts = max(0, int(kwargs.pop("member_retry_attempts", 0) or 0))
        member_retry_delay = max(0.0, float(kwargs.pop("member_retry_delay", 0.0) or 0.0))
        member_skills = self._normalize_member_skills(kwargs.pop("member_skills", {}))
        member_skill_instructions = await self._load_member_skill_instructions(member_skills)

        base_nodes = list(self.nodes)
        all_nodes, child_session_ids = await self._resolve_scoped_team_nodes(
            base_nodes,
            parent_session_id,
            team_run_id,
        )
        from wolfharness.talk.talk import Talk

        execution_talks: list[Talk[Any]] = []
        member_prompts: dict[str, list[PromptCompatible | None]] = {}
        for node in all_nodes:
            talk = Talk[Any](node, [], connection_type="run", queued=True, queue_strategy="latest")
            execution_talks.append(talk)
            self._team_talk.append(talk)
            resolved = self._resolve_member_prompt(
                node.name,
                default_prompt,
                prompts,
                template_vars,
            )
            resolved = self._inject_member_skill_instructions(
                node.name,
                resolved,
                member_skill_instructions,
            )
            member_prompts[node.name] = resolved

        state = _TeamGraphState(
            prompts=prompts,
            member_prompts=member_prompts,
            kwargs=kwargs,
            shared_prompt=self.shared_prompt,
            child_session_ids=child_session_ids,
            parent_session_id=parent_session_id,
            member_timeout=timeout,
            member_retry_attempts=member_retry_attempts,
            member_retry_delay=member_retry_delay,
            execution_talks=execution_talks,
            error_mode=self._error_mode,
        )

        try:
            return await run_team_graph(all_nodes, state)
        finally:
            await self._close_scoped_team_nodes(child_session_ids)

    @logfire.instrument("team.execute_sequential")
    async def _execute_sequential(
        self,
        *prompts: PromptCompatible | None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Run all agents sequentially via pydantic-graph chained steps."""
        from wolfharness.delegation.graph_team import run_teamrun_graph

        self._team_talk.clear()
        start_time = get_now()
        prompts_list = list(prompts)
        if self.shared_prompt:
            prompts_list.insert(0, self.shared_prompt)

        all_nodes = list(self.nodes)
        if self.validator:
            all_nodes.append(self.validator)

        connections: list[Talk[Any]] = []
        for source, target in pairwise(all_nodes):
            talk = Talk[Any](
                source=source,
                targets=[target],
                connection_type="run",
                queued=True,
            )
            connections.append(talk)
            self._team_talk.append(talk)

        responses = await run_teamrun_graph(
            nodes=all_nodes,
            prompts=tuple(prompts_list),
            kwargs=kwargs,
            connections=connections,
        )

        if len(responses) == len(all_nodes) and len(all_nodes) > 1:
            last_response = responses[-1]
            last_talk: Talk[Any] = Talk[Any](all_nodes[-1], [], connection_type="run")
            if last_response.message:
                last_talk._stats.messages.append(last_response.message)
            self._team_talk.append(last_talk)

        team_responses = list(responses)
        errors: dict[str, Exception] = {}
        return TeamResponse(team_responses, start_time, errors=errors)

    async def _run_iter_parallel(
        self,
        *prompts: AnyPromptType,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages as they arrive from parallel execution."""
        queue: asyncio.Queue[ChatMessage[Any] | None] = asyncio.Queue()
        failures: dict[str, Exception] = {}
        timeout = self.member_timeout

        async def _run(node: MessageNode[TDeps, Any]) -> None:
            try:
                coro = node.run(*prompts, **kwargs)
                message = (
                    await asyncio.wait_for(coro, timeout=timeout)
                    if timeout is not None
                    else await coro
                )
                await queue.put(message)
            except TimeoutError:
                logger.warning(
                    "Team member timed out",
                    member=node.name,
                    team=self.name,
                    timeout=timeout,
                )
                failures[node.name] = TimeoutError(
                    f"Member {node.name!r} exceeded {timeout}s deadline"
                )
                await queue.put(None)
            except Exception as e:
                logger.exception("Error executing node", name=node.name)
                failures[node.name] = e
                await queue.put(None)

        all_nodes = list(self.nodes)
        tasks = [asyncio.create_task(_run(n), name=f"run_{n.name}") for n in all_nodes]
        try:
            for _ in all_nodes:
                if msg := await queue.get():
                    yield msg
            if failures:
                error_details = "\n".join(f"- {name}: {error}" for name, error in failures.items())
                error_msg = f"Some nodes failed to execute:\n{error_details}"
                raise RuntimeError(error_msg)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

    async def _run_iter_sequential(
        self,
        *prompts: PromptCompatible,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages from the sequential execution chain."""
        from wolfharness.talk.talk import Talk

        all_nodes = list(self.nodes)
        if self.validator:
            all_nodes.append(self.validator)

        connections: list[Talk[Any]] = []
        for source, target in pairwise(all_nodes):
            talk = Talk[Any](
                source=source,
                targets=[target],
                connection_type="run",
                queued=True,
            )
            connections.append(talk)
            self._team_talk.append(talk)

        from wolfharness.delegation.graph_team import run_teamrun_graph

        responses = await run_teamrun_graph(
            nodes=all_nodes,
            prompts=prompts,
            kwargs=kwargs,
            connections=connections,
        )

        for response in responses:
            if response.message:
                yield response.message

    async def _run_stream_parallel(
        self,
        *prompts: PromptCompatible,
        depth: int = 0,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Stream responses from all team members in parallel."""
        from wolfharness.common_types import SupportsRunStream
        from wolfharness.utils.identifiers import generate_session_id

        session_id_kwarg: str | None = kwargs.pop("session_id", None)
        kwargs.pop("depth", None)
        parent_session_id_kwarg: str | None = kwargs.pop("parent_session_id", None)

        child_depth = depth + 1
        if child_depth > MAX_DELEGATION_DEPTH:
            raise DelegationDepthError(child_depth)

        parent_sid: str | None = parent_session_id_kwarg or session_id_kwarg

        base_nodes = list(self.nodes)
        all_nodes, child_session_ids = await self._resolve_scoped_team_nodes(
            base_nodes,
            parent_sid,
            uuid4().hex,
        )
        timeout = self.member_timeout

        async def wrap_stream(
            node: MessageNode[Any, Any],
            child_sid: str,
        ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
            source_type = get_source_type(node)

            yield SpawnSessionStart(
                child_session_id=child_sid,
                parent_session_id=parent_sid or "",
                source_type=source_type,
                source_name=node.name,
                display_name=node.display_name if node.display_name != node.name else None,
                depth=child_depth,
                description=f"Spawning {node.name} as team member",
                spawn_mechanism="spawn",
            )

            if not isinstance(node, SupportsRunStream):
                return

            node_model_id: str | None = None
            if isinstance(node, BaseAgent):
                node_model_id = node.model_name

            stream = node.run_stream(
                *prompts,
                session_id=child_sid,
                parent_session_id=parent_sid,
                depth=child_depth,
                **kwargs,
            )
            async for event in _timeout_stream(stream, timeout, node.name, self.name):
                if isinstance(event, SubAgentEvent):
                    yield SubAgentEvent(
                        source_name=event.source_name,
                        source_type=event.source_type,
                        event=event.event,
                        depth=event.depth + 1,
                        model_id=event.model_id,
                        mode=event.mode,
                        child_session_id=event.child_session_id,
                        parent_session_id=event.parent_session_id,
                    )
                else:
                    yield SubAgentEvent(
                        source_name=node.name,
                        source_type=source_type,
                        event=event,
                        depth=child_depth,
                        model_id=node_model_id,
                        child_session_id=child_sid,
                        parent_session_id=parent_sid,
                    )

        streams = [
            wrap_stream(node, child_session_ids.get(node.name, generate_session_id()))
            for node in all_nodes
        ]
        try:
            async for event in as_generated(streams):
                yield event
        finally:
            await self._close_scoped_team_nodes(child_session_ids)

    async def _run_stream_sequential(
        self,
        *prompts: PromptCompatible,
        require_all: bool = True,
        depth: int = 0,
        **kwargs: Any,
    ) -> AsyncIterator[RichAgentStreamEvent[Any]]:
        """Stream responses through the chain of team members."""
        from wolfharness.agents.events import StreamCompleteEvent
        from wolfharness.common_types import SupportsRunStream
        from wolfharness.utils.identifiers import generate_session_id

        session_id_kwarg: str | None = kwargs.pop("session_id", None)
        kwargs.pop("depth", None)
        parent_session_id_kwarg: str | None = kwargs.pop("parent_session_id", None)

        parent_session_id: str | None = parent_session_id_kwarg or session_id_kwarg

        child_depth = depth + 1
        if child_depth > MAX_DELEGATION_DEPTH:
            raise DelegationDepthError(child_depth)

        current_message = prompts
        for node in self.nodes:
            source_type = get_source_type(node)

            if not isinstance(node, SupportsRunStream):
                raise TypeError(f"Node {node.name} does not support streaming")
            try:
                ctx = self.host_context
                if (
                    ctx is not None
                    and ctx.session_pool is not None
                    and parent_session_id is not None
                ):
                    child_state = await ctx.session_pool.create_session(
                        session_id=generate_session_id(),
                        parent_session_id=parent_session_id,
                        agent_name=node.name,
                        agent_type=node.agent_type,
                        generate_title=False,
                    )
                    child_sid = child_state.session_id
                else:
                    child_sid = generate_session_id()

                yield SpawnSessionStart(
                    child_session_id=child_sid,
                    parent_session_id=parent_session_id or "",
                    spawn_mechanism="spawn",
                    source_type=source_type,
                    source_name=node.name,
                    display_name=node.display_name if node.display_name != node.name else None,
                    depth=child_depth,
                    description=f"Sequential team member {node.name!r}",
                )

                node_model_id: str | None = None
                if isinstance(node, BaseAgent):
                    node_model_id = node.model_name

                async for event in node.run_stream(
                    *current_message,
                    session_id=child_sid,
                    parent_session_id=parent_session_id,
                    depth=child_depth,
                    **kwargs,
                ):
                    if isinstance(event, SubAgentEvent):
                        yield SubAgentEvent(
                            source_name=event.source_name,
                            source_type=event.source_type,
                            event=event.event,
                            depth=event.depth + 1,
                            child_session_id=event.child_session_id,
                            parent_session_id=event.parent_session_id,
                            model_id=event.model_id,
                            mode=event.mode,
                        )
                    else:
                        yield SubAgentEvent(
                            source_name=node.name,
                            source_type=source_type,
                            event=event,
                            depth=child_depth,
                            child_session_id=child_sid,
                            parent_session_id=parent_session_id,
                            model_id=node_model_id,
                        )
                        if isinstance(event, StreamCompleteEvent):
                            current_message = (event.message.content,)

            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {node.name}: {e}"
                    logger.exception(msg)
                    raise ValueError(msg) from e
                logger.warning("Chain handler failed", name=node.name, error=e)


if __name__ == "__main__":

    async def main() -> None:
        from wolfharness import Agent, BaseTeam

        agent = Agent("My Agent", model="openai:gpt-5-nano")
        agent_2 = Agent("My Agent", model="openai:gpt-5-nano")
        team = BaseTeam([agent, agent_2], mcp_servers=["uvx mcp-server-git"])
        async with team:
            print(await agent._get_all_tools())

    anyio.run(main)
