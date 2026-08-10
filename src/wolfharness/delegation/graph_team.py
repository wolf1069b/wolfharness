"""Graph-based team execution using pydantic-graph Fork + Join and sequential chains.

This module provides graph-based implementations for both parallel (Fork + Join)
and sequential (chained steps) team execution using :class:`pydantic_graph.GraphBuilder`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from pydantic_graph import GraphBuilder, Step, StepContext, reduce_list_append

from wolfharness.log import get_logger
from wolfharness.messaging import AgentResponse, ChatMessage, TeamResponse
from wolfharness.talk.talk import Talk, TeamTalk
from wolfharness.utils.time_utils import get_now


if TYPE_CHECKING:
    from datetime import datetime

    from wolfharness.messaging.messagenode import MessageNode


logger = get_logger(__name__)


def _raise_no_message(node_name: str) -> NoReturn:
    """Raise RuntimeError for a member that returned no message."""
    msg = f"Member {node_name!r} returned no message"
    raise RuntimeError(msg)


@dataclass
class _MemberOutput:
    """Result from a single team member execution within the graph."""

    agent_name: str
    """Name of the agent that produced this result."""

    response: AgentResponse[Any] | None = None
    """Successful response, if any."""

    exception: Exception | None = None
    """Exception raised during execution, if any."""


@dataclass
class _TeamGraphState:
    """Shared state passed through the pydantic-graph execution."""

    prompts: tuple[Any, ...] = field(default_factory=tuple)
    """Input prompts for this execution."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to member ``run()``."""

    shared_prompt: str | None = None
    """Optional prompt prepended to all member inputs."""

    member_prompts: dict[str, list[Any]] = field(default_factory=dict)
    """Resolved prompt list per member name."""

    child_session_ids: dict[str, str] = field(default_factory=dict)
    """Session id allocated for each member in this team run."""

    parent_session_id: str | None = None
    """Parent session id for scoped team member runs."""

    member_timeout: float | None = None
    """Maximum seconds a member may run before being cancelled."""

    member_retry_attempts: int = 0
    """Additional attempts for non-timeout runtime member failures."""

    member_retry_delay: float = 0.0
    """Seconds to wait before retrying a failed member."""

    execution_talks: list[Talk[Any]] = field(default_factory=list)
    """Talk connections for tracking execution stats."""

    error_mode: Literal["fail_all", "collect_exceptions"] = "collect_exceptions"
    """How to handle member failures:
    - ``fail_all``: raise immediately on first failure
    - ``collect_exceptions``: catch all failures and return them in errors dict
    """


def _make_member_step(
    node: MessageNode[Any, Any],
) -> Any:
    """Create a pydantic-graph step function for a team member.

    The returned step runs ``node.run()`` with the prompts and kwargs stored
    in :attr:`_TeamGraphState`, records timing, updates the corresponding
    :class:`Talk` stats, and returns a :class:`_MemberOutput`.

    Args:
        node: The team member node to wrap.

    Returns:
        An async callable compatible with :meth:`GraphBuilder.step`.
    """

    async def _step(
        ctx: StepContext,
    ) -> _MemberOutput:
        state = cast(_TeamGraphState, ctx.state)
        final_prompt = state.member_prompts.get(node.name)
        if final_prompt is None:
            final_prompt = list(state.prompts)
        if state.shared_prompt and node.name not in state.member_prompts:
            final_prompt.insert(0, state.shared_prompt)

        try:
            start = perf_counter()
            run_kwargs = dict(state.kwargs)
            if child_session_id := state.child_session_ids.get(node.name):
                run_kwargs["session_id"] = child_session_id
            if state.parent_session_id:
                run_kwargs["parent_session_id"] = state.parent_session_id
            attempts = max(1, state.member_retry_attempts + 1)
            message = None
            for attempt_index in range(attempts):
                try:
                    coro = node.run(*final_prompt, **run_kwargs)
                    message = (
                        await asyncio.wait_for(coro, timeout=state.member_timeout)
                        if state.member_timeout is not None
                        else await coro
                    )
                    break
                except TimeoutError:
                    raise
                except RuntimeError as exc:
                    if attempt_index >= attempts - 1:
                        raise
                    logger.warning(
                        "Team member failed; retrying",
                        member=node.name,
                        attempt=attempt_index + 1,
                        max_attempts=attempts,
                        error=str(exc),
                    )
                    if state.member_retry_delay > 0:
                        await asyncio.sleep(state.member_retry_delay)
            if message is None:
                _raise_no_message(node.name)
            timing = perf_counter() - start
            response = AgentResponse(agent_name=node.name, message=message, timing=timing)

            # Update talk stats for this agent
            talk = next(
                (t for t in state.execution_talks if t.source == node),
                None,
            )
            if talk is not None:
                talk._stats.messages.append(message)

            return _MemberOutput(agent_name=node.name, response=response)

        except TimeoutError as exc:
            logger.warning(
                "Team member timed out",
                member=node.name,
                timeout=state.member_timeout,
            )
            if state.error_mode == "fail_all":
                raise
            timeout = state.member_timeout
            error = TimeoutError(f"Member {node.name!r} exceeded {timeout}s deadline")
            error.__cause__ = exc
            return _MemberOutput(agent_name=node.name, exception=error)

        except Exception as exc:
            if state.error_mode == "fail_all":
                raise
            return _MemberOutput(agent_name=node.name, exception=exc)

    return _step


def build_team_graph(
    nodes: list[MessageNode[Any, Any]],
) -> GraphBuilder:
    r"""Build a pydantic-graph that forks to all members and joins results.

    Graph topology::

        start_node
            |
           Fork  <-- broadcasts input to all members
          / | \
        m1 m2 m3  <-- member steps (parallel)
           \\ | /
           Join  <-- reduce_list_append collects outputs
            |
        end_node

    Args:
        nodes: Team members to execute in parallel.

    Returns:
        A :class:`GraphBuilder` ready to be built and run.
    """
    builder = GraphBuilder(
        state_type=_TeamGraphState,
        output_type=list[_MemberOutput],
    )

    # Create a step for each team member.
    # Use a positional suffix to ensure unique node IDs when the same
    # agent appears in multiple steps (e.g. parallel teams with duplicates).
    member_steps = []
    for index, node in enumerate(nodes):
        step_fn = _make_member_step(node)
        step = builder.step(call=step_fn, node_id=f"{node.name}_{index}")
        member_steps.append(step)

    # Join that collects all member outputs into a list
    collect = builder.join(
        reduce_list_append,
        initial_factory=list[_MemberOutput],
        node_id="team_join",
    )

    # Wire: start -> fork -> members -> join -> end
    builder.add(
        builder.edge_from(builder.start_node).to(*member_steps),
        builder.edge_from(*member_steps).to(collect),
        builder.edge_from(collect).to(builder.end_node),
    )

    return builder


async def run_team_graph(
    nodes: list[MessageNode[Any, Any]],
    state: _TeamGraphState,
) -> TeamResponse:
    """Execute a team via pydantic-graph and return a :class:`TeamResponse`.

    Args:
        nodes: Team members to execute.
        state: Shared graph state carrying prompts, kwargs, and tracking data.

    Returns:
        A :class:`TeamResponse` with successful responses and any errors.
    """
    start_time = get_now()
    graph = build_team_graph(nodes).build()
    results: list[_MemberOutput] = await graph.run(state=state)

    responses: list[AgentResponse[Any]] = []
    errors: dict[str, Exception] = {}
    for output in results:
        if output.exception is not None:
            errors[output.agent_name] = output.exception
        elif output.response is not None:
            responses.append(output.response)

    return TeamResponse(
        responses=responses,
        start_time=start_time,
        errors=errors,
        child_session_ids=state.child_session_ids,
    )


# ---------------------------------------------------------------------------
# Sequential (TeamRun) graph execution
# ---------------------------------------------------------------------------

ResultMode = Literal["last", "concat"]


@dataclass
class _TeamRunGraphState:
    """Shared state for sequential (TeamRun) graph execution."""

    prompts: tuple[Any, ...] = field(default_factory=tuple)
    """Input prompts for this execution."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to member ``run()``."""

    connections: list[Talk[Any]] = field(default_factory=list)
    """Talk connections for tracking execution stats."""

    responses: list[AgentResponse[Any]] = field(default_factory=list)
    """Collected responses from completed steps."""


@dataclass(frozen=True, kw_only=True)
class ExtendedTeamTalk(TeamTalk):
    """TeamTalk that also provides error tracking."""

    errors: list[tuple[str, str, datetime]] = field(default_factory=list)

    def clear(self) -> None:
        """Reset all tracking data."""
        super().clear()
        self.errors.clear()

    def add_error(self, agent: str, error: str) -> None:
        """Track errors from AgentResponses."""
        self.errors.append((agent, error, get_now()))


def _make_sequential_step(
    node: MessageNode[Any, Any],
    node_index: int,
) -> Any:
    """Create a pydantic-graph step for a sequential team member.

    Args:
        node: The team member node to wrap.
        node_index: Index of the node in the pipeline (0 = first).

    Returns:
        An async callable compatible with :meth:`GraphBuilder.step`.
    """

    async def _step(
        ctx: StepContext,
    ) -> ChatMessage[Any]:
        state = cast(_TeamRunGraphState, ctx.state)
        start = perf_counter()
        if node_index == 0:
            result = await node.run(*state.prompts, **state.kwargs)
        else:
            result = await node.run_message(ctx.inputs)
        timing = perf_counter() - start
        response = AgentResponse(agent_name=node.name, message=result, timing=timing)
        state.responses.append(response)

        # Update talk stats for the edge leaving this node (if any)
        if node_index < len(state.connections):
            talk = state.connections[node_index]
            if result:
                talk._stats.messages.append(result)

        return result

    return _step


async def run_teamrun_graph(
    nodes: list[MessageNode[Any, Any]],
    prompts: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    connections: list[Talk[Any]] | None = None,
) -> list[AgentResponse[Any]]:
    """Execute a sequential team pipeline via pydantic-graph.

    Builds and runs a chained graph: start -> step1 -> step2 -> ... -> end,
    where each step runs one node with the output of the previous step.

    Args:
        nodes: Team members to execute sequentially.
        prompts: Input prompts for the first node.
        kwargs: Additional keyword arguments for member ``run()``.
        connections: Optional Talk connections for tracking stats.

    Returns:
        List of :class:`AgentResponse` from each node, in execution order.
    """
    from pydantic_graph.id_types import NodeID

    state = _TeamRunGraphState(
        prompts=prompts,
        kwargs=kwargs or {},
        connections=connections or [],
    )

    # Build steps
    steps: list[Any] = []
    for i, node in enumerate(nodes):
        step_fn = _make_sequential_step(node, i)
        step = Step(
            id=NodeID(f"{node.name}_{i}"),
            call=step_fn,
            label=node.description or node.name,
        )
        steps.append(step)

    # Build graph: start -> step1 -> step2 -> ... -> end
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
    except Exception as exc:
        # Unwrap single-exception ExceptionGroups produced by anyio
        # task groups inside agent run_stream().
        if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
            raise exc.exceptions[0] from None
        raise

    return state.responses
