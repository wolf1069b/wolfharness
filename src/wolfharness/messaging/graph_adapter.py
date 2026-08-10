"""Graph adapter for wrapping MessageNode as a pydantic-graph Step."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import logfire
from pydantic_graph import GraphBuilder, Step, StepContext
from pydantic_graph.id_types import NodeID


if TYPE_CHECKING:
    from pydantic_graph.graph_builder import Graph

    from wolfharness.agents.events import RichAgentStreamEvent
    from wolfharness.messaging import ChatMessage


@dataclass
class AgentPoolState:
    """Shared state passed through pydantic-graph execution.

    Holds the input prompts, node reference, and a conduit for
    streaming events back to the caller.
    """

    node: Any
    """The MessageNode being executed."""

    prompts: tuple[Any, ...] = field(default_factory=tuple)
    """Input prompts for this execution."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to run()."""

    event_queue: asyncio.Queue[RichAgentStreamEvent[Any]] = field(default_factory=asyncio.Queue)
    """Queue for streaming events from the Step back to run_stream()."""

    result: ChatMessage[Any] | None = None
    """Final result populated by the Step upon completion."""


class MessageNodeStep:
    """Wraps a MessageNode as a pydantic-graph Step.

    The Step delegates execution back to the wrapped node's
    :meth:`_execute_node` method, preserving signal emission and
    existing behavior. This adapter is intended for composing
    MessageNodes into larger pydantic-graph workflows.
    """

    def __init__(self, node: Any) -> None:
        """Initialize the step wrapper.

        Args:
            node: The MessageNode to wrap.
        """
        self.node = node

    @logfire.instrument("graph.step.execute")
    async def _execute(self, ctx: StepContext) -> Any:
        """Step function that runs the wrapped node.

        Signal emission (``message_received`` / ``message_sent``) is handled
        by :class:`SignalEmittingGraphRun` which wraps the graph run at the
        ``MessageNode.run()`` / ``MessageNode.run_stream()`` level.

        Args:
            ctx: pydantic-graph StepContext containing state, deps, and inputs.

        Returns:
            The ChatMessage result from the node.
        """
        state = cast(AgentPoolState, ctx.state)
        node = state.node

        # Delegate to the node's core execution logic, injecting state
        # under a private key so _execute_node can access the event queue
        merged_kwargs = {**state.kwargs, "_state": state}
        result = await node._execute_node(*state.prompts, **merged_kwargs)

        state.result = result
        return result

    def as_step(self) -> Step:
        """Return the pydantic-graph Step for this node.

        Returns:
            A Step configured with this node's execution logic.
        """
        return Step(
            id=NodeID(self.node.name),
            call=self._execute,
            label=self.node.description or self.node.name,
        )

    def build_single_node_graph(self) -> Graph:
        """Build a single-node graph containing only this node's Step.

        The graph has start_node -> this node's Step -> end_node.

        Returns:
            An immutable Graph ready for execution.
        """
        builder = GraphBuilder(
            state_type=AgentPoolState,
            output_type=Any,
        )
        step = self.as_step()
        builder.add_edge(builder.start_node, step)
        builder.add_edge(step, builder.end_node)
        return builder.build()
