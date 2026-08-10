"""Signal-emitting wrapper for pydantic-graph GraphRun.

This module provides :class:`SignalEmittingGraphRun`, a thin wrapper around
pydantic-graph's builder-based ``GraphRun`` that emits AgentPool's existing
``anyenv.Signal`` events at step boundaries.  It enables zero-change migration
for downstream consumers (ACP, OpenCode, AG-UI) that already subscribe to
``MessageNode.message_received``, ``MessageNode.message_sent``,
``Talk.connection_processed`` and ``Talk.message_forwarded``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

import logfire
from pydantic_graph.graph_builder import EndMarker, GraphRun, GraphTask

from wolfharness.log import get_logger
from wolfharness.messaging import ChatMessage


if TYPE_CHECKING:
    from pydantic_graph.id_types import NodeID

    from wolfharness.messaging.messagenode import MessageNode
    from wolfharness.talk import Talk

logger = get_logger(__name__)

StateT = TypeVar("StateT")
DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")


class SignalEmittingGraphRun[StateT, DepsT, OutputT]:
    """Wraps a pydantic-graph ``GraphRun`` to emit ``anyenv.Signal`` events.

    The adapter intercepts ``GraphRun`` iteration to emit the following signals
    without modifying existing subscriber code:

    - ``MessageNode.message_received`` - when a ``GraphTask`` is yielded,
      signalling that the corresponding step is about to execute.
    - ``MessageNode.message_sent`` - on the *next* yield, signalling that the
      previously yielded tasks have completed.
    - ``Talk.connection_processed`` - when an edge traversal is detected
      (previous tasks produced new destination tasks).
    - ``Talk.message_forwarded`` - alongside ``connection_processed`` when a
      mapped talk exists for the traversed edge.

    The wrapper preserves the exact ``GraphRun`` async-iteration protocol, so
    consumers may ``async for`` over it exactly as they would a raw
    ``GraphRun``.
    """

    def __init__(
        self,
        graph_run: GraphRun,
        node_mapping: dict[NodeID, MessageNode[Any, Any]],
        talk_mapping: dict[tuple[NodeID, NodeID], Talk[Any]] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the signal-emitting wrapper.

        Args:
            graph_run: The pydantic-graph ``GraphRun`` to wrap.
            node_mapping: Maps ``NodeID`` to the corresponding
                :class:`MessageNode` instance.  Used to resolve which
                ``MessageNode`` signals to emit.
            talk_mapping: Maps ``(source_node_id, destination_node_id)``
                tuples to :class:`Talk` instances.  Used to emit
                ``connection_processed`` and ``message_forwarded`` signals
                during edge traversal.
            session_id: Optional session ID injected into reconstructed
                :class:`ChatMessage` payloads.
        """
        self._graph_run = graph_run
        self._node_mapping = node_mapping
        self._talk_mapping = talk_mapping or {}
        self._session_id = session_id
        self._previous_tasks: list[GraphTask] = []
        self._completed = False

    def __aiter__(self) -> SignalEmittingGraphRun[StateT, DepsT, OutputT]:
        return self

    @logfire.instrument("graph.signal.next")
    async def __anext__(self) -> EndMarker | Sequence[GraphTask]:
        """Advance the graph run and emit signals at step boundaries.

        Drives the underlying ``GraphRun`` forward (which executes pending
        tasks), then emits ``message_sent`` for tasks that completed during
        that advance, and ``message_received`` for newly discovered tasks.

        Returns:
            The next result from the wrapped ``GraphRun``.

        Raises:
            StopAsyncIteration: When the graph run has completed.
            Exception: Re-raised from an internal ``ErrorMarker``.
        """
        # 1. Drive the underlying GraphRun forward.
        #    This executes any tasks that were scheduled in the previous
        #    yield, populating state.result before we emit message_sent.
        try:
            result = await self._graph_run.__anext__()
        except StopAsyncIteration:
            # Previous tasks have now completed — emit message_sent.
            if self._previous_tasks:
                await self._emit_tasks_completed(self._previous_tasks)
            self._previous_tasks = []
            self._completed = True
            raise

        # 2. Previous tasks have now completed — emit message_sent.
        if self._previous_tasks:
            await self._emit_tasks_completed(self._previous_tasks)

        # 3. Detect edge traversals: previous tasks produced this result.
        if self._previous_tasks and isinstance(result, Sequence):
            await self._emit_edge_traversals(self._previous_tasks, list(result))

        # 4. Track new tasks and emit message_received for them.
        if isinstance(result, Sequence):
            self._previous_tasks = list(result)
            await self._emit_tasks_received(result)
        else:
            self._previous_tasks = []
            if isinstance(result, EndMarker):
                self._completed = True

        return result

    async def _emit_tasks_received(self, tasks: Sequence[GraphTask]) -> None:
        """Emit ``message_received`` for each task about to run."""
        for task in tasks:
            node = self._node_mapping.get(task.node_id)
            if node is None:
                continue
            msg = self._task_to_chat_message(task, role="user")
            try:
                await node.message_received.emit(msg)
            except Exception:
                logger.exception("Error emitting message_received for node %s", task.node_id)

    async def _emit_tasks_completed(self, tasks: Sequence[GraphTask]) -> None:
        """Emit ``message_sent`` for each task that has finished.

        When the graph state is an :class:`AgentPoolState` with a populated
        ``result``, that result is used as the sent message — it carries the
        actual output of the node execution rather than the (often ``None``)
        task inputs.
        """
        from wolfharness.messaging.graph_adapter import AgentPoolState

        for task in tasks:
            node = self._node_mapping.get(task.node_id)
            if node is None:
                continue
            state = self._graph_run.state
            if isinstance(state, AgentPoolState):
                state_result = cast(AgentPoolState, state).result
                if state_result is not None:
                    msg = state_result
                else:
                    msg = self._task_to_chat_message(task, role="assistant")
            else:
                msg = self._task_to_chat_message(task, role="assistant")
            try:
                await node.message_sent.emit(msg)
            except Exception:
                logger.exception("Error emitting message_sent for node %s", task.node_id)

    async def _emit_edge_traversals(
        self,
        source_tasks: Sequence[GraphTask],
        destination_tasks: Sequence[GraphTask],
    ) -> None:
        """Emit ``connection_processed`` and ``message_forwarded`` for edges.

        For each unique source node that produced the destination tasks,
        finds matching talks and emits routing signals.
        """
        source_node_ids = {t.node_id for t in source_tasks}
        dest_node_ids = {t.node_id for t in destination_tasks}

        for (src_id, dst_id), talk in self._talk_mapping.items():
            if src_id not in source_node_ids or dst_id not in dest_node_ids:
                continue

            source_node = self._node_mapping.get(src_id)
            dest_node = self._node_mapping.get(dst_id)
            if source_node is None or dest_node is None:
                continue

            source_task = next(t for t in source_tasks if t.node_id == src_id)
            msg = self._task_to_chat_message(source_task)

            try:
                await talk.connection_processed.emit(
                    talk.ConnectionProcessed(
                        message=msg,
                        source=source_node,
                        targets=[dest_node],
                        queued=False,
                        connection_type="run",
                    )
                )
            except Exception:
                logger.exception(
                    "Error emitting connection_processed for %s -> %s",
                    src_id,
                    dst_id,
                )

            try:
                await talk.message_forwarded.emit(msg)
            except Exception:
                logger.exception(
                    "Error emitting message_forwarded for %s -> %s",
                    src_id,
                    dst_id,
                )

    def _task_to_chat_message(
        self,
        task: GraphTask,
        *,
        role: str = "user",
    ) -> ChatMessage[Any]:
        """Convert a ``GraphTask`` into a :class:`ChatMessage` for signals.

        Args:
            task: The graph task to convert.
            role: Role for the message ("user" for received, "assistant"
                for sent).

        Returns:
            A :class:`ChatMessage` wrapping the task's inputs.
        """
        content = task.inputs
        if not isinstance(content, str):
            content = str(content)
        return ChatMessage(
            content=content,
            role=role,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            session_id=self._session_id,
            metadata={
                "node_id": task.node_id,
                "task_id": task.task_id,
            },
        )

    @property
    def graph_run(self) -> GraphRun:
        """Access the underlying ``GraphRun`` instance."""
        return self._graph_run

    @property
    def is_completed(self) -> bool:
        """Whether the graph run has reached an ``EndMarker``."""
        return self._completed
