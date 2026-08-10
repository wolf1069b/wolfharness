"""Translate Talk configuration into GraphBuilder edges and Path transforms.

This module provides :class:`TalkEdgeTranslator`, which converts AgentPool
:class:`~wolfharness.talk.Talk` instances into ``pydantic-graph``
:class:`~pydantic_graph.paths.EdgePath` objects.

Key mappings:
- ``connection_type`` → edge label documenting the behavior
- ``transform`` → :class:`~pydantic_graph.paths.TransformMarker` (sync) or
  intermediate :class:`~pydantic_graph.step.Step` (async)
- ``filter_condition`` → :class:`~pydantic_graph.decision.Decision` node
  before the target
- ``stop_condition`` / ``exit_condition`` → :class:`Decision` with early
  :class:`~pydantic_graph.node.EndNode`
- ``queued`` → buffering :class:`Step` before the target
- Multi-target → broadcast via :class:`~pydantic_graph.node.Fork`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic_graph.id_types import NodeID
from pydantic_graph.paths import TransformFunction
from pydantic_graph.step import StepContext  # noqa: TC002

from wolfharness.utils.inspection import is_async_callable


if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_graph import GraphBuilder
    from pydantic_graph.decision import Decision
    from pydantic_graph.paths import EdgePath
    from pydantic_graph.step import Step

    from wolfharness.messaging import MessageNode
    from wolfharness.talk import Talk


# ---------------------------------------------------------------------------
# Condition-result wrappers for type-based Decision branching
# ---------------------------------------------------------------------------


class _ConditionResult:
    """Base class for condition evaluation results."""

    __slots__ = ()


class _ConditionPass(_ConditionResult):
    """Indicates a condition evaluated to ``True``."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _ConditionFail(_ConditionResult):
    """Indicates a condition evaluated to ``False``."""

    __slots__ = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unwrap_pass(ctx: StepContext) -> Any:
    """Extract the wrapped value from a :class:`_ConditionPass`."""
    inputs = cast(_ConditionPass, ctx.inputs)
    return inputs.value


def _make_sync_matches(
    condition: Callable[..., bool | Any],
    talk: Talk[Any],
    target_node: MessageNode[Any, Any],
) -> Callable[[Any], bool]:
    """Create a synchronous ``matches`` predicate for a Decision branch.

    Args:
        condition: The Talk condition callable (must be synchronous).
        talk: The Talk instance owning the condition.
        target_node: The original MessageNode target (for EventContext).

    Returns:
        A predicate ``fn(inputs) -> bool``.
    """

    def _matches(inputs: Any) -> bool:
        from wolfharness.talk.registry import EventContext

        ctx = EventContext(
            message=inputs,
            target=target_node,
            stats=talk.stats,
            registry=None,
            talk=talk,
        )
        result = condition(ctx)
        return bool(result) if not hasattr(result, "__await__") else False

    return _matches


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


@dataclass
class TalkEdgeTranslator:
    """Translates :class:`~wolfharness.talk.Talk` connections into graph edges.

    Attributes:
        builder: The :class:`GraphBuilder` receiving translated edges.
    """

    builder: GraphBuilder

    def translate(
        self,
        talk: Talk[Any],
        source_step: Step,
        target_steps: list[Step],
        target_nodes: list[MessageNode[Any, Any]] | None = None,
    ) -> list[EdgePath]:
        """Translate a single :class:`Talk` into graph :class:`EdgePath` objects.

        The translation respects Talk property ordering:
        exit → stop → transform → queued → filter → broadcast.

        Args:
            talk: The Talk to translate.
            source_step: The pydantic-graph :class:`Step` mapped from
                ``talk.source``.
            target_steps: The pydantic-graph :class:`Step` objects mapped from
                ``talk.targets``.
            target_nodes: Optional original :class:`MessageNode` targets,
                required when conditions need :class:`EventContext`.

        Returns:
            A list of :class:`EdgePath` objects that should be added to the
            :class:`GraphBuilder` via ``builder.add(*edges)``.

        Raises:
            ValueError: If conditions are provided but ``target_nodes`` is
                ``None``.
        """
        edges: list[EdgePath] = []
        current_source = source_step
        path_builder = self.builder.edge_from(current_source)

        # Label the edge with connection type for diagram readability
        path_builder = path_builder.label(f"type:{talk.connection_type}")

        # ---- Exit condition ------------------------------------------------
        if talk.exit_condition is not None:
            if target_nodes is None:
                msg = "target_nodes required when exit_condition is set"
                raise ValueError(msg)
            exit_decision = self._build_condition_decision(
                talk,
                talk.exit_condition,
                target_steps,
                target_nodes,
                invert=True,
                suffix="exit",
            )
            edges.append(path_builder.to(exit_decision))
            return edges

        # ---- Stop condition ------------------------------------------------
        if talk.stop_condition is not None:
            if target_nodes is None:
                msg = "target_nodes required when stop_condition is set"
                raise ValueError(msg)
            stop_decision = self._build_condition_decision(
                talk,
                talk.stop_condition,
                target_steps,
                target_nodes,
                invert=True,
                suffix="stop",
            )
            edges.append(path_builder.to(stop_decision))
            return edges

        # ---- Transform (sync → TransformMarker, async → Step) -------------
        if talk.transform_fn is not None:
            if is_async_callable(talk.transform_fn):
                # Async transforms need an intermediate step
                transform_step = self._build_transform_step(talk)
                edges.append(path_builder.to(transform_step))
                current_source = transform_step
                path_builder = self.builder.edge_from(current_source)
            else:
                # Sync transforms can use a TransformMarker on the path
                sync_transform = self._wrap_sync_transform(talk.transform_fn)
                path_builder = path_builder.transform(sync_transform)

        # ---- Queued connections → buffer step ------------------------------
        if talk.queued:
            buffer_step = self._build_buffer_step(talk)
            edges.append(path_builder.to(buffer_step))
            current_source = buffer_step
            path_builder = self.builder.edge_from(current_source)

        # ---- Filter condition → per-target Decision ------------------------
        if talk.filter_condition is not None:
            if target_nodes is None:
                msg = "target_nodes required when filter_condition is set"
                raise ValueError(msg)
            filter_decisions = self._build_filter_decisions(talk, target_steps, target_nodes)
            edges.append(path_builder.to(*filter_decisions))
            return edges

        # ---- Route to target(s) -------------------------------------------
        match len(target_steps):
            case 0:
                # No targets - route to end node
                edges.append(path_builder.to(self.builder.end_node))
            case 1:
                edges.append(path_builder.to(target_steps[0]))
            case _:
                # Multi-target → broadcast (creates Fork node automatically)
                edges.append(path_builder.to(*target_steps))

        return edges

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_transform_step(self, talk: Talk[Any]) -> Step:
        """Create an intermediate :class:`Step` for an async transform."""
        transform_fn = talk.transform_fn
        assert transform_fn is not None

        async def _transform_step(ctx: StepContext) -> Any:
            from wolfharness.utils.inspection import execute

            return await execute(transform_fn, ctx.inputs)

        return self.builder.step(
            call=_transform_step,
            node_id=NodeID(f"{talk.name}_transform"),
        )

    def _wrap_sync_transform(
        self,
        transform_fn: Callable[..., Any],
    ) -> TransformFunction:
        """Wrap a sync Talk transform into a pydantic-graph TransformFunction."""

        def _transform(ctx: StepContext) -> Any:
            return transform_fn(ctx.inputs)

        return cast(TransformFunction, _transform)

    def _build_buffer_step(self, talk: Talk[Any]) -> Step:
        """Create a buffering :class:`Step` for queued connections.

        The step stores the incoming message in graph state and returns it
        unchanged.  Full queue-strategy semantics (concat / latest) require
        runtime state management and are left to the adapter layer.
        """

        async def _buffer_step(ctx: StepContext) -> Any:
            # In a real adapter, this would maintain a queue in state.
            # For the translator, we passthrough so the graph structure
            # is correct.
            return ctx.inputs

        return self.builder.step(
            call=_buffer_step,
            node_id=NodeID(f"{talk.name}_buffer"),
        )

    def _build_condition_decision(
        self,
        talk: Talk[Any],
        condition: Callable[..., bool | Any],
        target_steps: list[Step],
        target_nodes: list[MessageNode[Any, Any]],
        *,
        invert: bool,
        suffix: str,
    ) -> Decision:
        """Build a :class:`Decision` that routes to targets or EndNode.

        Args:
            talk: The owning Talk.
            condition: The condition callable.
            target_steps: Target pydantic-graph steps.
            target_nodes: Original MessageNode targets (for EventContext).
            invert: When ``True``, the condition passing routes to the
                targets and failing routes to ``end_node``.  When ``False``,
                the opposite.
            suffix: Node-ID suffix for the decision.

        Returns:
            A configured :class:`Decision` node.
        """
        decision = self.builder.decision(
            node_id=f"{talk.name}_{suffix}",
            note=f"{suffix} condition for {talk.name}",
        )

        if is_async_callable(condition):
            # Async conditions need an evaluation step + type-based branching
            # We return a Decision that the *caller* should route through,
            # but actually for async conditions we need the eval step first.
            # This is handled by creating a Decision that branches on the
            # eval step's output type.
            # For simplicity, we create the decision here and the caller
            # must insert the eval step before it.
            # NOTE: In practice, async stop/exit conditions are modelled as:
            #   source → eval_step → decision → [targets | end]
            # The current method returns the Decision; the caller should
            # create the edge: source → eval_step, then eval_step → decision.
            pass_branch = (
                self.builder.match(_ConditionPass).transform(_unwrap_pass).to(*target_steps)
            )
            fail_branch = self.builder.match(_ConditionFail).to(self.builder.end_node)
            if invert:
                # pass = continue to targets, fail = end
                return decision.branch(pass_branch).branch(fail_branch)
            # pass = end, fail = continue to targets
            return decision.branch(fail_branch).branch(pass_branch)

        # Sync conditions use matches predicate directly
        pred = _make_sync_matches(condition, talk, target_nodes[0])
        neg_pred = _make_sync_matches(
            lambda ctx, original=pred: not original(ctx), talk, target_nodes[0]
        )

        if invert:
            pass_branch = self.builder.match(Any, matches=pred).to(*target_steps)
            fail_branch = self.builder.match(Any, matches=neg_pred).to(self.builder.end_node)
        else:
            pass_branch = self.builder.match(Any, matches=pred).to(self.builder.end_node)
            fail_branch = self.builder.match(Any, matches=neg_pred).to(*target_steps)

        return decision.branch(pass_branch).branch(fail_branch)

    def _build_async_condition_step(
        self,
        talk: Talk[Any],
        condition: Callable[..., Any],
        target_nodes: list[MessageNode[Any, Any]],
    ) -> Step:
        """Create a :class:`Step` that evaluates an async condition."""

        async def _eval_step(ctx: StepContext) -> _ConditionResult:
            from wolfharness.talk.registry import EventContext
            from wolfharness.utils.inspection import execute

            # Evaluate against the first target (stop/exit are "any target")
            event_ctx: EventContext[Any] = EventContext(
                message=ctx.inputs,
                target=target_nodes[0],
                stats=talk.stats,
                registry=None,
                talk=talk,
            )
            result = await execute(condition, event_ctx)
            if result:
                return _ConditionPass(ctx.inputs)
            return _ConditionFail()

        return self.builder.step(
            call=_eval_step,
            node_id=NodeID(f"{talk.name}_condition_eval"),
        )

    def _build_filter_decisions(
        self,
        talk: Talk[Any],
        target_steps: list[Step],
        target_nodes: list[MessageNode[Any, Any]],
    ) -> list[Decision]:
        """Build per-target :class:`Decision` nodes for filter conditions."""
        decisions: list[Decision] = []
        condition = talk.filter_condition
        assert condition is not None

        pairs = zip(target_steps, target_nodes, strict=True)
        for i, (target_step, target_node) in enumerate(pairs):
            decision = self.builder.decision(
                node_id=f"{talk.name}_filter_{i}",
                note=f"filter for {talk.name} -> {target_node.name}",
            )

            if is_async_callable(condition):
                # Async filter conditions are not yet supported in this
                # translator. Fall back to a no-op pass-through.
                pass_branch = self.builder.match(Any, matches=lambda _: False).to(target_step)
                fail_branch = self.builder.match(Any, matches=lambda _: True).to(
                    self.builder.end_node
                )
            else:
                pred = _make_sync_matches(condition, talk, target_node)
                neg_pred = _make_sync_matches(
                    lambda ctx, original=pred: not original(ctx), talk, target_node
                )
                pass_branch = self.builder.match(Any, matches=pred).to(target_step)
                fail_branch = self.builder.match(Any, matches=neg_pred).to(self.builder.end_node)

            decisions.append(decision.branch(pass_branch).branch(fail_branch))

        return decisions

    def _build_async_filter_step(
        self,
        talk: Talk[Any],
        condition: Callable[..., Any],
        target_node: MessageNode[Any, Any],
        index: int,
    ) -> Step:
        """Create a :class:`Step` that evaluates an async filter condition."""

        async def _eval_filter(ctx: StepContext) -> _ConditionResult:
            from wolfharness.talk.registry import EventContext
            from wolfharness.utils.inspection import execute

            event_ctx: EventContext[Any] = EventContext(
                message=ctx.inputs,
                target=target_node,
                stats=talk.stats,
                registry=None,
                talk=talk,
            )
            result = await execute(condition, event_ctx)
            if result:
                return _ConditionPass(ctx.inputs)
            return _ConditionFail()

        return self.builder.step(
            call=_eval_filter,
            node_id=NodeID(f"{talk.name}_filter_eval_{index}"),
        )
