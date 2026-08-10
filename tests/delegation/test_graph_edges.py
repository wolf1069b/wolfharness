"""Tests verifying Talk to GraphBuilder edge translation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

from pydantic_graph import GraphBuilder
from pydantic_graph.decision import Decision
from pydantic_graph.id_types import NodeID
from pydantic_graph.node import EndNode, Fork
from pydantic_graph.paths import BroadcastMarker, DestinationMarker, TransformMarker
from pydantic_graph.step import Step
import pytest


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from wolfharness.messaging import ChatMessage, MessageNode
from wolfharness.talk import Talk
from wolfharness.talk.graph_edges import TalkEdgeTranslator
from wolfharness.utils.time_utils import get_now


pytestmark = pytest.mark.unit


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMessageNode(MessageNode[Any, Any]):
    """Minimal MessageNode for testing."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.agent_pool = None

    @property
    def name(self) -> str:
        return self._name

    async def run(self, prompt: str) -> Any:
        return prompt

    async def run_message(self, message: ChatMessage[Any], **kwargs: Any) -> ChatMessage[Any]:
        return message

    async def get_stats(self) -> Any:
        from wolfharness.talk.stats import MessageStats

        return MessageStats()

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:

        async def _gen() -> AsyncIterator[ChatMessage[Any]]:
            for p in prompts:
                yield ChatMessage(content=p, role="user")

        return _gen()


def make_chat_message(content: str = "hello") -> ChatMessage[Any]:
    return ChatMessage(content=content, role="user", timestamp=get_now())


def count_transform_markers(edges_by_source: dict[NodeID, list[Any]]) -> int:
    """Count TransformMarker instances in all paths."""
    count = 0
    for paths in edges_by_source.values():
        for path in paths:
            for item in path.items:
                if isinstance(item, TransformMarker):
                    count += 1
    return count


def count_decision_nodes(nodes: dict[NodeID, Any]) -> int:
    """Count Decision nodes."""
    return sum(1 for n in nodes.values() if isinstance(n, Decision))


def count_broadcast_markers(edges_by_source: dict[NodeID, list[Any]]) -> int:
    """Count BroadcastMarker instances in all paths."""
    count = 0
    for paths in edges_by_source.values():
        for path in paths:
            for item in path.items:
                if isinstance(item, BroadcastMarker):
                    count += 1
    return count


def path_leads_to_end(path: Any, end_id: NodeID) -> bool:
    """Check if a path ends at the end node."""
    for item in path.items:
        if isinstance(item, DestinationMarker) and item.destination_id == end_id:
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_talk() -> None:
    """Simple Talk: source -> target translates to single edge."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    talk = Talk(
        source=source_node,
        targets=[target_node],
        connection_type="run",
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    source_edges = graph.edges_by_source.get(NodeID("source_step"), [])
    assert len(source_edges) == 1
    assert any(
        isinstance(item, DestinationMarker) and item.destination_id == NodeID("target_step")
        for path in source_edges
        for item in path.items
    )


def test_talk_with_transform() -> None:
    """Talk with sync transform translates to edge with TransformMarker."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    def sync_transform(msg: ChatMessage[Any]) -> ChatMessage[Any]:
        return msg

    talk = Talk(
        source=source_node,
        targets=[target_node],
        transform=sync_transform,
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    assert count_transform_markers(graph.edges_by_source) >= 1


def test_talk_with_filter() -> None:
    """Talk with filter translates to Decision + conditional edge."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    def sync_filter(ctx: Any) -> bool:
        return True

    talk = Talk(
        source=source_node,
        targets=[target_node],
        filter_condition=sync_filter,
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step], target_nodes=[target_node])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    assert count_decision_nodes(graph.nodes) >= 1


def test_talk_with_stop() -> None:
    """Talk with stop_condition translates to Decision with early End."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    def sync_stop(ctx: Any) -> bool:
        return False

    talk = Talk(
        source=source_node,
        targets=[target_node],
        stop_condition=sync_stop,
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step], target_nodes=[target_node])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    assert count_decision_nodes(graph.nodes) >= 1
    has_end_branch = any(
        path_leads_to_end(branch.path, EndNode.id)
        for node in graph.nodes.values()
        if isinstance(node, Decision)
        for branch in node.branches
    )
    assert has_end_branch


def test_multi_target_talk() -> None:
    """Multi-target Talk translates to Fork + edges."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target1 = FakeMessageNode("target1")
    target2 = FakeMessageNode("target2")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step1 = builder.step(lambda ctx: ctx.inputs, node_id="target_step1")
    target_step2 = builder.step(lambda ctx: ctx.inputs, node_id="target_step2")

    talk = Talk(
        source=source_node,
        targets=[target1, target2],
        connection_type="run",
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step1, target_step2])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    broadcast_count = count_broadcast_markers(graph.edges_by_source)
    fork_count = sum(1 for n in graph.nodes.values() if isinstance(n, Fork) and not n.is_map)
    assert broadcast_count >= 1 or fork_count >= 1


def test_queued_talk() -> None:
    """Queued Talk creates a buffer step before the target."""
    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    talk = Talk(
        source=source_node,
        targets=[target_node],
        queued=True,
    )

    translator = TalkEdgeTranslator(builder)
    edges = translator.translate(talk, source_step, [target_step])
    builder.add(*edges)
    graph = builder.build(validate_graph_structure=False)

    buffer_nodes = [
        n for n in graph.nodes.values() if isinstance(n, Step) and "buffer" in str(n.id)
    ]
    assert len(buffer_nodes) >= 1


def test_connection_type_labels() -> None:
    """Connection types are labeled on the path."""
    from pydantic_graph.paths import LabelMarker

    builder = GraphBuilder(output_type=str)
    source_node = FakeMessageNode("source")
    target_node = FakeMessageNode("target")

    source_step = builder.step(lambda ctx: ctx.inputs, node_id="source_step")
    target_step = builder.step(lambda ctx: ctx.inputs, node_id="target_step")

    for conn_type in ("run", "context", "forward"):
        talk = Talk(
            source=source_node,
            targets=[target_node],
            connection_type=conn_type,
        )

        translator = TalkEdgeTranslator(builder)
        edges = translator.translate(talk, source_step, [target_step])
        builder.add(*edges)

    graph = builder.build(validate_graph_structure=False)

    label_count = 0
    for paths in graph.edges_by_source.values():
        for path in paths:
            for item in path.items:
                if isinstance(item, LabelMarker):
                    label_count += 1

    assert label_count >= 3
