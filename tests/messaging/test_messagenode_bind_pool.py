"""Unit tests for MessageNode._bind_pool() internal method."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from wolfharness.messaging import ChatMessage
from wolfharness.messaging.messagenode import MessageNode


if TYPE_CHECKING:
    from wolfharness import AgentPool


class ConcreteMessageNode(MessageNode[Any, Any]):
    """Concrete implementation of MessageNode for testing."""

    async def run(self, *prompts: Any, **kwargs: Any) -> ChatMessage[Any]:
        return ChatMessage(content="test", role="assistant")

    async def get_stats(self) -> None:
        pass

    def run_iter(self, *prompts: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.unit
def test_bind_pool_sets_internal_field(minimal_pool: AgentPool) -> None:
    """_bind_pool() sets _agent_pool without emitting DeprecationWarning."""
    node = ConcreteMessageNode(name="test_node")

    node._bind_pool(minimal_pool)

    assert node._agent_pool is minimal_pool


@pytest.mark.unit
def test_bind_pool_with_none_clears_field(minimal_pool: AgentPool) -> None:
    """_bind_pool(None) clears the _agent_pool field."""
    node = ConcreteMessageNode(name="test_node")
    node._bind_pool(minimal_pool)

    node._bind_pool(None)

    assert node._agent_pool is None


@pytest.mark.unit
def test_bind_pool_does_not_emit_deprecation_warning(minimal_pool: AgentPool) -> None:
    """_bind_pool() must NOT emit DeprecationWarning (unlike the public setter)."""
    import warnings

    node = ConcreteMessageNode(name="test_node")

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        node._bind_pool(minimal_pool)

    deprecation_warnings = [w for w in warning_record if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 0


@pytest.mark.unit
def test_host_context_returns_non_none_after_bind_pool() -> None:
    """After _bind_pool(), host_context returns a non-None HostContext."""
    from wolfharness.delegation.pool import AgentPool

    pool = AgentPool()  # type: ignore[type-arg]
    node = ConcreteMessageNode(name="test_node")

    node._bind_pool(pool)

    assert node.host_context is not None
