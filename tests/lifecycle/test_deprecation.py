"""Tests for agent_pool deprecation and host_context migration (M2).

Verifies that:
1. DeprecationWarning is emitted on agent_pool property access.
2. Warning message mentions HostContext.
3. agent_pool still returns the pool reference (compatibility shim).
4. agent_pool returns None when no pool is set (no crash).
5. Migrated code using host_context produces no DeprecationWarning.
"""

from __future__ import annotations

from typing import Any
import warnings

import pytest

from wolfharness.messaging.messagenode import MessageNode


class _TestNode(MessageNode[Any, Any]):
    """Minimal concrete MessageNode for testing deprecation behavior."""

    async def get_stats(self) -> Any:  # type: ignore[override]
        return None

    def run_iter(self, *prompts: Any, **kwargs: Any) -> Any:
        return None


@pytest.mark.unit
def test_agent_pool_access_emits_deprecation_warning():
    """Accessing agent_pool property should emit DeprecationWarning."""
    node = _TestNode(name="test_node")
    node._agent_pool = None  # Explicitly set to None
    with pytest.warns(DeprecationWarning, match="agent_pool is deprecated"):
        _ = node.agent_pool


@pytest.mark.unit
def test_agent_pool_warning_mentions_host_context():
    """The deprecation warning message should mention HostContext."""
    node = _TestNode(name="test_node")
    with pytest.warns(DeprecationWarning, match="HostContext"):
        _ = node.agent_pool


@pytest.mark.unit
def test_agent_pool_returns_none_when_no_pool():
    """agent_pool should return None when no pool is set (no crash)."""
    node = _TestNode(name="test_node")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = node.agent_pool
    assert result is None


@pytest.mark.unit
def test_agent_pool_returns_pool_when_set():
    """agent_pool should still return the pool reference (compatibility shim)."""
    sentinel = object()  # Stand-in for a pool object
    node = _TestNode(name="test_node")
    node._agent_pool = sentinel  # type: ignore[assignment]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = node.agent_pool
    assert result is sentinel


@pytest.mark.unit
def test_agent_pool_setter_does_not_warn():
    """Setting agent_pool should NOT emit a deprecation warning."""
    node = _TestNode(name="test_node")
    sentinel = object()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        node.agent_pool = sentinel  # type: ignore[assignment]
    # If we reach here, no warning was raised
    assert node._agent_pool is sentinel


@pytest.mark.unit
def test_host_context_no_warning_when_pool_none():
    """host_context should not emit DeprecationWarning when pool is None."""
    node = _TestNode(name="test_node")
    node._agent_pool = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = node.host_context
    assert result is None


@pytest.mark.unit
def test_storage_property_no_warning_when_no_pool():
    """Storage property should not emit DeprecationWarning (uses _agent_pool)."""
    node = _TestNode(name="test_node")
    node._agent_pool = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = node.storage
    assert result is None


@pytest.mark.unit
def test_migrated_code_produces_no_deprecation_warning():
    """Code using host_context should not produce DeprecationWarning.

    This simulates what migrated call sites do: access host_context
    instead of agent_pool.
    """
    node = _TestNode(name="test_node")
    node._agent_pool = None

    # Access host_context — should NOT raise DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = node.host_context

    # Access storage — should NOT raise DeprecationWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = node.storage
