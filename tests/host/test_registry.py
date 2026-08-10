"""Unit tests for AgentRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wolfharness.host.registry import AgentRegistry


pytestmark = pytest.mark.unit


def _make_agent(name: str = "test_agent") -> MagicMock:
    """Create a MagicMock standing in for a BaseAgent."""
    mock = MagicMock()
    mock.name = name
    return mock


def test_get_returns_agent_for_existing_name() -> None:
    """Given a registry with an agent, get() returns that agent."""
    agent = _make_agent("alpha")
    registry = AgentRegistry({"alpha": agent})

    result = registry.get("alpha")

    assert result is agent


def test_get_raises_key_error_for_missing_agent() -> None:
    """Given an empty registry, get() raises KeyError."""
    registry = AgentRegistry()

    with pytest.raises(KeyError, match="Agent not found: ghost"):
        registry.get("ghost")


def test_get_or_none_returns_none_for_missing_agent() -> None:
    """Given a registry without the agent, get_or_none() returns None."""
    registry = AgentRegistry()

    assert registry.get_or_none("missing") is None


def test_get_or_none_returns_agent_for_existing() -> None:
    """Given a registry with an agent, get_or_none() returns that agent."""
    agent = _make_agent("beta")
    registry = AgentRegistry({"beta": agent})

    assert registry.get_or_none("beta") is agent


def test_exists_returns_true_for_existing_agent() -> None:
    """Given a registry with an agent, exists() returns True."""
    registry = AgentRegistry({"gamma": _make_agent("gamma")})

    assert registry.exists("gamma") is True


def test_exists_returns_false_for_missing_agent() -> None:
    """Given a registry without the agent, exists() returns False."""
    registry = AgentRegistry()

    assert registry.exists("nope") is False


def test_list_names_returns_sorted_list() -> None:
    """Given unsorted insertion order, list_names() returns sorted names."""
    registry = AgentRegistry({
        "zebra": _make_agent("zebra"),
        "alpha": _make_agent("alpha"),
        "mango": _make_agent("mango"),
    })

    assert registry.list_names() == ["alpha", "mango", "zebra"]


def test_add_adds_agent_to_registry() -> None:
    """Given an empty registry, add() makes the agent retrievable."""
    registry = AgentRegistry()
    agent = _make_agent("delta")

    registry.add("delta", agent)

    assert registry.get("delta") is agent
    assert registry.exists("delta")


def test_len_returns_correct_count() -> None:
    """Given a registry with 2 agents, __len__ returns 2."""
    registry = AgentRegistry({
        "a": _make_agent("a"),
        "b": _make_agent("b"),
    })

    assert len(registry) == 2


def test_contains_works_as_alias_for_exists() -> None:
    """Given a registry, the ``in`` operator matches exists()."""
    registry = AgentRegistry({"echo": _make_agent("echo")})

    assert "echo" in registry
    assert "foxtrot" not in registry


def test_iter_iterates_over_names() -> None:
    """Given a registry, __iter__ yields agent names (keys)."""
    registry = AgentRegistry({
        "x": _make_agent("x"),
        "y": _make_agent("y"),
    })

    result = sorted(registry)

    assert result == ["x", "y"]


def test_empty_registry() -> None:
    """Given an empty registry, all queries reflect emptiness."""
    registry = AgentRegistry()

    assert len(registry) == 0
    assert registry.list_names() == []
    assert registry.exists("anything") is False
    assert list(registry) == []


def test_add_overwrites_existing_agent() -> None:
    """Given a registry with an agent, add() with same name replaces it."""
    original = _make_agent("orig")
    replacement = _make_agent("repl")
    registry = AgentRegistry({"key": original})

    registry.add("key", replacement)

    assert registry.get("key") is replacement
    assert len(registry) == 1


def test_init_with_none_agents() -> None:
    """Given None passed to __init__, registry starts empty."""
    registry = AgentRegistry(None)

    assert len(registry) == 0


def _make_registry_with_agents(*names: str) -> AgentRegistry:
    """Helper: build a registry from agent names."""
    agents: dict[str, MagicMock] = {n: _make_agent(n) for n in names}
    return AgentRegistry(agents)


def test_list_names_empty_returns_empty_list() -> None:
    """Given an empty registry, list_names() returns []."""
    registry = _make_registry_with_agents()

    assert registry.list_names() == []


def test_get_or_none_does_not_raise_on_empty() -> None:
    """Given an empty registry, get_or_none() safely returns None."""
    registry = AgentRegistry()

    assert registry.get_or_none("whatever") is None
