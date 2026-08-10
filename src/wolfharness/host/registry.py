"""Typed wrapper around a dict of agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Iterator

    from wolfharness.agents.base_agent import BaseAgent


class AgentRegistry:
    """Typed wrapper around a dict of agents.

    Provides a thin abstraction over ``dict[str, BaseAgent[Any, Any]]`` with
    explicit error messages and convenience query methods.
    """

    def __init__(self, agents: dict[str, BaseAgent[Any, Any]] | None = None) -> None:
        self._agents: dict[str, BaseAgent[Any, Any]] = agents or {}

    def get(self, name: str) -> BaseAgent[Any, Any]:
        """Get agent by name.

        Raises:
            KeyError: If agent is not found.
        """
        if name not in self._agents:
            msg = f"Agent not found: {name}"
            raise KeyError(msg)
        return self._agents[name]

    def get_or_none(self, name: str) -> BaseAgent[Any, Any] | None:
        """Get agent by name, returning None if not found."""
        return self._agents.get(name)

    def list_names(self) -> list[str]:
        """Return sorted list of agent names."""
        return sorted(self._agents.keys())

    def exists(self, name: str) -> bool:
        """Check if agent exists."""
        return name in self._agents

    def add(self, name: str, agent: BaseAgent[Any, Any]) -> None:
        """Add agent to registry."""
        self._agents[name] = agent

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __iter__(self) -> Iterator[str]:
        return iter(self._agents.keys())
