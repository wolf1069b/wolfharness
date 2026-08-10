"""Runtime agent registry for pool-less agent lookup.

Provides agent config lookup without pool-level registration.
When the ``eliminate-pool-level-agents`` branch removed pool-level agent
storage, ``SessionController.get_or_create_session_agent()`` and subagent
tools could no longer resolve programmatically-created agents. This registry
bridges that gap by allowing tools to register agent configs at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wolfharness.models.manifest import AnyAgentConfig


class RuntimeAgentRegistry:
    """Registry for programmatically-created agents.

    Provides agent config lookup without pool-level registration.
    Thread-safe via dict access (single-threaded async).
    """

    def __init__(self) -> None:
        self._agents: dict[str, AnyAgentConfig] = {}

    def register(self, name: str, config: AnyAgentConfig) -> None:
        """Register an agent config at runtime.

        Args:
            name: The agent name (key used for lookup).
            config: The agent configuration to register.
        """
        self._agents[name] = config

    def lookup(self, name: str) -> AnyAgentConfig | None:
        """Look up an agent config by name.

        Args:
            name: The agent name to look up.

        Returns:
            The agent config if found, ``None`` otherwise.
        """
        return self._agents.get(name)

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry.

        Args:
            name: The agent name to remove.
        """
        self._agents.pop(name, None)

    def names(self) -> list[str]:
        """Return all registered agent names.

        Returns:
            A list of all registered agent names.
        """
        return list(self._agents.keys())
