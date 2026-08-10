"""DelegationService Protocol and AgentNotFoundError exception.

Defines the limited interface that agent tools use to spawn subagents.
The Protocol is implemented by RunLoop in M2 (task group 15), not by
AgentFactory or AgentPool.

The Protocol intentionally exposes only two methods so that tools know
WHAT they can do (spawn a subagent by name), not HOW RunLoop implements
spawning (queue, priority, background task).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class AgentNotFoundError(Exception):
    """Raised when a requested agent is not found within the current scope.

    The error message deliberately does not reveal the existence of
    agents outside the current scope.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent not found: {agent_name}")


@runtime_checkable
class DelegationService(Protocol):
    """Limited interface for subagent spawning.

    ``spawn_subagent()`` and ``get_available_agents()`` are deprecated.
    Use ``ctx.host.session_pool.run_agent()`` and
    ``ctx.agent_registry.list_names()`` instead.

    For child session creation, use ``SessionPool.create_child_session()``
    directly or ``AgentRunContext.create_child_session()`` which also
    handles agent registration and ``SpawnSessionStart`` emission.

    Implemented by RunLoop (M2 task group 15). Tools access this
    through ``ctx.deps.delegation`` on an ``AgentContext`` instance.

    Methods:
        - ``spawn_subagent(name, prompt)``: deprecated, use
          ``ctx.host.session_pool.run_agent()`` instead.
        - ``get_available_agents()``: deprecated, use
          ``ctx.agent_registry.list_names()`` instead.
    """

    def spawn_subagent(
        self,
        name: str,
        prompt: str,
    ) -> AsyncIterator[Any]:
        """Spawn a subagent by name with the given prompt.

        .. deprecated::
            Use ``ctx.host.session_pool.run_agent()`` instead.

        Args:
            name: Name of the agent to spawn.
            prompt: Input prompt for the subagent.

        Yields:
            Stream events or results from the subagent's execution.

        Raises:
            AgentNotFoundError: If the agent is not in the current scope.
        """
        ...

    def get_available_agents(self) -> list[str]:
        """Return names of agents available within the current scope.

        .. deprecated::
            Use ``ctx.agent_registry.list_names()`` instead.

        Only agents authorized for the current RunScope are included.
        Agents from other tenants or configs are excluded.
        """
        ...
