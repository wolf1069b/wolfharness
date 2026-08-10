"""RunLoopDelegationService — concrete DelegationService for RunLoop.

Implements the ``DelegationService`` Protocol by delegating subagent
spawning to the ``AgentPool``'s session infrastructure. This is the
runtime bridge between ``SubagentCapability`` (which calls
``ctx.deps.delegation.spawn_subagent()``) and the actual agent spawning
machinery in ``SessionController``.

.. deprecated::
    Both ``spawn_subagent()`` and ``get_available_agents()`` emit
    ``DeprecationWarning``. Use ``ctx.host.session_pool.run_agent()``
    and ``ctx.agent_registry.list_names()`` instead.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
import warnings

from wolfharness.observability.spans import safe_span


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wolfharness.host.context import HostContext
    from wolfharness.host.registry import AgentRegistry


class RunLoopDelegationService:
    """Concrete ``DelegationService`` backed by the AgentPool registry.

    ``spawn_subagent()`` and ``get_available_agents()`` are deprecated.

    Constructed by ``RunHandle`` at turn start using the agent's
    ``HostContext`` and the compiled ``AgentRegistry``. Provides
    subagent spawning by creating a new session via the pool's
    ``SessionPool``.

    Attributes:
        _registry: Read-only registry of available agents.
        _host: Host context for infrastructure access.
        _session_id: Current session ID for parent-child linking.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        host: HostContext,
        session_id: str,
    ) -> None:
        """Initialize the delegation service.

        Args:
            registry: Read-only registry of compiled agents.
            host: Host context with infrastructure handles.
            session_id: Current session ID for parent-child linking.
        """
        self._registry = registry
        self._host = host
        self._session_id = session_id

    async def spawn_subagent(
        self,
        name: str,
        prompt: str,
    ) -> AsyncIterator[Any]:
        """Spawn a subagent by name with the given prompt.

        .. deprecated::
            Use ``ctx.host.session_pool.run_agent()`` instead.

        Delegates to the pool's ``SessionController`` to create a child
        session, run the named agent, and stream events back.

        Args:
            name: Name of the agent to spawn.
            prompt: Input prompt for the subagent.

        Yields:
            Stream events from the subagent's execution.

        Raises:
            AgentNotFoundError: If the agent is not in the registry.
        """
        warnings.warn(
            "RunLoopDelegationService.spawn_subagent() is deprecated. "
            "Use ctx.host.session_pool.run_agent() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from wolfharness.capabilities.delegation import AgentNotFoundError

        if not self._registry.exists(name):
            raise AgentNotFoundError(name)

        session_pool = self._host.session_pool
        if session_pool is None:
            msg = "SessionPool is not available for subagent spawning"
            raise RuntimeError(msg)

        child_session_id = f"{self._session_id}::child::{name}"

        from wolfharness.agents.events import RunErrorEvent, StreamCompleteEvent

        with safe_span(
            "delegation.subagent",
            parent_session_id=self._session_id,
            child_agent_name=name,
        ):
            message_id = await session_pool.send_message(
                session_id=child_session_id,
                content=prompt,
            )
            if message_id is None:
                return

            # send_message() already delivered the prompt and started a
            # background _consume_run task that calls run_handle.start().
            # We subscribe to the EventBus to stream events from the child
            # session instead of calling start("") a second time (which
            # would race with the background task and corrupt state).
            event_bus = session_pool.event_bus
            if event_bus is None:
                return
            bus_queue = await event_bus.subscribe(child_session_id, scope="session")
            try:
                async with asyncio.timeout(300):
                    while True:
                        envelope = await bus_queue.get()
                        event = envelope.event
                        yield event
                        if isinstance(event, StreamCompleteEvent | RunErrorEvent):
                            break
            except TimeoutError:
                from wolfharness.agents.events import RunErrorEvent

                yield RunErrorEvent(
                    message=f"Subagent '{name}' timed out after 300s",
                    run_id="",
                    agent_name=name,
                )
            finally:
                await event_bus.unsubscribe(child_session_id, bus_queue)

    def get_available_agents(self) -> list[str]:
        """Return names of agents available within the current scope.

        .. deprecated::
            Use ``ctx.agent_registry.list_names()`` instead.

        Returns:
            Sorted list of agent names in the registry.
        """
        warnings.warn(
            "RunLoopDelegationService.get_available_agents() is deprecated. "
            "Use ctx.agent_registry.list_names() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._registry.list_names()
