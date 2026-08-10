"""AG-UI server implementation for wolfharness pool.

This module provides a server that exposes all agents in an AgentPool
via the AG-UI protocol, with each agent accessible at its own route.

Supports all agent types (native, ACP, Claude Code, Codex, AG-UI) through
the BaseAgentAGUIAdapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wolfharness.log import get_logger
from wolfharness_server.agui_server.base_agent_adapter import BaseAgentAGUIAdapter
from wolfharness_server.http_server import HTTPServer
from wolfharness_server.mixins import ProtocolEventConsumerMixin


if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    from wolfharness import AgentPool
    from wolfharness.orchestrator.core import EventBus, EventEnvelope


logger = get_logger(__name__)

DEFAULT_PORT = 8002


class AGUIServer(HTTPServer, ProtocolEventConsumerMixin):
    """AG-UI server for exposing pool agents on separate routes.

    Provides a unified HTTP server that exposes all agents in the pool via
    the AG-UI protocol. Each agent is accessible at `/{agent_name}` route.

    Example:
        ```python
        pool = AgentPool(manifest)

        server = AGUIServer(pool, host="localhost", port=8002)

        async with server:
            async with server.run_context():
                # Agents accessible at:
                # POST http://localhost:8002/agent1
                # POST http://localhost:8002/agent2
                await do_other_work()
        ```
    """

    # AG-UI is stateless HTTP; events are not consumed here.
    _skip_event_processing = True

    def __init__(
        self,
        pool: AgentPool,
        *,
        name: str | None = None,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        raise_exceptions: bool = False,
    ) -> None:
        """Initialize AG-UI server.

        Args:
            pool: AgentPool containing available agents
            name: Optional server name (auto-generated if None)
            host: Host to bind server to
            port: Port to bind server to
            raise_exceptions: Whether to raise exceptions during server start
        """
        super().__init__(pool, name=name, host=host, port=port, raise_exceptions=raise_exceptions)
        ProtocolEventConsumerMixin.__init__(self)

    @property
    def event_bus(self) -> EventBus:
        """Return the EventBus instance to subscribe to."""
        session_pool = self.pool.session_pool
        if session_pool is None:
            raise RuntimeError("SessionPool not available")
        return session_pool.event_bus

    def _get_subscription_scope(self) -> str:
        """Return the EventBus subscription scope.

        Returns:
            The subscription scope string.
        """
        return "session"

    async def _handle_event(self, session_id: str, envelope: EventEnvelope) -> None:
        """Handle a single event from the EventBus.

        AG-UI server is stateless HTTP; events are not consumed here.
        """

    async def _on_spawn_session_start(self, session_id: str, envelope: EventEnvelope) -> None:
        """Start a child event consumer for the newly spawned session.

        Args:
            session_id: The session whose consumer received the event.
            envelope: The event envelope containing the spawn session start event.
        """
        event = envelope.event
        child_sid = getattr(event, "child_session_id", None)
        if child_sid and child_sid != session_id:
            await self.start_event_consumer(child_sid)

    async def get_routes(self) -> list[Route]:
        """Get Starlette routes for AG-UI protocol.

        Returns:
            List of Route objects for each agent plus root listing endpoint
        """
        from starlette.routing import Route

        routes: list[Route] = []

        # Create route for each agent in the pool (all agent types supported)
        for agent_name in self.pool.manifest.agents:

            async def agent_handler(request: Request, agent_name: str = agent_name) -> Response:
                """Handle AG-UI requests for a specific agent."""
                from starlette.responses import JSONResponse

                sp = self.pool.session_pool
                if sp is not None:
                    from wolfharness.utils.identifiers import generate_session_id

                    session_id = generate_session_id()
                    await sp.create_session(session_id, agent_name=agent_name)
                    pool_agent = await sp.sessions.get_or_create_session_agent(
                        session_id, agent_name
                    )
                else:
                    pool_agent = None
                if pool_agent is None:
                    msg = f"Agent {agent_name!r} not found"
                    return JSONResponse({"error": msg}, status_code=404)
                try:
                    # Use BaseAgentAGUIAdapter which works with any agent type
                    return await BaseAgentAGUIAdapter.dispatch_request(request, agent=pool_agent)
                except Exception as e:
                    self.log.exception("Error handling AG-UI request", agent=agent_name)
                    return JSONResponse({"error": str(e)}, status_code=500)

            routes.append(Route(f"/{agent_name}", agent_handler, methods=["POST"]))
            self.log.debug("Registered AG-UI route", agent=agent_name, route=f"/{agent_name}")

        # Add root endpoint that lists available agents
        async def list_agents(request: Request) -> Response:
            """List all available agents."""
            from starlette.responses import JSONResponse

            agent_list = [
                {"name": name, "route": f"/{name}", "model": str(getattr(agent, "model", ""))}
                for name, agent in self.pool.manifest.agents.items()
            ]
            return JSONResponse({"agents": agent_list, "count": len(agent_list)})

        routes.append(Route("/", list_agents, methods=["GET"]))
        self.log.info("Created AG-UI routes", agent_count=len(self.pool.manifest.agents))
        return routes

    def get_agent_url(self, agent_name: str) -> str:
        """Get the endpoint URL for a specific agent."""
        return f"{self.base_url}/{agent_name}"

    def list_agent_routes(self) -> dict[str, str]:
        """List all agent routes.

        Returns:
            Dictionary mapping agent names to their URLs
        """
        return {name: self.get_agent_url(name) for name in self.pool.manifest.agents}
