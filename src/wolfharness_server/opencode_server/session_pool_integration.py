"""OpenCode server integration with SessionPool orchestration.

Provides :class:`OpenCodeSessionPoolIntegration` which bridges OpenCode server
routes with the SessionPool orchestration layer. This is the canonical integration
point for routing messages through :meth:`SessionPool.receive_request` and
consuming events from the SessionPool's EventBus.

The implementation is split across mixin modules:
- :mod:`opencode_session_routes` — session lifecycle and routing methods
- :mod:`opencode_event_bridge` — event conversion and EventBus consumer
- :mod:`opencode_message_bridge` — message format conversion and tool-part management

This module re-exports all public functions for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wolfharness_server.mixins import ProtocolEventConsumerMixin
from wolfharness_server.opencode_server.opencode_event_bridge import (
    OpenCodeEventBridgeMixin,
)
from wolfharness_server.opencode_server.opencode_message_bridge import (
    OpenCodeMessageBridgeMixin,
    append_message_to_session,
    get_messages_for_session,
    set_messages_for_session,
)
from wolfharness_server.opencode_server.opencode_session_routes import (
    OpenCodeSessionRoutesMixin,
    ensure_session,
    get_session_status,
    set_session_status,
)


if TYPE_CHECKING:
    from wolfharness.agents.events.events import SpawnSessionStart
    from wolfharness.orchestrator.core import SessionPool
    from wolfharness_server.opencode_server.event_adapter import OpenCodeEventAdapter
    from wolfharness_server.opencode_server.event_processor_context import (
        EventProcessorContext,
    )
    from wolfharness_server.opencode_server.state import ServerState


__all__ = [
    "OpenCodeSessionPoolIntegration",
    "append_message_to_session",
    "ensure_session",
    "get_messages_for_session",
    "get_session_status",
    "set_messages_for_session",
    "set_session_status",
]


class OpenCodeSessionPoolIntegration(
    OpenCodeSessionRoutesMixin,
    OpenCodeEventBridgeMixin,
    OpenCodeMessageBridgeMixin,
    ProtocolEventConsumerMixin,
):
    """Integration layer between OpenCode server routes and SessionPool.

    Encapsulates session lifecycle, message routing, event subscription,
    and status synchronization. Protocol handlers should create one instance
    and reuse it across requests.

    Args:
        session_pool: The SessionPool to route through.
        server_state: The OpenCode server state for broadcasting SSE events.
    """

    def __init__(self, session_pool: SessionPool, server_state: ServerState) -> None:
        """Initialize the integration with a SessionPool and ServerState."""
        super().__init__()
        self.session_pool = session_pool
        self.server_state = server_state
        # Per-session state for mixin hooks
        self._contexts: dict[str, EventProcessorContext] = {}
        self._adapters: dict[str, OpenCodeEventAdapter] = {}
        self._message_registered: dict[str, bool] = {}
        self._steer_split_ids: dict[str, set[str]] = {}
        self._child_to_parent: dict[str, str] = {}
        self._child_spawns: dict[str, SpawnSessionStart] = {}
        self._children_of: dict[str, set[str]] = {}
        # Serialized context data for session resume (keyed by session_id).
        # Populated by external orchestrator before start_event_consumer() is
        # called for a resumed session. Consumed (popped) by _before_consumer_loop.
        self._resume_contexts: dict[str, dict[str, Any]] = {}
        # Pending canonical message IDs from REST handlers (D14).
        # Populated by route_message(), consumed by _before_consumer_loop.
        self._pending_message_ids: dict[str, str] = {}
        # Pending message metadata from REST handlers (agent/model propagation).
        # Populated by route_message(), consumed by _before_consumer_loop.
        # Each entry: {message_id, model_id, provider_id} (all optional).
        self._pending_message_metadata: dict[str, dict[str, str | None]] = {}
