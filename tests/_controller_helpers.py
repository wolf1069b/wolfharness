"""Test helpers for SessionController routing without SessionPool.

Provides a ``send_via_controller`` function that mirrors what
``SessionController.receive_request()`` used to do (resolve session +
agent, delegate to ``_route_message``) without the deprecated method.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from wolfharness.lifecycle.types import DeliveryMode


if TYPE_CHECKING:
    from wolfharness.orchestrator.session_controller import SessionController


async def send_via_controller(
    controller: SessionController,
    session_id: str,
    content: str | list[Any],
    *,
    mode: DeliveryMode = DeliveryMode.QUEUE,
    message_id: str | None = None,
    input_provider: Any = None,
    deps: Any = None,
) -> str | None:
    """Send a message via SessionController (test helper).

    Resolves session and agent, then delegates to ``_route_message``.
    This replaces the removed ``receive_request()`` method in tests
    that use ``SessionController`` directly without a ``SessionPool``.

    Args:
        controller: The SessionController instance.
        session_id: Target session.
        content: Message content.
        mode: Delivery mode (STEER or QUEUE).
        message_id: Optional message ID.
        input_provider: Optional input provider.
        deps: Optional dependencies.

    Returns:
        message_id string on success, None on failure.
    """
    session = controller.get_session(session_id)
    if session is None:
        return None
    session.last_active_at = time.monotonic()
    if input_provider is not None:
        session.input_provider = input_provider
    agent = await controller.get_or_create_session_agent(session_id, input_provider=input_provider)
    if agent is None:
        return None
    priority = "asap" if mode is DeliveryMode.STEER else "when_idle"
    return await controller._route_message(
        session,
        agent,
        session_id,
        content,
        priority=priority,
        deps=deps,
        message_id=message_id,
    )
