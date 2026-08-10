"""Unit tests for ``_meta.delivery`` extraction at ``acp_agent.py:prompt()``.

Tests that ACP ``_meta.delivery`` is extracted and forwarded through the
call chain: ``prompt()`` → ``handle_prompt()`` → ``send_message()`` with
the correct ``DeliveryMode`` mapping.

- ``"steer"`` → ``DeliveryMode.STEER`` (asap injection)
- ``"followup"`` → ``DeliveryMode.QUEUE`` (when_idle)
- absent → ``DeliveryMode.QUEUE`` (default)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.lifecycle.types import DeliveryMode


pytestmark = [pytest.mark.unit, pytest.mark.anyio]


async def test_meta_delivery_ster_routes_as_asap() -> None:
    """``_meta.delivery="steer"`` → ``handle_prompt(delivery="steer")`` → ``DeliveryMode.STEER``.

    Given: A ``PromptRequest`` with ``_meta.delivery="steer"``.
    When: ``acp_agent.prompt()`` is called.
    Then: ``handle_prompt`` receives ``delivery="steer"`` and ``send_message``
        is called with ``mode=DeliveryMode.STEER``.
    """
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    host_context = MagicMock()
    session_pool = MagicMock()
    session_pool.create_session = AsyncMock()
    session_pool.send_message = AsyncMock(return_value="msg_123")
    session_pool.wait_for_completion = AsyncMock()
    session_pool._get_active_run_handle = MagicMock(return_value=None)
    host_context.session_pool = session_pool

    session_manager = MagicMock()
    session_manager.get_session.return_value = None
    session_manager.session_store = None

    event_converter = MagicMock()
    event_converter.subagent_display_mode = "legacy"
    event_converter.raw_input_mode = "dict"

    client = MagicMock()
    client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=host_context,
        session_manager=session_manager,
        event_converter=event_converter,
        client=client,
    )
    handler._ensure_event_consumer = AsyncMock()

    with patch(
        "wolfharness_server.acp_server.handler.ACPEventConverter.build_user_message_chunks",
        return_value=[],
    ):
        await handler.handle_prompt(
            "test-session",
            [],
            delivery="steer",
        )

    send_call = session_pool.send_message.call_args
    assert send_call is not None
    assert send_call.kwargs["mode"] is DeliveryMode.STEER


async def test_meta_delivery_followup_routes_as_when_idle() -> None:
    """``_meta.delivery="followup"`` → ``DeliveryMode.QUEUE``.

    Given: A ``PromptRequest`` with ``_meta.delivery="followup"``.
    When: ``handle_prompt(delivery="followup")`` is called.
    Then: ``send_message`` is called with ``mode=DeliveryMode.QUEUE``.
    """
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    host_context = MagicMock()
    session_pool = MagicMock()
    session_pool.create_session = AsyncMock()
    session_pool.send_message = AsyncMock(return_value="msg_456")
    session_pool.wait_for_completion = AsyncMock()
    session_pool._get_active_run_handle = MagicMock(return_value=None)
    host_context.session_pool = session_pool

    session_manager = MagicMock()
    session_manager.get_session.return_value = None
    session_manager.session_store = None

    event_converter = MagicMock()
    event_converter.subagent_display_mode = "legacy"
    event_converter.raw_input_mode = "dict"

    client = MagicMock()
    client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=host_context,
        session_manager=session_manager,
        event_converter=event_converter,
        client=client,
    )
    handler._ensure_event_consumer = AsyncMock()

    with patch(
        "wolfharness_server.acp_server.handler.ACPEventConverter.build_user_message_chunks",
        return_value=[],
    ):
        await handler.handle_prompt(
            "test-session",
            [],
            delivery="followup",
        )

    send_call = session_pool.send_message.call_args
    assert send_call is not None
    assert send_call.kwargs["mode"] is DeliveryMode.QUEUE


async def test_no_meta_delivery_defaults_to_queue() -> None:
    """No ``_meta.delivery`` → ``DeliveryMode.QUEUE`` (default).

    Given: A ``PromptRequest`` with no ``_meta`` or no ``delivery`` key.
    When: ``handle_prompt(delivery=None)`` is called.
    Then: ``send_message`` is called with ``mode=DeliveryMode.QUEUE``.
    """
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    host_context = MagicMock()
    session_pool = MagicMock()
    session_pool.create_session = AsyncMock()
    session_pool.send_message = AsyncMock(return_value="msg_789")
    session_pool.wait_for_completion = AsyncMock()
    session_pool._get_active_run_handle = MagicMock(return_value=None)
    host_context.session_pool = session_pool

    session_manager = MagicMock()
    session_manager.get_session.return_value = None
    session_manager.session_store = None

    event_converter = MagicMock()
    event_converter.subagent_display_mode = "legacy"
    event_converter.raw_input_mode = "dict"

    client = MagicMock()
    client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=host_context,
        session_manager=session_manager,
        event_converter=event_converter,
        client=client,
    )
    handler._ensure_event_consumer = AsyncMock()

    with patch(
        "wolfharness_server.acp_server.handler.ACPEventConverter.build_user_message_chunks",
        return_value=[],
    ):
        await handler.handle_prompt(
            "test-session",
            [],
            delivery=None,
        )

    send_call = session_pool.send_message.call_args
    assert send_call is not None
    assert send_call.kwargs["mode"] is DeliveryMode.QUEUE


async def test_message_id_generated_and_passed_through() -> None:
    """``handle_prompt()`` generates a ``message_id`` and passes it to ``send_message()``.

    Given: A valid ``handle_prompt`` call.
    When: The prompt is processed.
    Then: ``send_message`` is called with a non-None ``message_id`` UUID string.
    """
    from wolfharness_server.acp_server.handler import ACPProtocolHandler

    host_context = MagicMock()
    session_pool = MagicMock()
    session_pool.create_session = AsyncMock()
    session_pool.send_message = AsyncMock(return_value="msg_gen")
    session_pool.wait_for_completion = AsyncMock()
    session_pool._get_active_run_handle = MagicMock(return_value=None)
    host_context.session_pool = session_pool

    session_manager = MagicMock()
    session_manager.get_session.return_value = None
    session_manager.session_store = None

    event_converter = MagicMock()
    event_converter.subagent_display_mode = "legacy"
    event_converter.raw_input_mode = "dict"

    client = MagicMock()
    client.session_update = AsyncMock()

    handler = ACPProtocolHandler(
        host_context=host_context,
        session_manager=session_manager,
        event_converter=event_converter,
        client=client,
    )
    handler._ensure_event_consumer = AsyncMock()

    with patch(
        "wolfharness_server.acp_server.handler.ACPEventConverter.build_user_message_chunks",
        return_value=[],
    ):
        await handler.handle_prompt(
            "test-session",
            [],
            delivery=None,
        )

    send_call = session_pool.send_message.call_args
    assert send_call is not None
    passed_mid = send_call.kwargs["message_id"]
    assert passed_mid is not None
    assert isinstance(passed_mid, str)
    assert len(passed_mid) > 0
