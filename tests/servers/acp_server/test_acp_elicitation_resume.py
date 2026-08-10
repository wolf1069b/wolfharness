"""Tests for ACP elicitation deferred resume: event consumer restart.

Bug 10: When `_handle_elicitation_deferred()` calls `resume_session()`,
the event consumer must be restarted first. The original consumer stopped
after the turn ended (StreamCompleteEvent → consumer loop exit →
_after_consumer_loop cleanup). Without restarting, events from the resumed
agent run are published to EventBus but nobody is listening, so the ACP
client never receives them.

Tests verify:
- `start_event_consumer` is called before `resume_session`
- If `start_event_consumer` fails, `resume_session` is NOT called
- The order matters: consumer must be ready before resume produces events
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wolfharness.agents.events.events import ElicitationDeferredEvent
from wolfharness_server.acp_server.event_converter import ACPEventConverter
from wolfharness_server.acp_server.handler import ACPProtocolHandler
from wolfharness_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from wolfharness import AgentPool


pytestmark = pytest.mark.unit


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Mock ACPSessionManager."""
    return MagicMock(spec=ACPSessionManager)


@pytest.fixture
def mock_event_converter() -> MagicMock:
    """Mock ACPEventConverter."""
    conv = MagicMock(spec=ACPEventConverter)
    conv.convert = AsyncMock()
    conv.display_mode = "compact"
    return conv


@pytest.fixture
def mock_client() -> MagicMock:
    """Mock ACP Client."""
    client = MagicMock()
    client.session_update = AsyncMock()
    return client


@pytest.fixture
def acp_handler(
    minimal_pool: AgentPool,
    mock_session_manager: MagicMock,
    mock_event_converter: MagicMock,
    mock_client: MagicMock,
) -> ACPProtocolHandler:
    """Return an ACPProtocolHandler with real pool context and mocked deps."""
    return ACPProtocolHandler(
        host_context=minimal_pool.get_context(),
        session_manager=mock_session_manager,
        event_converter=mock_event_converter,
        client=mock_client,
        client_capabilities=MagicMock(elicitation=True),
    )


def _make_elicitation_event(
    session_id: str = "test-elicit-session",
    deferred_handle: str = "tc-elicit-001",
) -> ElicitationDeferredEvent:
    """Create an ElicitationDeferredEvent for testing."""
    return ElicitationDeferredEvent(
        deferred_handle=deferred_handle,
        message="Do you agree?",
        requested_schema={"type": "object", "properties": {"q0": {"type": "string"}}},
        mode="form",
        session_id=session_id,
    )


class _FakeElicitationResponse:
    """Fake elicitation response from ACP client."""

    def __init__(self, action: str = "accept", content: dict | None = None) -> None:
        self.action = action
        self.content = content or {"q0": "yes"}


@pytest.mark.anyio
async def test_start_event_consumer_called_before_resume(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """start_event_consumer must be called before resume_session.

    Without this ordering, events from the resumed agent run are published
    to EventBus before any consumer is subscribed, so the ACP client
    never receives them.
    """
    session_id = "test-elicit-session"
    event = _make_elicitation_event(session_id=session_id)

    # Track call order
    call_order: list[str] = []

    async def mock_start_consumer(sid: str) -> None:
        call_order.append(f"start_event_consumer:{sid}")

    async def mock_resume(sid: str, **kwargs: object) -> None:
        call_order.append(f"resume_session:{sid}")

    acp_handler.start_event_consumer = mock_start_consumer
    # Monkeypatch the real SessionPool's resume_session
    assert minimal_pool.session_pool is not None
    minimal_pool.session_pool.resume_session = mock_resume  # type: ignore[assignment]

    # Mock ACPRequests.elicitation_create to return immediately
    with (
        patch.object(
            acp_handler,
            "client",
        ),
        patch(
            "wolfharness_server.acp_server.handler.ACPRequests",
        ) as mock_acp_requests_class,
    ):
        mock_acp_requests = MagicMock()
        mock_acp_requests.elicitation_create = AsyncMock(
            return_value=_FakeElicitationResponse(action="accept"),
        )
        mock_acp_requests_class.return_value = mock_acp_requests

        await acp_handler._handle_elicitation_deferred(session_id, event)

    assert call_order[0] == f"start_event_consumer:{session_id}", (
        f"start_event_consumer must be called first, got: {call_order}"
    )
    assert call_order[1] == f"resume_session:{session_id}", (
        f"resume_session must be called second, got: {call_order}"
    )


@pytest.mark.anyio
async def test_resume_not_called_if_start_consumer_fails(
    acp_handler: ACPProtocolHandler,
    minimal_pool: AgentPool,
) -> None:
    """If start_event_consumer raises, resume_session must NOT be called.

    Starting the consumer is a prerequisite — without it, resume events
    would be lost. If consumer startup fails, we should not proceed with
    resume.
    """
    session_id = "test-elicit-session"
    event = _make_elicitation_event(session_id=session_id)

    resume_called = False

    async def mock_start_consumer(sid: str) -> None:
        raise RuntimeError("Consumer startup failed")

    async def mock_resume(sid: str, **kwargs: object) -> None:
        nonlocal resume_called
        resume_called = True

    acp_handler.start_event_consumer = mock_start_consumer
    assert minimal_pool.session_pool is not None
    minimal_pool.session_pool.resume_session = mock_resume  # type: ignore[assignment]

    with patch(
        "wolfharness_server.acp_server.handler.ACPRequests",
    ) as mock_acp_requests_class:
        mock_acp_requests = MagicMock()
        mock_acp_requests.elicitation_create = AsyncMock(
            return_value=_FakeElicitationResponse(action="accept"),
        )
        mock_acp_requests_class.return_value = mock_acp_requests

        # Should not raise — the handler catches exceptions
        await acp_handler._handle_elicitation_deferred(session_id, event)

    assert not resume_called, "resume_session must NOT be called if start_event_consumer fails"
