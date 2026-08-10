"""Tests for summarize_session endpoint with SessionPool migration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_summarize_uses_session_pool(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
    mock_pool: Mock,
):
    """Summarize always uses SessionPool.run_stream (no feature flag branching)."""
    from pydantic_ai import RequestUsage, TextPart, TextPartDelta

    from wolfharness.agents.events import PartDeltaEvent, PartStartEvent, StreamCompleteEvent
    from wolfharness.messaging.messages import ChatMessage
    from wolfharness_server.opencode_server.models import (
        AssistantMessage,
        MessagePath,
        MessageTime,
        MessageWithParts,
        TextPart as OpenCodeTextPart,
    )

    # Create session and add a message
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Pre-populate messages so summarize doesn't 400
    user_msg = AssistantMessage(
        id="m1",
        session_id=session_id,
        parent_id="",
        model_id="default",
        provider_id="wolfharness",
        mode="ask",
        agent="test-agent",
        path=MessagePath(cwd=server_state.working_dir, root=server_state.working_dir),
        time=MessageTime(created=0),
    )
    existing_messages = [
        MessageWithParts(
            info=user_msg,
            parts=[OpenCodeTextPart(id="p1", message_id="m1", session_id=session_id, text="hello")],
        )
    ]

    # Track whether session_pool.run_stream was called
    run_stream_called = False

    async def mock_run_stream(*args: object, **kwargs: object):
        nonlocal run_stream_called
        run_stream_called = True
        yield PartStartEvent(index=0, part=TextPart(content="Summary"))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" text"))
        yield StreamCompleteEvent(
            message=ChatMessage(
                content="Summary text",
                role="assistant",
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        )

    mock_pool.session_pool.run_stream = mock_run_stream

    # Mock compact_conversation and get_messages_for_session
    with (
        patch(
            "wolfharness.messaging.compaction.compact_conversation",
            new=AsyncMock(),
        ),
        patch(
            "wolfharness_server.opencode_server.routes.session_routes.get_messages_for_session",
            return_value=existing_messages,
        ),
    ):
        response = await async_client.post(f"/session/{session_id}/summarize")

    assert response.status_code == 200
    assert run_stream_called is True

    result = response.json()
    assert result is True


async def test_summarize_routes_through_session_pool(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
    mock_pool: Mock,
):
    """SessionPool is the default execution path for summarization."""
    from pydantic_ai import RequestUsage, TextPart, TextPartDelta

    from wolfharness.agents.events import PartDeltaEvent, PartStartEvent, StreamCompleteEvent
    from wolfharness.messaging.messages import ChatMessage
    from wolfharness_server.opencode_server.models import (
        AssistantMessage,
        MessagePath,
        MessageTime,
        MessageWithParts,
        TextPart as OpenCodeTextPart,
    )

    # Create session and add a message
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Pre-populate messages
    user_msg = AssistantMessage(
        id="m1",
        session_id=session_id,
        parent_id="",
        model_id="default",
        provider_id="wolfharness",
        mode="ask",
        agent="test-agent",
        path=MessagePath(cwd=server_state.working_dir, root=server_state.working_dir),
        time=MessageTime(created=0),
    )
    existing_messages = [
        MessageWithParts(
            info=user_msg,
            parts=[OpenCodeTextPart(id="p1", message_id="m1", session_id=session_id, text="hello")],
        )
    ]

    # Mock session_pool.run_stream
    mock_pool.session_pool.run_stream = AsyncMock(
        return_value=[
            PartStartEvent(index=0, part=TextPart(content="SessionPool summary")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" done")),
            StreamCompleteEvent(
                message=ChatMessage(
                    content="SessionPool summary done",
                    role="assistant",
                    usage=RequestUsage(input_tokens=5, output_tokens=3),
                )
            ),
        ]
    )

    # Mock compact_conversation and get_messages_for_session
    with (
        patch(
            "wolfharness.messaging.compaction.compact_conversation",
            new=AsyncMock(),
        ),
        patch(
            "wolfharness_server.opencode_server.routes.session_routes.get_messages_for_session",
            return_value=existing_messages,
        ),
    ):
        response = await async_client.post(f"/session/{session_id}/summarize")

    assert response.status_code == 200

    # Verify session_pool.run_stream was called
    assert mock_pool.session_pool.run_stream.call_count == 1

    result = response.json()
    assert result is True
