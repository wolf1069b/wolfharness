"""Tests for init_session endpoint with SessionPool migration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest


if TYPE_CHECKING:
    from httpx import AsyncClient

    from wolfharness_server.opencode_server.state import ServerState


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_init_session_routes_through_session_pool_with_correct_args(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
    mock_pool: Mock,
):
    """SessionPool.receive_request receives correct session_id and prompt."""
    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    # Track receive_request calls and capture arguments
    receive_request_called = False
    captured_args: tuple[object, ...] = ()
    captured_kwargs: dict[str, object] = {}

    async def mock_receive_request(*args: object, **kwargs: object) -> Mock:
        nonlocal receive_request_called, captured_args, captured_kwargs
        receive_request_called = True
        captured_args = args
        captured_kwargs = kwargs
        run_handle_mock = Mock()
        run_handle_mock.run_id = "test-run-id"
        return run_handle_mock

    mock_pool.session_pool.send_message = AsyncMock(side_effect=mock_receive_request)

    response = await async_client.post(f"/session/{session_id}/init")

    assert response.status_code == 200
    assert response.json() is True
    assert receive_request_called is True

    # Verify correct session_id and prompt were passed
    assert captured_args[0] == session_id
    assert isinstance(captured_args[1], str)
    assert "Please analyze this codebase" in captured_args[1]


async def test_init_session_routes_through_session_pool(
    async_client: AsyncClient,
    server_state: ServerState,
    mock_agent: Mock,
    mock_pool: Mock,
):
    """SessionPool is the default execution path for init."""
    # Create session
    response = await async_client.post("/session", json={"title": "Test Session"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    response = await async_client.post(f"/session/{session_id}/init")

    assert response.status_code == 200
    assert response.json() is True

    # Verify session_pool.receive_request was called
    assert hasattr(mock_pool.session_pool.send_message, "call_count")
    assert mock_pool.session_pool.send_message.call_count == 1
