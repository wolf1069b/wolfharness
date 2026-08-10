"""Regression tests for async prompt handling in OpenCode server."""

from __future__ import annotations

import asyncio

import pytest

from wolfharness_server.opencode_server.models import MessageRequest, TextPartInput


pytestmark = pytest.mark.integration


class TestPromptAsync:
    """Tests for `/prompt_async` session serialization via SessionPool."""

    @pytest.mark.asyncio
    async def test_prompt_async_returns_204_and_routes_via_session_pool(
        self,
        async_client,
        server_state,
    ) -> None:
        """The async prompt endpoint returns 204 and routes through SessionPool."""
        response = await async_client.post("/session", json={"title": "Async Lock"})
        session_id = response.json()["id"]

        # Spy on SessionPool.receive_request
        pool = server_state.pool
        original_receive_request = pool.session_pool.send_message
        receive_calls: list[dict] = []

        async def spy_receive_request(*args, **kwargs):
            receive_calls.append(kwargs)
            return await original_receive_request(*args, **kwargs)

        pool.session_pool.send_message = spy_receive_request

        request = MessageRequest(
            parts=[TextPartInput(text="first")],
            agent="default",
            message_id="msg-1",
        )
        response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 204
        assert len(receive_calls) == 1
        assert receive_calls[0]["session_id"] == session_id

        second_request = MessageRequest(
            parts=[TextPartInput(text="second")],
            agent="default",
            message_id="msg-2",
        )
        response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=second_request.model_dump(mode="json"),
        )
        assert response.status_code == 204
        assert len(receive_calls) == 2

    @pytest.mark.asyncio
    async def test_prompt_async_multiple_requests_accepted(
        self,
        async_client,
        server_state,
    ) -> None:
        """Multiple async prompts to the same session are accepted without error."""
        response = await async_client.post("/session", json={"title": "Async Queue"})
        session_id = response.json()["id"]

        first_request = MessageRequest(
            parts=[TextPartInput(text="first")],
            agent="default",
            message_id="msg-1",
        )
        second_request = MessageRequest(
            parts=[TextPartInput(text="second")],
            agent="default",
            message_id="msg-2",
        )

        first_response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=first_request.model_dump(mode="json"),
        )
        second_response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=second_request.model_dump(mode="json"),
        )

        assert first_response.status_code == 204
        assert second_response.status_code == 204

    @pytest.mark.asyncio
    async def test_prompt_async_nonexistent_session_returns_404(
        self,
        async_client,
    ) -> None:
        """Async prompt on nonexistent session should return 404."""
        request = MessageRequest(
            parts=[TextPartInput(text="hello")],
            agent="default",
            message_id="msg-1",
        )
        response = await async_client.post(
            "/session/nonexistent-id/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_prompt_async_creates_user_message(
        self,
        async_client,
        server_state,
    ) -> None:
        """Async prompt creates a user message in session history."""
        response = await async_client.post("/session", json={"title": "Async Message"})
        session_id = response.json()["id"]

        from wolfharness_server.opencode_server.session_pool_integration import (
            get_messages_for_session,
        )

        request = MessageRequest(
            parts=[TextPartInput(text="hello world")],
            agent="default",
            message_id="msg-1",
        )
        response = await async_client.post(
            f"/session/{session_id}/prompt_async",
            json=request.model_dump(mode="json"),
        )
        assert response.status_code == 204

        # Give a moment for the message to be appended
        await asyncio.sleep(0.05)

        messages = await get_messages_for_session(server_state, session_id)
        assert len(messages) >= 1
