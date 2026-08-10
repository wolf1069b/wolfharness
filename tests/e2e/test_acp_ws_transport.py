"""L4 subprocess E2E tests for the ACP server over WebSocket (streamable-http).

L4a smoke tests (``@pytest.mark.e2e``, NOT slow):
    - test_ws_server_startup: WebSocket connect + initialize handshake
    - test_ws_basic_prompt: session/new + session/prompt over WebSocket
    - test_ws_shutdown: verify clean process exit after WebSocket interaction

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from acp import (
    ClientSideConnection,
    DefaultACPClient,
    InitializeRequest,
    NewSessionRequest,
    PromptRequest,
    TextContentBlock,
)
from acp.transports import _WebSocketReadStream, _WebSocketWriteStream
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS, _spawn_acp_ws_server


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from acp.schema import (
        SessionNotification,
    )
    from tests.e2e.conftest import ACPWSServerHandle, ProcessRegistry


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper: connect to the ACP WebSocket server and wrap in ClientSideConnection
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _connect_ws(
    handle: ACPWSServerHandle,
) -> AsyncIterator[tuple[ClientSideConnection, DefaultACPClient]]:
    """Open a WebSocket connection to the ACP server and wrap it in a ClientSideConnection.

    Args:
        handle: The running ACPWSServerHandle.

    Yields:
        A tuple of (ClientSideConnection, DefaultACPClient).
    """
    import websockets

    client = DefaultACPClient(allow_file_operations=False)
    async with websockets.connect(handle.ws_url) as ws:
        reader = _WebSocketReadStream(ws)
        writer = _WebSocketWriteStream(ws)
        conn = ClientSideConnection(client, writer, reader)
        try:
            yield conn, client
        finally:
            await conn.close()


# ---------------------------------------------------------------------------
# L4a Smoke Tests
# ---------------------------------------------------------------------------


async def test_ws_server_startup(
    e2e_config: Path,
    process_registry: ProcessRegistry,
) -> None:
    """L4a: Start serve-acp streamable-http, connect via WebSocket, initialize."""
    async with _spawn_acp_ws_server(e2e_config, process_registry) as handle:
        assert handle.process.returncode is None, "ACP WS server exited early"

        async with _connect_ws(handle) as (conn, _client):
            resp = await conn.initialize(InitializeRequest(protocol_version=1))
            assert resp.protocol_version == 1, (
                f"Expected protocol_version=1, got {resp.protocol_version}"
            )


async def test_ws_basic_prompt(
    e2e_config: Path,
    process_registry: ProcessRegistry,
) -> None:
    """L4a: session/new + session/prompt over WebSocket, verify stop_reason + notifications."""
    async with (
        _spawn_acp_ws_server(e2e_config, process_registry) as handle,
        _connect_ws(handle) as (conn, client),
    ):
        await conn.initialize(InitializeRequest(protocol_version=1))

        new_sess = await conn.new_session(NewSessionRequest(cwd="/tmp", mcp_servers=[]))
        session_id: str = new_sess.session_id
        assert session_id, "Expected non-empty session_id"

        client.notifications.clear()
        resp = await conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text="Hello over WebSocket!")],
            )
        )
        assert resp.stop_reason is not None, "Expected stop_reason in PromptResponse"

        notifications: list[SessionNotification] = list(client.notifications)
        assert len(notifications) > 0, "Expected at least one session/update notification"


async def test_ws_shutdown(
    e2e_config: Path,
    process_registry: ProcessRegistry,
) -> None:
    """L4a: Verify clean process exit after WebSocket interaction.

    The fixture teardown sends SIGTERM and waits for clean exit. We verify
    the process is still alive after a basic interaction, confirming the
    server didn't crash during WebSocket operation.
    """
    async with _spawn_acp_ws_server(e2e_config, process_registry) as handle:
        async with _connect_ws(handle) as (conn, _client):
            await conn.initialize(InitializeRequest(protocol_version=1))

        # Process should still be alive after WebSocket interaction completes.
        assert handle.process.returncode is None, (
            "ACP WS server process exited unexpectedly during operation"
        )
