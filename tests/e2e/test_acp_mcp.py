"""L4 subprocess E2E tests for ACP MCP methods (B5 group).

Tests MCP-over-ACP tunneling:
    - B5.1 test_mcp_message_agent_side: send mcp/message to agent, verify response
    - B5.2 test_mcp_connect_client_side [skip]: mcp/connect client-side
    - B5.3 test_mcp_disconnect_client_side [skip]: mcp/disconnect client-side
    - B5.4 test_mcp_message_client_side [skip]: mcp/message client-side

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from acp import (
    ClientSideConnection,
    CloseSessionRequest,
    DefaultACPClient,
    InitializeRequest,
    NewSessionRequest,
)
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    import asyncio
    from pathlib import Path


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper (mirrors test_acp_subprocess.py)
# ---------------------------------------------------------------------------


class _ACPServerHandle:
    """Handle to a spawned ACP server subprocess with a client connection."""

    def __init__(
        self,
        conn: ClientSideConnection,
        process: asyncio.subprocess.Process,
        client: DefaultACPClient,
    ) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> object:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))

    async def new_session(self, cwd: str = "/tmp") -> object:
        return await self.conn.new_session(NewSessionRequest(cwd=cwd, mcp_servers=[]))

    async def close_session(self, session_id: str) -> None:
        await self.conn.close_session(CloseSessionRequest(session_id=session_id))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_mcp_message_agent_side(e2e_config: Path) -> None:
    """B5.1: Trigger mcp/message (agent-side) and verify the server responds.

    The agent-side ``mcp/message`` handler (in ``ACPProtocolHandler.ext_method``)
    accepts a ``connectionId`` and broadcasts the message to active sessions.
    For an unknown connectionId, the server logs a warning and returns ``{}``.
    We verify the server does not crash and returns a valid (empty) response.
    """
    import contextlib
    import os

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"

    client = DefaultACPClient(allow_file_operations=False)
    async with spawn_agent_process(
        lambda _conn: client,
        "wolfharness",
        "serve-acp",
        str(e2e_config),
        "--agent",
        "test_agent",
        env=env,
        log_stderr=False,
    ) as (conn, process):
        handle = _ACPServerHandle(conn=conn, process=process, client=client)
        await handle.initialize()
        new_sess = await handle.new_session()
        session_id = new_sess.session_id  # type: ignore[attr-defined]

        # Send mcp/message with an unknown connectionId.
        # The agent-side handler should return {} without crashing.
        response = await conn.send_request(
            "mcp/message",
            {
                "connectionId": "unknown-conn-id",
                "message": {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                },
            },
        )

        # The server should return an empty dict for unknown connections.
        assert response is not None, "Expected a response from mcp/message"
        assert isinstance(response, dict), f"Expected dict, got {type(response)}"

        # Server process should still be alive.
        assert process.returncode is None, "Server process exited after mcp/message"

        with contextlib.suppress(Exception):
            await handle.close_session(session_id)


async def test_mcp_connect_client_side(e2e_config: Path) -> None:
    """Test intent: Send mcp/connect request with a valid MCP server config.

    (``name``, ``transport``, ``command``/``url``). Expect ``result.connected = true``
    and ``result.server_info`` containing server ``name`` and ``version``. Verify
    ``mcp_server_added`` SessionUpdate emitted. Error case: invalid URL or missing
    required fields → error response with ``code = -32602`` (invalid params).
    """


async def test_mcp_disconnect_client_side(e2e_config: Path) -> None:
    """Test intent: Send mcp/connect to establish a connection, then.

    mcp/disconnect with the same server ``name``. Expect
    ``result.disconnected = true``. Verify ``mcp_server_removed`` SessionUpdate
    emitted. Error case: disconnect non-existent server → error with
    ``code = -32002`` (resource not found).
    """


async def test_mcp_message_client_side(e2e_config: Path) -> None:
    """Test intent: Send mcp/message from client side with a valid MCP tool call.

    payload (``method``, ``params`` containing tool ``name`` and ``arguments``).
    Expect ``result`` containing tool response with ``content`` array. Verify
    response matches MCP tool schema. Error case: unknown tool name → error with
    ``code = -32602`` (invalid params); server not connected → ``code = -32002``.
    """
