"""L4 subprocess E2E tests for ACP session configuration methods.

Covers 3 config methods: session/set_mode, session/set_model,
session/set_config_option.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from acp import (
    ClientSideConnection,
    DefaultACPClient,
    InitializeRequest,
    NewSessionRequest,
    SetSessionModelRequest,
    SetSessionModeRequest,
)
from acp.exceptions import RequestError
from acp.schema import SetSessionConfigOptionRequest
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from acp.schema import InitializeResponse, NewSessionResponse


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper fixture
# ---------------------------------------------------------------------------


class ACPConfigHandle:
    """Handle to a spawned ACP server with a client connection for config tests."""

    def __init__(self, conn: ClientSideConnection, process: Any, client: DefaultACPClient) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> InitializeResponse:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))

    async def new_session(self, cwd: str = "/tmp") -> NewSessionResponse:
        return await self.conn.new_session(NewSessionRequest(cwd=cwd, mcp_servers=[]))


@pytest.fixture
async def acp_server(e2e_config: Path) -> AsyncIterator[ACPConfigHandle]:
    """Spawn an ACP stdio server and initialize it.

    Suppresses ``RequestError`` during teardown to avoid spurious test errors
    from background task cleanup.
    """
    import os

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"

    client = DefaultACPClient(allow_file_operations=False)
    cm = spawn_agent_process(
        lambda _conn: client,
        "wolfharness",
        "serve-acp",
        str(e2e_config),
        "--agent",
        "test_agent",
        env=env,
        log_stderr=False,
    )
    try:
        conn, process = await cm.__aenter__()
        handle = ACPConfigHandle(conn=conn, process=process, client=client)
        await handle.initialize()
        yield handle
    finally:
        with contextlib.suppress(RequestError, Exception):
            await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# B3.1 — session/set_mode
# ---------------------------------------------------------------------------


async def test_session_set_mode(acp_server: ACPConfigHandle) -> None:
    """B3.1: Create session, set mode, verify mode change (no exception)."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    # Use a plausible mode id; TestModel-backed server should accept or no-op.
    response = await acp_server.conn.set_session_mode(
        SetSessionModeRequest(mode_id="default", session_id=session_id)
    )
    assert response is not None


# ---------------------------------------------------------------------------
# B3.2 — session/set_model
# ---------------------------------------------------------------------------


async def test_session_set_model(acp_server: ACPConfigHandle) -> None:
    """B3.2: Create session, set model, verify model change (no exception)."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    response = await acp_server.conn.set_session_model(
        SetSessionModelRequest(model_id="test", session_id=session_id)
    )
    assert response is not None


# ---------------------------------------------------------------------------
# B3.3 — session/set_config_option
# ---------------------------------------------------------------------------


async def test_session_set_config_option(acp_server: ACPConfigHandle) -> None:
    """B3.3: Create session, set config option (e.g., agent_role), verify change."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    response = await acp_server.conn.set_session_config_option(
        SetSessionConfigOptionRequest(
            config_id="agent_role", session_id=session_id, value="test_agent"
        )
    )
    assert response is not None
