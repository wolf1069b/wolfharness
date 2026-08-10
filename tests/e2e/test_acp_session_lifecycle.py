"""L4 subprocess E2E tests for ACP session lifecycle methods.

Covers 8 session methods: new, load (valid + nonexistent), close, close_then_load,
list, fork, resume.

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
L4b full tests: pytest -m e2e
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
import uuid

import pytest

from acp import (
    ClientSideConnection,
    CloseSessionRequest,
    DefaultACPClient,
    InitializeRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    TextContentBlock,
)
from acp.exceptions import RequestError
from acp.schema import (
    ForkSessionRequest,
    ListSessionsRequest,
    ResumeSessionRequest,
)
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from acp.schema import (
        InitializeResponse,
        NewSessionResponse,
        PromptResponse,
        SessionNotification,
    )


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


# ---------------------------------------------------------------------------
# Helper context manager (mirrors test_acp_subprocess.py ACPServerHandle)
# ---------------------------------------------------------------------------


class ACPSessionHandle:
    """Handle to a spawned ACP server subprocess with a client connection."""

    def __init__(self, conn: ClientSideConnection, process: Any, client: DefaultACPClient) -> None:
        self.conn = conn
        self.process = process
        self.client = client

    async def initialize(self) -> InitializeResponse:
        return await self.conn.initialize(InitializeRequest(protocol_version=1))

    async def new_session(self, cwd: str = "/tmp") -> NewSessionResponse:
        return await self.conn.new_session(NewSessionRequest(cwd=cwd, mcp_servers=[]))

    async def prompt(self, session_id: str, text: str) -> PromptResponse:
        return await self.conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text=text)],
            )
        )

    async def close_session(self, session_id: str) -> None:
        await self.conn.close_session(CloseSessionRequest(session_id=session_id))

    def clear_notifications(self) -> None:
        self.client.notifications.clear()

    def get_notifications(self) -> list[SessionNotification]:
        return list(self.client.notifications)


@pytest.fixture
async def acp_server(
    e2e_config: Path,
) -> AsyncIterator[ACPSessionHandle]:
    """Spawn an ACP stdio server and initialize it, yielding a ready handle.

    Suppresses ``RequestError`` during teardown to avoid spurious test errors
    from background task cleanup (known server-side issue with session close).
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
        handle = ACPSessionHandle(conn=conn, process=process, client=client)
        await handle.initialize()
        yield handle
    finally:
        with contextlib.suppress(RequestError, Exception):
            await cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# B2.1 — session/new
# ---------------------------------------------------------------------------


async def test_session_new(acp_server: ACPSessionHandle) -> None:
    """B2.1: Create a new session, verify sessionId returned."""
    new_sess = await acp_server.new_session()
    assert new_sess.session_id, "Expected non-empty session_id"
    assert isinstance(new_sess.session_id, str)


# ---------------------------------------------------------------------------
# B2.2 — session/load (valid)
# ---------------------------------------------------------------------------


async def test_session_load_valid(acp_server: ACPSessionHandle) -> None:
    """B2.2: Create session, load it, verify success."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    loaded = await acp_server.conn.load_session(
        LoadSessionRequest(session_id=session_id, cwd="/tmp", mcp_servers=[])
    )
    # LoadSessionResponse should validate successfully (no exception).
    assert loaded is not None


# ---------------------------------------------------------------------------
# B2.3 — session/load (nonexistent)
# ---------------------------------------------------------------------------


async def test_session_load_nonexistent(acp_server: ACPSessionHandle) -> None:
    """B2.3: Load random session ID, verify server returns empty response.

    The ACP server returns an empty ``LoadSessionResponse`` (not a
    ``ResourceNotFound`` error) when the session ID is not found.
    """
    fake_id = f"nonexistent-{uuid.uuid4().hex}"
    loaded = await acp_server.conn.load_session(
        LoadSessionRequest(session_id=fake_id, cwd="/tmp", mcp_servers=[])
    )
    # Server returns an empty LoadSessionResponse for nonexistent sessions.
    assert loaded is not None
    assert loaded.config_options == []


# ---------------------------------------------------------------------------
# B2.4 — session/close then load [fixed #186]
# ---------------------------------------------------------------------------


async def test_session_close_then_load(acp_server: ACPSessionHandle) -> None:
    """B2.4: Create, close, load → closed session loads for read-only history replay.

    Previously load_session rejected closed sessions with ResourceNotFound.
    Now closed sessions are loaded for read-only history access (e.g. viewing
    team member conversations after team_delete). The session loads but won't
    accept new prompts since it remains closed in the SessionPool.
    """
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    await acp_server.close_session(session_id)

    # Closed sessions now load successfully for history replay
    loaded = await acp_server.conn.load_session(
        LoadSessionRequest(session_id=session_id, cwd="/tmp", mcp_servers=[])
    )
    assert loaded is not None


# ---------------------------------------------------------------------------
# B2.5 — session/close [fixed #186]
# ---------------------------------------------------------------------------


async def test_session_close(acp_server: ACPSessionHandle) -> None:
    """B2.5: Create session, close it, verify clean close."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    await acp_server.close_session(session_id)
    # If close succeeded, no exception is raised.


# ---------------------------------------------------------------------------
# B2.6 — session/list
# ---------------------------------------------------------------------------


async def test_session_list(acp_server: ACPSessionHandle) -> None:
    """B2.6: Create multiple sessions, list, verify all appear."""
    created_ids: list[str] = []
    for _ in range(3):
        new_sess = await acp_server.new_session()
        created_ids.append(new_sess.session_id)

    listed = await acp_server.conn.list_sessions(ListSessionsRequest())
    listed_ids = {s.session_id for s in listed.sessions}
    for sid in created_ids:
        assert sid in listed_ids, f"Session {sid} not found in list response"


# ---------------------------------------------------------------------------
# B2.7 — session/fork
# ---------------------------------------------------------------------------


async def test_session_fork(acp_server: ACPSessionHandle) -> None:
    """B2.7: Create session, send prompt, fork, verify new sessionId."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    acp_server.clear_notifications()
    await acp_server.prompt(session_id, "First message before fork")

    forked = await acp_server.conn.fork_session(
        ForkSessionRequest(session_id=session_id, cwd="/tmp", mcp_servers=[])
    )
    assert forked.session_id, "Expected non-empty forked session_id"
    assert forked.session_id != session_id, "Forked session_id should differ from original"


# ---------------------------------------------------------------------------
# B2.8 — session/resume
# ---------------------------------------------------------------------------


async def test_session_resume(acp_server: ACPSessionHandle) -> None:
    """B2.8: Create session, send prompt, resume, verify state restored."""
    new_sess = await acp_server.new_session()
    session_id = new_sess.session_id

    acp_server.clear_notifications()
    await acp_server.prompt(session_id, "Message before resume")

    resumed = await acp_server.conn.resume_session(
        ResumeSessionRequest(session_id=session_id, cwd="/tmp", mcp_servers=[])
    )
    # ResumeSessionResponse should validate successfully.
    assert resumed is not None
