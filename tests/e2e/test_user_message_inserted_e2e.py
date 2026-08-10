"""L4a smoke E2E test for ``UserMessageInsertedEvent`` through the ACP server.

Verifies that a ``UserMessageChunk`` notification appears in ``session/update``
notifications when a prompt is sent with ``_meta.delivery="steer"``.

L4a smoke: ``pytest -m "e2e and not slow"`` (~30s).
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
    PromptRequest,
    TextContentBlock,
)
from acp.stdio import spawn_agent_process
from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from acp.schema import InitializeResponse, NewSessionResponse, PromptResponse


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows stdio subprocess issues"),
]


@contextlib.asynccontextmanager
async def _spawn_acp_server(
    config_path: Path | str,
    *,
    agent: str = "test_agent",
) -> AsyncIterator[tuple[ClientSideConnection, Any, DefaultACPClient]]:
    """Spawn ``wolfharness serve-acp`` and return (conn, process, client).

    Args:
        config_path: Path to the YAML config file.
        agent: Agent name to use.
    """
    import os

    env = os.environ.copy()
    env["OBSERVABILITY_ENABLED"] = "false"
    env["LOGFIRE_DISABLE"] = "true"

    client = DefaultACPClient(allow_file_operations=False)
    async with spawn_agent_process(
        lambda _conn: client,
        "wolfharness",
        "serve-acp",
        str(config_path),
        "--agent",
        agent,
        env=env,
        log_stderr=False,
    ) as (conn, process):
        yield conn, process, client


async def test_acp_server_steer_user_message_chunk_position(
    e2e_config: Path,
) -> None:
    """L4a: Verify ``UserMessageChunk`` appears for a steer-marked prompt.

    Spawns a real ``wolfharness serve-acp`` subprocess, connects an ACP client,
    sends an initial prompt, then sends a second prompt with
    ``_meta.delivery="steer"``. The ACP server's ``_emit_user_message_chunks``
    emits ``UserMessageChunk`` session updates for the user's input before
    processing. This test verifies that the steer prompt's text appears in
    at least one ``user_message_chunk`` notification.

    Steps:
        1. Spawn serve-acp subprocess with TestModel config.
        2. Initialize JSON-RPC, create a session.
        3. Send first prompt (initial) and wait for completion.
        4. Send second prompt with ``field_meta={"delivery": "steer"}``.
        5. Collect all ``session/update`` notifications.
        6. Assert at least one ``user_message_chunk`` contains the steer text.
    """
    steer_text = "STEER_MESSAGE_MARKER_42"
    initial_text = "Hello from initial prompt"

    async with _spawn_acp_server(e2e_config) as (conn, process, client):
        assert process.returncode is None, "ACP server process exited early"

        # 1. Initialize
        init_resp: InitializeResponse = await conn.initialize(InitializeRequest(protocol_version=1))
        assert init_resp.protocol_version == 1

        # 2. Create session
        new_sess: NewSessionResponse = await conn.new_session(
            NewSessionRequest(cwd="/tmp", mcp_servers=[])
        )
        session_id = new_sess.session_id
        assert session_id, "Expected non-empty session_id"

        # 3. Send initial prompt and wait for completion
        client.notifications.clear()
        initial_resp: PromptResponse = await conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text=initial_text)],
            )
        )
        assert initial_resp.stop_reason is not None

        # 4. Send steer prompt with _meta.delivery="steer"
        client.notifications.clear()
        steer_resp: PromptResponse = await conn.prompt(
            PromptRequest(
                session_id=session_id,
                prompt=[TextContentBlock(text=steer_text)],
                field_meta={"delivery": "steer"},
            )
        )
        assert steer_resp.stop_reason is not None

        # 5. Collect all session/update notifications from the steer prompt
        notifications = list(client.notifications)

        # 6. Assert at least one user_message_chunk contains the steer text
        user_message_chunks = [
            n for n in notifications if n.update.session_update == "user_message_chunk"
        ]
        assert len(user_message_chunks) > 0, (
            "Expected at least one user_message_chunk notification for the steer prompt; "
            f"got {len(notifications)} total notifications with types: "
            f"{[n.update.session_update for n in notifications]}"
        )

        # Extract text content from the user_message_chunk notifications
        chunk_texts: list[str] = []
        for chunk in user_message_chunks:
            content = chunk.update.content
            if content is not None and hasattr(content, "text") and content.text:
                chunk_texts.append(content.text)

        assert any(steer_text in t for t in chunk_texts), (
            f"Expected steer text '{steer_text}' in user_message_chunk content; "
            f"got chunk_texts: {chunk_texts}"
        )
