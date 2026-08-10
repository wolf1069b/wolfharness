"""Shared helpers for ACP VCR tests.

Extracts the paired in-process pipe pattern (design D7) used by
``test_acp_protocol.py``, ``test_acp_tool_call.py``, and
``test_acp_subagent.py``. The pattern is ported from
``tests/servers/acp_server/test_rpc.py``.

The ACP protocol stack (JSON-RPC framing, event conversion, session
management) runs for real in-process. VCR intercepts only the model API
HTTP calls. The client and agent sides are connected via paired
``asyncio.StreamReader``/``StreamWriter`` pipes.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Self, cast

import anyio
from anyio.abc import ByteReceiveStream, ByteSendStream

from acp import AgentSideConnection, ClientSideConnection, DefaultACPClient
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent


if TYPE_CHECKING:
    from typing import Any, Protocol

    from wolfharness import AgentPool

    class _PoolWithGetAgent(Protocol):
        """Protocol for a pool with the ``get_agent`` compat shim attached."""

        def get_agent(self, name: str) -> Any: ...


__all__ = [
    "AsyncioReaderAdapter",
    "AsyncioWriterAdapter",
    "PairedPipe",
    "build_acp_agent",
    "send_prompt",
    "wait_for_notifications",
    "wire_connections",
]


# ---------------------------------------------------------------------------
# Pipe adapters — adapt asyncio streams to anyio's ByteStream interface
# ---------------------------------------------------------------------------


class AsyncioReaderAdapter(ByteReceiveStream):
    """Adapt ``asyncio.StreamReader`` to anyio's ``ByteReceiveStream``."""

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader

    async def receive(self, max_bytes: int = 65536) -> bytes:
        data = await self._reader.read(max_bytes)
        if not data:
            raise anyio.EndOfStream
        return data

    async def aclose(self) -> None:
        # StreamReader doesn't need explicit close
        pass


class AsyncioWriterAdapter(ByteSendStream):
    """Adapt ``asyncio.StreamWriter`` to anyio's ``ByteSendStream``."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def send(self, item: bytes) -> None:
        self._writer.write(item)
        await self._writer.drain()

    async def aclose(self) -> None:
        self._writer.close()
        with contextlib.suppress(Exception):
            await self._writer.wait_closed()


class PairedPipe:
    """Create paired asyncio pipes for ACP client/agent connections.

    A TCP server on ``127.0.0.1`` (ephemeral port) bridges the two sides
    so both can use ``asyncio.StreamReader``/``StreamWriter``. The server
    side (agent) and client side are connected via a single accepted
    connection.
    """

    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self.server_reader: asyncio.StreamReader | None = None
        self.server_writer: asyncio.StreamWriter | None = None
        self.client_reader: asyncio.StreamReader | None = None
        self.client_writer: asyncio.StreamWriter | None = None

    async def __aenter__(self) -> Self:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self.server_reader = reader
            self.server_writer = writer

        self._server = await asyncio.start_server(handle, host="127.0.0.1", port=0)
        host, port = self._server.sockets[0].getsockname()[:2]
        self.client_reader, self.client_writer = await asyncio.open_connection(host, port)
        # Wait until the server side accepts the connection.
        for _ in range(100):
            if self.server_reader and self.server_writer:
                break
            await anyio.sleep(0.01)
        assert self.server_reader is not None
        assert self.server_writer is not None
        return self

    async def __aexit__(self, *exc: object) -> None:
        for writer in (self.client_writer, self.server_writer):
            if writer:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# ACP agent builder
# ---------------------------------------------------------------------------


def build_acp_agent(
    pool: AgentPool,
    agent_name: str = "test_agent",
) -> tuple[AgentPoolACPAgent, DefaultACPClient]:
    """Build an ``AgentPoolACPAgent`` wired to a concrete ``DefaultACPClient``.

    The client is returned alongside the agent so tests can inspect
    ``client.notifications`` (a plain ``list[SessionNotification]``) after
    running a prompt through the ACP protocol stack.

    Args:
        pool: Real ``AgentPool`` with pre-created agents (use the
            ``vcr_pool`` / ``vcr_pool_with_tool`` / ``vcr_pool_with_subagent``
            fixtures). The ``get_agent`` compat shim must be attached
            (done by the VCR conftest fixtures).
        agent_name: Name of the agent in the pool to use as the default
            agent. Defaults to ``"test_agent"``.

    Returns:
        A ``(AgentPoolACPAgent, DefaultACPClient)`` tuple. The client is
        the same instance attached to ``agent.client``.
    """
    pool_with_shim = cast("_PoolWithGetAgent", pool)
    default_agent: Any = pool_with_shim.get_agent(agent_name)
    client = DefaultACPClient(allow_file_operations=False, use_real_files=False)
    acp_agent = AgentPoolACPAgent(client=client, default_agent=default_agent)
    return acp_agent, client


def wire_connections(
    pipe: PairedPipe,
    acp_agent: AgentPoolACPAgent,
    client: DefaultACPClient,
) -> ClientSideConnection:
    """Wire both sides of a paired pipe to the ACP agent and client.

    Returns the ``ClientSideConnection`` (used to drive initialize /
    new_session / prompt from the client side). The
    ``AgentSideConnection`` is created as a side effect and kept alive
    via the returned connection's lifecycle.
    """
    assert pipe.client_writer is not None
    assert pipe.client_reader is not None
    assert pipe.server_writer is not None
    assert pipe.server_reader is not None

    client_conn = ClientSideConnection(
        lambda _conn: client,
        AsyncioWriterAdapter(pipe.client_writer),
        AsyncioReaderAdapter(pipe.client_reader),
    )
    _agent_conn = AgentSideConnection(
        lambda _conn: acp_agent,
        AsyncioWriterAdapter(pipe.server_writer),
        AsyncioReaderAdapter(pipe.server_reader),
    )
    return client_conn


# ---------------------------------------------------------------------------
# Prompt sending
# ---------------------------------------------------------------------------


async def send_prompt(
    client_conn: ClientSideConnection,
    session_id: str,
    text: str,
) -> None:
    """Send a user prompt to the agent via the ACP ``session/prompt`` method.

    This is the correct client→agent request path. The agent processes
    the prompt and emits session notifications (agent message chunks,
    tool call events, etc.) back to the client via
    ``client.session_update()`` — which are collected in
    ``client.notifications``.
    """
    from acp import PromptRequest, TextContentBlock

    await client_conn.prompt(
        PromptRequest(
            session_id=session_id,
            prompt=[TextContentBlock(text=text)],
        )
    )


# ---------------------------------------------------------------------------
# Notification polling
# ---------------------------------------------------------------------------


async def wait_for_notifications(
    client: DefaultACPClient,
    expected_count: int,
    timeout: float = 15.0,
    poll_interval: float = 0.01,
) -> list[SessionNotification]:
    """Poll ``client.notifications`` until ``expected_count`` or ``timeout``.

    ``DefaultACPClient.notifications`` is a plain ``list`` (not async
    iterable) because the agent calls ``client.session_update()`` directly
    in-process. This helper mirrors the synchronous polling pattern from
    ``tests/servers/acp_server/test_rpc.py``.
    """
    iterations = int(timeout / poll_interval) + 1
    for _ in range(iterations):
        if len(client.notifications) >= expected_count:
            break
        await anyio.sleep(poll_interval)
    return list(client.notifications)


if TYPE_CHECKING:
    from acp import SessionNotification
