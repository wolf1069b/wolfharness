"""End-to-end integration tests for MCP capability building and ACP transport lifecycle.

Covers five scenarios that exercise the full chain from MCPManager
through AcpMcpConnectionManager, AcpMcpTransport, and
SessionConnectionPool:

G2: get_capabilities() with ACP config creates a real toolset cached on
    the session context.
G3: connect_acp_mcp_server() -> add_acp_transport() -> snapshot merge
    leaves the correct state for get_capabilities() to find.
G4: get_capabilities() returns a non-empty list and the underlying ACP
    transport's connect_session() + ClientSession.initialize() works
    end-to-end through the mock ACP client.
G10: cleanup_session() properly removes all resources when the ACP
    transport fails mid-execution, without hanging.
G12: child session inherits parent's ACP transports via
    copy_pre_created_transports() and get_capabilities() finds them.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
from mcp.types import (
    Implementation,
    InitializeResult,
    ServerCapabilities,
)
import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness.mcp_server.config_snapshot import McpConfigEntry, McpConfigSnapshot
from wolfharness.mcp_server.manager import MCPManager
from wolfharness_config.mcp_server import AcpMCPServerConfig
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.acp_mcp_transport import AcpMcpTransport


# ---------------------------------------------------------------------------
# Shared fakes and helpers
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Fake ClientTransport for testing add_acp_transport and get_capabilities.

    Mimics the interface needed by MCPToolset: provides a
    ``connect_session`` async context manager that yields ``None``
    (no real ClientSession).
    """

    def __init__(self, label: str = "fake-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self) -> Any:
        yield None


class _FailingTransport:
    """Transport whose connect_session raises on entry."""

    def __init__(self, label: str = "failing-acp") -> None:
        self.label = label

    @asynccontextmanager
    async def connect_session(self) -> Any:
        raise RuntimeError("transport failure")
        yield  # pragma: no cover — unreachable


class _FakeToolset:
    """Fake MCPToolset that captures the transport for inspection."""

    def __init__(self, **kwargs: Any) -> None:
        self.client: Any = kwargs.get("client")
        self.id: Any = kwargs.get("id")
        self.is_running = False

    async def __aenter__(self) -> Self:
        self.is_running = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.is_running = False


class _FakeMCP:
    """Fake MCP capability that exposes the underlying toolset."""

    def __init__(
        self,
        local: Any = None,
        allowed_tools: list[str] | None = None,
        id: str | None = None,  # noqa: A002
        **kwargs: Any,
    ) -> None:
        self.local = local
        self.allowed_tools = allowed_tools
        self.id = id


def _make_wired_managers() -> tuple[MCPManager, AcpMcpConnectionManager]:
    """Create wired MCPManager + AcpMcpConnectionManager for testing.

    Returns a tuple ``(mcp_manager, acp_manager)`` where
    ``mcp_manager._acp_mcp_manager`` is set to ``acp_manager`` so that
    ``cleanup_session()`` delegates to ``AcpMcpConnectionManager``.
    """
    acp_manager = AcpMcpConnectionManager()
    mcp_manager = MCPManager(name="test")
    mcp_manager._acp_mcp_manager = acp_manager
    return mcp_manager, acp_manager


def _make_mock_send_request(connection_id: str) -> AsyncMock:
    """Create a mock send_request that returns connectionId for mcp/connect."""

    async def _mock(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "mcp/connect":
            return {"connectionId": connection_id}
        return {}

    return AsyncMock(side_effect=_mock)


def _make_responsive_send_request(
    connection_id: str,
    received_messages: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a send_request callable that responds to mcp/connect and mcp/message.

    Responds to ``initialize`` with a valid InitializeResult and to
    ``tools/list`` with a single test tool.  All received mcp/message
    params are appended to ``received_messages`` if provided.
    """

    async def _send(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "mcp/connect":
            return {"connectionId": connection_id}
        if method == "mcp/message":
            if received_messages is not None:
                received_messages.append(params)
            req_method = params.get("method")
            if req_method == "initialize":
                result = InitializeResult(
                    protocolVersion="2024-11-05",
                    capabilities=ServerCapabilities(),
                    serverInfo=Implementation(name="test", version="1.0"),
                )
                return cast(
                    dict[str, Any],
                    result.model_dump(by_alias=True, mode="json", exclude_none=True),
                )
            if req_method == "tools/list":
                return {
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        }
                    ]
                }
        return {}

    return _send


# ---------------------------------------------------------------------------
# G2: get_capabilities with ACP config creates real toolset
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_capabilities_with_acp_config_creates_real_toolset() -> None:
    """get_capabilities(session_id) with ACP config creates a real toolset.

    Verifies that when a snapshot contains an ACP config entry and the
    session's connection pool has a pre-created transport for that
    config's client_id, ``get_capabilities()`` returns a non-empty list
    with a toolset that is:
    1. Cached in ``ctx.toolset_cache``.
    2. Constructed with the transport from ``ctx.connection_pool``.
    """
    mcp_manager, _acp_manager = _make_wired_managers()
    session_id = "g2-session"
    acp_config = AcpMCPServerConfig(name="g2-server", acp_id="g2-acp-id")
    client_id = acp_config.client_id
    transport = _FakeTransport("g2-transport")

    try:
        ctx = mcp_manager.get_or_create_session(session_id)
        assert ctx.connection_pool is not None
        await ctx.connection_pool.add_transport(client_id, cast(Any, transport))

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        mcp_manager.update_session_snapshot(session_id, snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=session_id)

        assert len(caps) == 1
        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is transport
        assert client_id in ctx.toolset_cache
        assert ctx.toolset_cache[client_id] is toolset
    finally:
        await mcp_manager.cleanup_session(session_id)
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# G3: connect_acp_mcp_server -> add_acp_transport -> snapshot merge
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_initialize_mcp_servers_full_chain() -> None:
    """connect_acp_mcp_server + add_acp_transport leaves correct state.

    Exercises the chain that ``ACPSession.initialize_mcp_servers()``
    would follow:
    1. ``connect_acp_mcp_server(server, session_id)`` sends mcp/connect,
       creates AcpMcpConnection, registers session + session connection.
    2. ``AcpMcpTransport(conn)`` wraps the connection.
    3. ``add_acp_transport()`` registers the transport in the session
       connection pool and tracks the ACP connection ID.

    After setup, verifies:
    - ``_session_connections`` has the session.
    - ``acp_connection_ids`` has the ``(connection_id, session_key)`` tuple.
    - Snapshot with ACP config entry allows ``get_capabilities()`` to find
      the transport.
    """
    from wolfharness import Agent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "g3-session"
    connection_id = "g3-conn"
    server_config = AcpMcpServer(name="g3-server", id="g3-acp-id")
    mock_client = MagicMock()
    mock_client.send_request = _make_mock_send_request(connection_id)

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    pool = AgentPool(manifest)
    agent = Agent.from_callback(
        name="test_agent",
        callback=simple_callback,
        agent_pool=pool,
    )
    agent.mcp = mcp_manager
    acp_agent = AgentPoolACPAgent(client=mock_client, default_agent=agent)
    acp_agent._mcp_manager = acp_manager

    try:
        conn_id, session_key = await acp_agent.connect_acp_mcp_server(
            server_config,
            session_id,
        )
        assert conn_id == connection_id

        conn = acp_manager.get_connection(connection_id)
        assert conn is not None

        transport = AcpMcpTransport(conn)

        acp_server_config = AcpMCPServerConfig(
            acp_id=server_config.id,
            name=server_config.name,
        )
        await mcp_manager.add_acp_transport(
            session_id,
            client_id=acp_server_config.client_id,
            transport=cast(Any, transport),
            connection_id=connection_id,
            session_key=session_key,
        )

        assert session_id in acp_manager._session_connections
        assert (connection_id, session_key) in acp_manager._session_connections[session_id]

        ctx = mcp_manager.get_or_create_session(session_id)
        assert (connection_id, session_key) in ctx.acp_connection_ids

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_server_config, source="session"),),
        )
        mcp_manager.update_session_snapshot(session_id, snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=session_id)

        assert len(caps) == 1
        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is transport
    finally:
        await mcp_manager.cleanup_session(session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# G4: Tool execution through get_capabilities and ACP transport
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_tool_execution_through_get_capabilities_and_acp_transport() -> None:  # noqa: PLR0915
    """get_capabilities + AcpMcpTransport.connect_session + ClientSession.initialize.

    This is the hardest test.  It verifies the complete end-to-end flow:

    1. ``connect_acp_mcp_server()`` creates an AcpMcpConnection with a
       mock client that responds to mcp/connect and mcp/message.
    2. ``AcpMcpTransport(conn)`` wraps the connection.
    3. ``add_acp_transport()`` registers the transport.
    4. ``get_capabilities(session_id)`` returns a non-empty list with a
       toolset built from the ACP transport.
    5. The ACP transport's ``connect_session()`` creates a real
       ``ClientSession``.
    6. ``ClientSession.initialize()`` sends an MCP ``initialize``
       request through the ACP channel (via ``_forward_to_client`` and
       ``send_to_client``).
    7. The mock client responds with a valid ``InitializeResult``.
    8. ``session.list_tools()`` sends ``tools/list`` and receives a
       tool list.

    If ``_forward_to_client`` is broken, ``initialize()`` hangs forever.
    """
    from wolfharness import Agent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest
    from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent

    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "g4-session"
    connection_id = "g4-conn"
    server_config = AcpMcpServer(name="g4-server", id="g4-acp-id")
    received_messages: list[dict[str, Any]] = []

    mock_client = MagicMock()
    mock_client.send_request = _make_responsive_send_request(
        connection_id,
        received_messages,
    )

    def simple_callback(message: str) -> str:
        return f"Test: {message}"

    manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    pool = AgentPool(manifest)
    agent = Agent.from_callback(
        name="test_agent",
        callback=simple_callback,
        agent_pool=pool,
    )
    agent.mcp = mcp_manager
    acp_agent = AgentPoolACPAgent(client=mock_client, default_agent=agent)
    acp_agent._mcp_manager = acp_manager

    try:
        conn_id, _session_key = await acp_agent.connect_acp_mcp_server(
            server_config,
            session_id,
        )
        assert conn_id == connection_id

        conn = acp_manager.get_connection(connection_id)
        assert conn is not None

        transport = AcpMcpTransport(conn, timeout=5.0)

        acp_server_config = AcpMCPServerConfig(
            acp_id=server_config.id,
            name=server_config.name,
            timeout=10.0,
        )
        await mcp_manager.add_acp_transport(
            session_id,
            client_id=acp_server_config.client_id,
            transport=cast(Any, transport),
            connection_id=connection_id,
            session_key=_session_key,
        )

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_server_config, source="session"),),
        )
        mcp_manager.update_session_snapshot(session_id, snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=session_id)

        assert len(caps) == 1
        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is transport

        with anyio.fail_after(10):
            async with transport.connect_session() as session:
                assert session is not None
                await session.initialize()

                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                assert "test_tool" in tool_names

        assert len(received_messages) >= 1
        init_found = any(msg.get("method") == "initialize" for msg in received_messages)
        assert init_found, f"No initialize request found in {len(received_messages)} messages"
        tools_list_found = any(msg.get("method") == "tools/list" for msg in received_messages)
        assert tools_list_found, f"No tools/list request found in {len(received_messages)} messages"
    finally:
        await mcp_manager.cleanup_session(session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# G10: ACP transport failure during tool execution -> cleanup works
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_acp_transport_failure_during_tool_execution_cleanup() -> None:
    """cleanup_session() removes all resources when ACP transport fails.

    When ``AcpMcpTransport.connect_session()`` or a tool call fails
    mid-execution, subsequent ``cleanup_session()`` must properly remove
    all resources without hanging.

    Setup:
    1. Create wired MCPManager + AcpMcpConnectionManager.
    2. Create ACP connection + register session.
    3. Add a failing transport (raises RuntimeError on connect_session).
    4. Build snapshot and call get_capabilities() (toolset created but
       not entered — no failure yet).
    5. Attempt to enter the toolset's transport -> RuntimeError.
    6. Call cleanup_session() — must complete without hanging.

    Asserts:
    - The session context is removed.
    - ``_session_connections`` is empty.
    - ``_connections`` is empty (connection removed because no active
      sessions remain).
    """
    mcp_manager, acp_manager = _make_wired_managers()
    session_id = "g10-session"
    connection_id = "g10-conn"
    server_config = AcpMcpServer(name="g10-server", id="g10-acp-id")
    send_to_client = AsyncMock(return_value=None)

    try:
        conn = await acp_manager.create_connection(
            connection_id,
            server_config,
            send_to_client,
        )
        _pair, session_key = conn.register_session()
        acp_manager.register_session_connection(
            session_id,
            connection_id,
            session_key,
        )

        failing_transport = _FailingTransport("g10-failing")
        acp_config = AcpMCPServerConfig(name="g10-server", acp_id="g10-acp-id")
        await mcp_manager.add_acp_transport(
            session_id,
            client_id=acp_config.client_id,
            transport=cast(Any, failing_transport),
            connection_id=connection_id,
            session_key=session_key,
        )

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        mcp_manager.update_session_snapshot(session_id, snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=session_id)

        assert len(caps) == 1
        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is failing_transport

        with pytest.raises(RuntimeError, match="transport failure"):
            async with failing_transport.connect_session():
                pass

        with anyio.fail_after(5):
            await mcp_manager.cleanup_session(session_id)

        assert mcp_manager.get_session_context(session_id) is None
        assert session_id not in acp_manager._session_connections
        assert connection_id not in acp_manager._connections
    finally:
        if mcp_manager.get_session_context(session_id) is not None:
            await mcp_manager.cleanup_session(session_id)
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# G12: Child session inherits parent ACP transports
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_child_session_inherits_parent_acp_transports() -> None:
    """copy_pre_created_transports() inherits parent ACP transports.

    Verifies that ``SessionConnectionPool.copy_pre_created_transports()``
    copies pre-created (ACP) transports from the parent's connection pool
    to the child's, and that the child's ``get_capabilities()`` can find
    the ACP transport.

    Setup:
    1. Create parent session with ACP transport via ``add_acp_transport()``.
    2. Create child session.
    3. Call ``copy_pre_created_transports()`` from parent's pool to
       child's pool.
    4. Build snapshot with ACP config entry for the child.
    5. Call ``get_capabilities(session_id=child_session_id)``.

    Asserts:
    - Child's ``connection_pool`` has the pre-created transport.
    - ``get_capabilities()`` returns a non-empty list for the child.
    - The child's toolset uses the inherited transport.
    """
    mcp_manager, _acp_manager = _make_wired_managers()
    parent_session_id = "g12-parent"
    child_session_id = "g12-child"
    acp_config = AcpMCPServerConfig(name="g12-server", acp_id="g12-acp-id")
    client_id = acp_config.client_id
    transport = _FakeTransport("g12-parent-transport")

    try:
        parent_ctx = mcp_manager.get_or_create_session(parent_session_id)
        assert parent_ctx.connection_pool is not None
        await parent_ctx.connection_pool.add_transport(client_id, cast(Any, transport))

        child_ctx = mcp_manager.get_or_create_session(child_session_id)
        assert child_ctx.connection_pool is not None
        await child_ctx.connection_pool.copy_pre_created_transports(
            parent_ctx.connection_pool,
        )

        child_pool = child_ctx.connection_pool
        inherited_transport = await child_pool.get_transport(acp_config)
        assert inherited_transport is cast(Any, transport)

        snapshot = McpConfigSnapshot(
            session_configs=(McpConfigEntry(server_config=acp_config, source="session"),),
        )
        mcp_manager.update_session_snapshot(child_session_id, snapshot)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=child_session_id)

        assert len(caps) == 1
        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is transport
        assert client_id in child_ctx.toolset_cache
    finally:
        await mcp_manager.cleanup_session(parent_session_id)
        await mcp_manager.cleanup_session(child_session_id)
        await mcp_manager.cleanup()
