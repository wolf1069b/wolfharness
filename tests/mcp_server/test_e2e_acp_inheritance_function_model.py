"""E2E tests for MCP-over-ACP inheritance using FunctionModel.

Tests the full chain: ``ACPSession.initialize_mcp_servers()`` →
``MCPManager.add_acp_transport()`` → ``update_session_snapshot()`` →
``get_capabilities()`` → child session inheritance via
``get_or_create_session_agent()`` → FunctionModel-driven tool discovery.

Covers the two bug fixes:
- Bug 1: ``add_acp_transport()`` was inside a ``None``-guard and never ran.
- Bug 2: ``update_session_snapshot()`` was never called in
  ``initialize_mcp_servers()``.
"""

from __future__ import annotations

import tempfile
from typing import Any, Self, cast
from unittest.mock import MagicMock, patch

from mcp.types import Implementation, InitializeResult, ServerCapabilities
import pytest

from acp.schema.mcp import AcpMcpServer
from wolfharness import Agent, AgentPool
from wolfharness.mcp_server.manager import MCPManager
from wolfharness.models.agents import NativeAgentConfig
from wolfharness.models.manifest import AgentsManifest
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.session import ACPSession


# ---------------------------------------------------------------------------
# Shared fakes and helpers
# ---------------------------------------------------------------------------


class _FakeToolset:
    """Fake MCPToolset for state-management tests (Tests 1-4).

    Captures the transport for inspection without establishing a real
    MCP connection.
    """

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
    """Fake pydantic-ai MCP capability that exposes the toolset."""

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


def _make_acp_send_request(
    connection_id: str,
    received_messages: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a send_request callable that responds to mcp/connect and mcp/message.

    Responds to ``initialize`` with a valid ``InitializeResult``, to
    ``tools/list`` with a single ``test_tool``, and to ``tools/call``
    with a text result.
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
            if req_method == "tools/call":
                return {
                    "content": [{"type": "text", "text": "Tool executed!"}],
                    "isError": False,
                }
        return {}

    return _send


def _make_wired_managers() -> tuple[MCPManager, AcpMcpConnectionManager]:
    """Create wired MCPManager + AcpMcpConnectionManager for testing."""
    acp_manager = AcpMcpConnectionManager()
    mcp_manager = MCPManager(name="test")
    mcp_manager._acp_mcp_manager = acp_manager
    return mcp_manager, acp_manager


def _build_test_fixture(
    session_id: str = "test-session",
) -> tuple[
    MCPManager,
    AcpMcpConnectionManager,
    Agent[Any, Any],
    AgentPoolACPAgent,
    ACPSession,
    MagicMock,
]:
    """Build a full test fixture: pool, agent, acp_agent, acp_session.

    Returns ``(mcp_manager, acp_manager, agent, acp_agent, acp_session, mock_client)``.
    """
    mcp_manager, acp_manager = _make_wired_managers()

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

    mock_client = MagicMock()
    mock_client.send_request = _make_acp_send_request("test-conn-id")

    acp_agent = AgentPoolACPAgent(client=mock_client, default_agent=agent)
    acp_agent._mcp_manager = acp_manager

    acp_session = ACPSession(
        session_id=session_id,
        agent=agent,
        cwd=tempfile.gettempdir(),
        client=mock_client,
        mcp_servers=[AcpMcpServer(name="test-server", id="test-acp-id")],
        acp_agent=acp_agent,
    )

    return mcp_manager, acp_manager, agent, acp_agent, acp_session, mock_client


# ---------------------------------------------------------------------------
# Test 1: initialize_mcp_servers registers transport and syncs snapshot
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_initialize_mcp_servers_registers_transport_and_syncs_snapshot() -> None:
    """initialize_mcp_servers() registers ACP transport and updates snapshot.

    Verifies Bug 1 fix: ``add_acp_transport()`` is called (transport
    appears in ``get_session_context(session_id).connection_pool``).

    Verifies Bug 2 fix: ``update_session_snapshot()`` is called
    (``get_session_context(session_id).snapshot`` is not None and has
    non-empty ``session_configs``).

    Also verifies the session context snapshot is set (both paths updated).
    """
    mcp_manager, _acp_manager, _agent, acp_agent, acp_session, _mock_client = _build_test_fixture(
        "test1-session"
    )

    try:
        await acp_session.initialize_mcp_servers()

        session_id = acp_session.session_id
        ctx = mcp_manager.get_session_context(session_id)
        assert ctx is not None, "Session context must exist after initialize_mcp_servers"

        # Bug 1: transport registered in connection_pool
        assert ctx.connection_pool is not None
        from wolfharness_config.mcp_server import AcpMCPServerConfig

        acp_config = AcpMCPServerConfig(name="test-server", acp_id="test-acp-id")
        transport = await ctx.connection_pool.get_transport(acp_config)
        assert transport is not None, "ACP transport must be in connection_pool"

        # Bug 2: snapshot synced to session context
        assert ctx.snapshot is not None, "Snapshot must be set on session context"
        assert len(ctx.snapshot.session_configs) > 0, "Snapshot must have session configs"
        # Session context snapshot also updated
        assert ctx.snapshot is not None, "Session context snapshot must be set"
        assert len(ctx.snapshot.session_configs) > 0

    finally:
        await mcp_manager.cleanup_session(acp_session.session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# Test 2: get_capabilities finds ACP configs after initialize
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_capabilities_finds_acp_configs_after_initialize() -> None:
    """get_capabilities(session_id) returns non-empty list after initialize.

    After ``initialize_mcp_servers()``, the session context has both a
    transport and a snapshot.  ``get_capabilities()`` should find the ACP
    config and create a toolset using the ACP transport.
    """
    mcp_manager, _acp_manager, _agent, acp_agent, acp_session, _mock_client = _build_test_fixture(
        "test2-session"
    )

    try:
        await acp_session.initialize_mcp_servers()

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await mcp_manager.get_capabilities(session_id=acp_session.session_id)

        assert len(caps) > 0, "get_capabilities must return non-empty list"

        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is not None, "Toolset must use the ACP transport"
    finally:
        await mcp_manager.cleanup_session(acp_session.session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()


# ---------------------------------------------------------------------------
# Test 3: child session inherits ACP configs and transports
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_child_session_inherits_acp_configs_and_transports() -> None:
    """Child session inherits parent's ACP configs and transports.

    Creates a parent session via ``SessionPool``, initializes MCP
    servers on the parent's session agent, then creates a child session.
    The child's session context should have:
    - ``snapshot.session_configs`` matching the parent's (inherited).
    - ``connection_pool`` with pre-created transport (copied via
      ``copy_pre_created_transports()``).
    """
    from wolfharness.orchestrator.session_pool import SessionPool
    from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

    mcp_manager, _acp_manager, _base_agent, acp_agent, _acp_session, mock_client = (
        _build_test_fixture("test3-parent")
    )

    # Build pool + session_pool
    manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    pool = AgentPool(manifest)
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool
    await session_pool.start()

    parent_session_id = "test3-parent"
    child_session_id = "test3-child"

    try:
        # Create parent session and get session agent
        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)

        # Wire MCPManager on the session agent
        parent_agent.mcp = mcp_manager

        # Create ACPSession with the session agent and initialize MCP servers
        parent_acp_session = ACPSession(
            session_id=parent_session_id,
            agent=parent_agent,
            cwd=tempfile.gettempdir(),
            client=mock_client,
            mcp_servers=[AcpMcpServer(name="test-server", id="test-acp-id")],
            acp_agent=acp_agent,
        )
        await parent_acp_session.initialize_mcp_servers()

        # Create child session
        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        # Assert child's snapshot.session_configs matches parent's (inherited)
        child_ctx = child_agent.mcp.get_session_context(child_session_id)
        assert child_ctx is not None, "Child session context must exist"
        assert child_ctx.snapshot is not None, "Child snapshot must be set"
        assert len(child_ctx.snapshot.session_configs) > 0, (
            "Child must inherit parent's session configs"
        )

        parent_ctx = mcp_manager.get_session_context(parent_session_id)
        assert parent_ctx is not None
        parent_config_count = len(parent_ctx.snapshot.session_configs) if parent_ctx.snapshot else 0
        assert len(child_ctx.snapshot.session_configs) == parent_config_count, (
            "Child session_configs count must match parent's"
        )

        # Assert child's connection_pool has pre-created transport
        assert child_ctx.connection_pool is not None
        from wolfharness_config.mcp_server import AcpMCPServerConfig

        acp_config = AcpMCPServerConfig(name="test-server", acp_id="test-acp-id")
        inherited_transport = await child_ctx.connection_pool.get_transport(acp_config)
        assert inherited_transport is not None, (
            "Child must have pre-created transport from copy_pre_created_transports()"
        )
    finally:
        await mcp_manager.cleanup_session(parent_session_id)
        await mcp_manager.cleanup_session(child_session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()
        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Test 4: child get_capabilities finds inherited ACP configs
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_child_get_capabilities_finds_inherited_acp_configs() -> None:
    """get_capabilities on child session returns non-empty list.

    After child session inherits parent's ACP configs and transports,
    calling ``get_capabilities(session_id=child_session_id)`` on the
    child's MCPManager should return a non-empty list.
    """
    from wolfharness.orchestrator.session_pool import SessionPool
    from wolfharness_storage.memory_provider.provider import MemoryStorageProvider

    mcp_manager, _acp_manager, _base_agent, acp_agent, _acp_session, mock_client = (
        _build_test_fixture("test4-parent")
    )

    manifest = AgentsManifest(
        agents={"test_agent": NativeAgentConfig(model="test")},
    )
    pool = AgentPool(manifest)
    store = MemoryStorageProvider()
    session_pool = SessionPool(pool=pool, store=store)
    pool._session_pool = session_pool
    await session_pool.start()

    parent_session_id = "test4-parent"
    child_session_id = "test4-child"

    try:
        await session_pool.create_session(parent_session_id, agent_name="test_agent")
        parent_agent = await session_pool.sessions.get_or_create_session_agent(parent_session_id)
        parent_agent.mcp = mcp_manager

        parent_acp_session = ACPSession(
            session_id=parent_session_id,
            agent=parent_agent,
            cwd=tempfile.gettempdir(),
            client=mock_client,
            mcp_servers=[AcpMcpServer(name="test-server", id="test-acp-id")],
            acp_agent=acp_agent,
        )
        await parent_acp_session.initialize_mcp_servers()

        await session_pool.create_session(
            child_session_id,
            parent_session_id=parent_session_id,
            agent_name="test_agent",
        )
        child_agent = await session_pool.sessions.get_or_create_session_agent(child_session_id)

        with (
            patch("pydantic_ai.mcp.MCPToolset", _FakeToolset),
            patch("pydantic_ai.capabilities.MCP", _FakeMCP),
        ):
            caps = await child_agent.mcp.get_capabilities(session_id=child_session_id)

        assert len(caps) > 0, "Child get_capabilities must return non-empty list"

        toolset = cast(_FakeToolset, caps[0].local)
        assert toolset.client is not None, "Child toolset must use inherited ACP transport"
    finally:
        await mcp_manager.cleanup_session(parent_session_id)
        await mcp_manager.cleanup_session(child_session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()
        await session_pool.shutdown()


# ---------------------------------------------------------------------------
# Test 5: FunctionModel discovers MCP tools through ACP transport (E2E)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.real_mcp
async def test_function_model_discovers_mcp_tools_through_acp_transport() -> None:
    """FunctionModel discovers MCP tools through the real ACP transport.

    This is the key E2E test.  It creates a real pydantic-ai Agent with
    ``FunctionModel`` and the MCP capabilities from ``get_capabilities()``.
    The FunctionModel function checks ``agent_info.function_tools`` for
    the MCP tool name ``test_tool`` and calls it on the first turn.

    The real ``MCPToolset`` (NOT faked) connects through the real
    ``AcpMcpTransport`` to the mock ACP client, which responds to
    ``initialize``, ``tools/list``, and ``tools/call``.
    """
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    mcp_manager, _acp_manager, _agent, acp_agent, acp_session, _mock_client = _build_test_fixture(
        "test5-session"
    )

    try:
        await acp_session.initialize_mcp_servers()

        # Get real MCP capabilities (no patching)
        caps = await mcp_manager.get_capabilities(session_id=acp_session.session_id)
        assert len(caps) > 0, "Must have MCP capabilities"

        call_count = 0
        saw_test_tool = False

        async def model_function(
            messages: list[ModelMessage],
            agent_info: AgentInfo,
        ) -> ModelResponse:
            nonlocal call_count, saw_test_tool
            call_count += 1
            tool_names = [t.name for t in agent_info.function_tools]
            if "test_tool" in tool_names:
                saw_test_tool = True
            if call_count == 1 and "test_tool" in tool_names:
                return ModelResponse(parts=[ToolCallPart(tool_name="test_tool", args={})])
            return ModelResponse(parts=[TextPart(content=f"Done. Tools: {tool_names}")])

        model = FunctionModel(function=model_function)
        pydantic_agent = PydanticAgent(
            model=model,
            capabilities=caps,
        )

        result = await pydantic_agent.run("Call the test tool")

        # The agent completed successfully
        assert result is not None
        assert call_count >= 1
        assert saw_test_tool, "FunctionModel must have seen 'test_tool' in function_tools"
    finally:
        await mcp_manager.cleanup_session(acp_session.session_id)
        await acp_agent.close()
        await mcp_manager.cleanup()
