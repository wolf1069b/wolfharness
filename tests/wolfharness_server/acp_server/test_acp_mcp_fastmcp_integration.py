"""FastMCP end-to-end integration tests for MCP-over-ACP.

Tests the complete chain using real fastmcp ClientSession via AcpMcpTransport,
plus correlation registry verification with simulated MCP servers.

Direction A (ACP client -> MCP session -> back):
  ext_method("mcp/message") -> handle_client_message() -> to_session
  -> simulated server reads -> writes response -> from_session
  -> forwarder -> send_to_client() -> fulfill_pending_request()
  -> ext_method() returns result (NOT {})

Direction B (MCP client initiates requests):
  ClientSession.initialize() / list_tools() -> from_session
  -> forwarder -> send_to_client() -> _send_to_client()
  -> response -> to_session -> ClientSession receives

These tests are marked @pytest.mark.slow because real ClientSession
initialization takes ~100ms per test.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import anyio
from mcp.shared.message import SessionMessage
from mcp.types import (
    ElicitResult,
    Implementation,
    InitializeResult,
    JSONRPCResponse,
    ServerCapabilities,
)
import pytest

from acp.exceptions import RequestError
from acp.schema.mcp import AcpMcpServer
from wolfharness import Agent
from wolfharness.delegation import AgentPool
from wolfharness_server.acp_server.acp_agent import AgentPoolACPAgent
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnection
from wolfharness_server.acp_server.acp_mcp_transport import AcpMcpTransport


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


pytestmark = [pytest.mark.asyncio, pytest.mark.slow, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return Mock()


@pytest.fixture
def default_test_agent() -> Agent:
    """Create a simple test agent with a pool."""
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.models.manifest import AgentsManifest

    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    manifest = AgentsManifest(agents={"test_agent": NativeAgentConfig(model="test")})
    pool = AgentPool(manifest)
    return Agent.from_callback(name="test_agent", callback=simple_callback, agent_pool=pool)


@pytest.fixture
def acp_agent(mock_connection, default_test_agent: Agent) -> AgentPoolACPAgent:
    """Create a mock ACP agent for testing."""
    return AgentPoolACPAgent(client=mock_connection, default_agent=default_test_agent)


@pytest.fixture
def server_config() -> AcpMcpServer:
    """Create a test ACP MCP server configuration."""
    return AcpMcpServer(name="test-server", id="test-id")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_stateful_mock_send_to_client(
    connection_id: str,
) -> AsyncMock:
    """Create a stateful mock that returns JSON-RPC responses based on method.

    Returns appropriate responses for initialize and tools/list requests.
    """
    _counters = {"initialize": 0, "tools/list": 0}

    async def mock_send_to_client(message: dict[str, Any]) -> dict[str, Any]:
        inner = message.get("message", {})
        if not isinstance(inner, dict):
            return {}

        method = inner.get("method")
        req_id = inner.get("id")

        if method == "initialize" and req_id is not None:
            _counters["initialize"] += 1
            result = InitializeResult(
                protocolVersion="2024-11-05",
                capabilities=ServerCapabilities(),
                serverInfo=Implementation(name="test", version="1.0"),
            )
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=req_id,
                result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
            ).model_dump(by_alias=True, mode="json", exclude_none=True)

        if method == "tools/list" and req_id is not None:
            _counters["tools/list"] += 1
            return JSONRPCResponse(
                jsonrpc="2.0",
                id=req_id,
                result={
                    "tools": [
                        {
                            "name": "test_tool",
                            "description": "A test tool",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                },
            ).model_dump(by_alias=True, mode="json", exclude_none=True)

        # For notifications (no id), just return empty dict
        return {}

    mock = AsyncMock(side_effect=mock_send_to_client)
    # Attach counters as a custom attribute for test assertions
    object.__setattr__(mock, "_call_counters", _counters)
    return mock


@asynccontextmanager
async def simulated_mcp_server(
    conn: AcpMcpConnection,
    responses: dict[Any, dict[str, Any]] | None = None,
) -> AsyncIterator[None]:
    """Simulated MCP server that reads from to_session and writes responses to from_session.

    Args:
        conn: The AcpMcpConnection to read from/write to.
        responses: Optional dict mapping request ids to response dicts.
                   If None, returns empty result for every request.
    """
    responses = responses or {}
    task: asyncio.Task | None = None

    async def _server_loop() -> None:
        try:
            async for msg in conn.to_session:
                if isinstance(msg, SessionMessage):
                    msg_dict = msg.message.model_dump(by_alias=True, mode="json", exclude_none=True)
                elif isinstance(msg, dict):
                    msg_dict = msg
                else:
                    continue

                req_id = msg_dict.get("id")

                if req_id is not None:
                    # Return configured response or empty result
                    response = responses.get(req_id, {"jsonrpc": "2.0", "id": req_id, "result": {}})
                    # Directly call send_to_client to trigger fulfill_pending_request
                    await conn.send_to_client(response)
        except (anyio.EndOfStream, anyio.ClosedResourceError):
            pass

    task = asyncio.create_task(_server_loop())
    # CRITICAL: Give the event loop a chance to start the async for loop
    # before any sender tries to write to the zero-buffer stream.
    await asyncio.sleep(0)
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# Direction B: Real ClientSession through AcpMcpTransport
# ---------------------------------------------------------------------------


async def test_client_session_initialize_roundtrip(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Real ClientSession.initialize() completes via AcpMcpTransport.

    Verifies Direction B:
    ClientSession -> from_session -> forwarder -> send_to_client()
    -> _send_to_client() -> response -> to_session -> ClientSession
    """
    mock_send = create_stateful_mock_send_to_client("test-conn-init")

    async def mock_send_request(method: str, params: dict) -> dict:
        if method == "mcp/connect":
            return {"connectionId": "test-conn-init"}
        if method == "mcp/message":
            return await mock_send(params)
        return {}

    acp_agent.client.send_request = mock_send_request  # type: ignore[method-assign]

    connection_id, _session_key = await acp_agent.connect_acp_mcp_server(
        server_config, "test-session-1"
    )
    assert connection_id == "test-conn-init"

    conn = acp_agent._mcp_manager.get_connection(connection_id)
    assert conn is not None

    transport = AcpMcpTransport(conn)
    async with transport.connect_session() as session:
        with anyio.fail_after(5):
            await session.initialize()

    assert mock_send._call_counters["initialize"] >= 1, (
        "initialize request was not sent through _send_to_client"
    )


async def test_client_session_list_tools_roundtrip(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Real ClientSession.list_tools() returns tools via AcpMcpTransport.

    Verifies the complete Direction B flow for tools/list.
    """
    mock_send = create_stateful_mock_send_to_client("test-conn-tools")

    async def mock_send_request(method: str, params: dict) -> dict:
        if method == "mcp/connect":
            return {"connectionId": "test-conn-tools"}
        if method == "mcp/message":
            return await mock_send(params)
        return {}

    acp_agent.client.send_request = mock_send_request  # type: ignore[method-assign]

    connection_id, _session_key = await acp_agent.connect_acp_mcp_server(
        server_config, "test-session-1"
    )
    conn = acp_agent._mcp_manager.get_connection(connection_id)
    assert conn is not None

    transport = AcpMcpTransport(conn)
    async with transport.connect_session() as session:
        with anyio.fail_after(5):
            await session.initialize()
            tools = await session.list_tools()

    assert mock_send._call_counters["tools/list"] >= 1
    assert len(tools.tools) >= 1
    assert tools.tools[0].name == "test_tool"


async def test_client_session_handles_notification_from_server(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """ClientSession sending notifications flows through forwarder gracefully.

    After initialize(), ClientSession may send notifications/initialized.
    This verifies the forwarder handles notifications (no id) without error.
    """
    mock_send = create_stateful_mock_send_to_client("test-conn-notify")
    notification_received = False

    async def mock_send_request(method: str, params: dict) -> dict:
        nonlocal notification_received
        if method == "mcp/connect":
            return {"connectionId": "test-conn-notify"}
        if method == "mcp/message":
            inner = params.get("message", {})
            if isinstance(inner, dict) and inner.get("method") == "notifications/initialized":
                notification_received = True
            return await mock_send(params)
        return {}

    acp_agent.client.send_request = mock_send_request  # type: ignore[method-assign]

    connection_id, _session_key = await acp_agent.connect_acp_mcp_server(
        server_config, "test-session-1"
    )
    conn = acp_agent._mcp_manager.get_connection(connection_id)
    assert conn is not None

    transport = AcpMcpTransport(conn)
    async with transport.connect_session() as session:
        with anyio.fail_after(5):
            await session.initialize()
        # Give the forwarder a moment to process any notifications
        await anyio.sleep(0.1)

    # ClientSession may or may not send notifications/initialized depending on version
    # The test passes as long as no exception is raised


# ---------------------------------------------------------------------------
# Direction A: Correlation Registry with Simulated Server
# ---------------------------------------------------------------------------


async def test_ext_method_blocks_on_client_request(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """ext_method blocks until simulated server responds (not returning {} immediately).

    Verifies Direction A:
    ext_method("mcp/message") -> handle_client_message() -> to_session
    -> simulated server reads -> writes response -> from_session
    -> forwarder -> send_to_client() -> fulfill_pending_request()
    -> ext_method() returns the result
    """
    # Setup: create connection manually (no ACP client involved for Direction A)
    conn = AcpMcpConnection(
        connection_id="conn-block",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-block"] = conn

    request = {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "tools/list",
        "params": {},
    }

    async with simulated_mcp_server(
        conn, responses={"req-1": {"jsonrpc": "2.0", "id": "req-1", "result": {"tools": []}}}
    ):
        result = await acp_agent.ext_method(
            "mcp/message",
            {"connectionId": "conn-block", "message": request},
        )

    assert result == {"tools": []}, (
        f"ext_method returned {result!r} instead of the expected result. "
        "It may have returned {} immediately without waiting for the response."
    )

    await conn.close()


async def test_ext_method_error_response_raises_request_error(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """When simulated server returns an error, ext_method raises RequestError."""
    conn = AcpMcpConnection(
        connection_id="conn-err",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-err"] = conn

    request = {"jsonrpc": "2.0", "id": "req-err", "method": "tools/list", "params": {}}
    error_response = {
        "jsonrpc": "2.0",
        "id": "req-err",
        "error": {"code": -32601, "message": "Method not found", "data": {"detail": "x"}},
    }

    async with simulated_mcp_server(conn, responses={"req-err": error_response}):
        with pytest.raises(RequestError) as exc_info:
            await acp_agent.ext_method(
                "mcp/message",
                {"connectionId": "conn-err", "message": request},
            )

    assert exc_info.value.code == -32601
    assert "Method not found" in str(exc_info.value)

    await conn.close()


async def test_ext_method_response_not_forwarded_to_acp_client(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Response consumed by correlation registry is NOT forwarded to _send_to_client."""
    mock_send = AsyncMock()
    conn = AcpMcpConnection(
        connection_id="conn-no-forward",
        server_config=server_config,
        send_to_client=mock_send,
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-no-forward"] = conn

    request = {"jsonrpc": "2.0", "id": "req-nf", "method": "tools/list", "params": {}}

    async with simulated_mcp_server(
        conn, responses={"req-nf": {"jsonrpc": "2.0", "id": "req-nf", "result": {"tools": []}}}
    ):
        await acp_agent.ext_method(
            "mcp/message",
            {"connectionId": "conn-no-forward", "message": request},
        )

    # _send_to_client should NOT be called for the response (consumed by registry)
    mock_send.assert_not_awaited()

    await conn.close()


async def test_correlation_registry_isolates_concurrent_requests(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Two concurrent requests with different ids get correct respective results."""
    mock_send = AsyncMock()
    conn = AcpMcpConnection(
        connection_id="conn-concurrent",
        server_config=server_config,
        send_to_client=mock_send,
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-concurrent"] = conn

    # Responses returned in REVERSE order to test isolation
    responses = {
        "req-a": {"jsonrpc": "2.0", "id": "req-a", "result": {"tool": "A"}},
        "req-b": {"jsonrpc": "2.0", "id": "req-b", "result": {"tool": "B"}},
    }

    async with simulated_mcp_server(conn, responses=responses):
        # Start both requests concurrently
        task_a = asyncio.create_task(
            acp_agent.ext_method(
                "mcp/message",
                {
                    "connectionId": "conn-concurrent",
                    "message": {"jsonrpc": "2.0", "id": "req-a", "method": "x"},
                },
            )
        )
        task_b = asyncio.create_task(
            acp_agent.ext_method(
                "mcp/message",
                {
                    "connectionId": "conn-concurrent",
                    "message": {"jsonrpc": "2.0", "id": "req-b", "method": "y"},
                },
            )
        )

        result_a, result_b = await asyncio.gather(task_a, task_b)

    assert result_a == {"tool": "A"}
    assert result_b == {"tool": "B"}

    await conn.close()


async def test_handle_server_to_client_message_intercepts_elicitation(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """_handle_server_to_client_message intercepts elicitation/create and uses input provider.

    Verifies that:
    1. Elicitation is handled locally via input provider (not forwarded to ACP client)
    2. Response is sent back to MCP server via handle_client_message (not returned)
    3. Method returns None so send_to_client doesn't forward to _to_session_send
    """
    # Setup connection so handle_client_message can be called
    conn = AcpMcpConnection(
        connection_id="conn-elicit",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-elicit"] = conn

    # Mock the input provider on the default agent
    mock_input_provider = AsyncMock()
    mock_input_provider.get_elicitation = AsyncMock(
        return_value=ElicitResult(action="accept", content={"value": True})
    )
    acp_agent.default_agent._input_provider = mock_input_provider  # type: ignore[attr-defined]

    # Mock the ACP client to verify it's NOT called for elicitation
    mock_client_send = AsyncMock(return_value={})
    acp_agent.client.send_request = mock_client_send  # type: ignore[method-assign]

    wrapped_msg = {
        "connectionId": "conn-elicit",
        "message": {
            "jsonrpc": "2.0",
            "id": "elicit-req-1",
            "method": "elicitation/create",
            "params": {
                "mode": "form",
                "message": "Do you accept?",
                "elicitationId": "elicit-req-1",
            },
        },
    }

    result = await acp_agent._handle_server_to_client_message(wrapped_msg, "conn-elicit")

    # Give handle_client_message time to send
    await asyncio.sleep(0.1)

    # Should return None so send_to_client doesn't forward to _to_session_send
    assert result is None

    # Input provider should be called
    mock_input_provider.get_elicitation.assert_awaited_once()

    # ACP client should NOT be called for elicitation
    mock_client_send.assert_not_awaited()

    # Response should be sent to MCP server (via handle_client_message)
    # The to_session stream should have the JSON-RPC response
    from_session_dict = await asyncio.wait_for(conn.to_session.receive(), timeout=1.0)
    assert from_session_dict["jsonrpc"] == "2.0"
    assert from_session_dict["id"] == "elicit-req-1"
    assert from_session_dict["result"]["action"] == "accept"

    await conn.close()


async def test_handle_server_to_client_message_forwards_non_elicitation(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """_handle_server_to_client_message forwards non-elicitation messages to ACP client."""
    mock_client_send = AsyncMock(return_value={"result": "ok"})
    acp_agent.client.send_request = mock_client_send  # type: ignore[method-assign]

    wrapped_msg = {
        "connectionId": "conn-regular",
        "message": {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/list",
            "params": {},
        },
    }

    result = await acp_agent._handle_server_to_client_message(wrapped_msg, "conn-regular")

    # Should forward to ACP client and return its response
    assert result == {"result": "ok"}
    mock_client_send.assert_awaited_once_with("mcp/message", wrapped_msg)


async def test_handle_server_to_client_message_elicitation_no_input_provider(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """When no input provider is available, elicitation returns decline action."""
    # Ensure no input provider is set
    if hasattr(acp_agent.default_agent, "_input_provider"):
        delattr(acp_agent.default_agent, "_input_provider")

    mock_client_send = AsyncMock(return_value={})
    acp_agent.client.send_request = mock_client_send  # type: ignore[method-assign]

    wrapped_msg = {
        "connectionId": "conn-no-provider",
        "message": {
            "jsonrpc": "2.0",
            "id": "elicit-req-2",
            "method": "elicitation/create",
            "params": {
                "mode": "form",
                "message": "Do you accept?",
            },
        },
    }

    result = await acp_agent._handle_server_to_client_message(wrapped_msg, "conn-no-provider")

    # Should return JSON-RPC response with decline action
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == "elicit-req-2"
    assert result["result"]["action"] == "decline"

    # ACP client should NOT be called
    mock_client_send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Notification Paths
# ---------------------------------------------------------------------------


async def test_ext_method_elicitation_create_bypasses_registry(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """elicitation/create routes to _handle_mcp_elicitation, not correlation registry."""
    conn = AcpMcpConnection(
        connection_id="conn-elicit",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-elicit"] = conn

    # Mock the input provider on the default agent
    mock_input_provider = AsyncMock()
    mock_input_provider.get_elicitation = AsyncMock(
        return_value=ElicitResult(action="accept", content={"value": True})
    )
    acp_agent.default_agent._input_provider = mock_input_provider  # type: ignore[attr-defined]

    message = {
        "jsonrpc": "2.0",
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "Test elicitation",
            "elicitationId": "elicit-1",
            "requestedSchema": {"type": "object"},
        },
    }

    result = await acp_agent.ext_method(
        "mcp/message",
        {"connectionId": "conn-elicit", "message": message},
    )

    # Should return the elicitation result, not block on correlation registry
    assert result.get("action") == "accept"
    mock_input_provider.get_elicitation.assert_awaited_once()

    await conn.close()


async def test_ext_method_elicitation_create_flat_format(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """elicitation/create in flat format (no 'message' wrapper) routes correctly.

    Regression test for production bug where ACP client sends:
    {"connectionId": "...", "method": "elicitation/create", "params": {...}}
    instead of wrapped format:
    {"connectionId": "...", "message": {"method": "elicitation/create", ...}}

    Without this fix, ext_method falls through to notification path and returns {}.
    """
    conn = AcpMcpConnection(
        connection_id="conn-elicit-flat",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-elicit-flat"] = conn

    # Mock the input provider
    mock_input_provider = AsyncMock()
    mock_input_provider.get_elicitation = AsyncMock(
        return_value=ElicitResult(action="accept", content={"confirmed": True})
    )
    acp_agent.default_agent._input_provider = mock_input_provider  # type: ignore[attr-defined]

    # Flat format: no "message" wrapper, MCP fields directly in params
    flat_params = {
        "connectionId": "conn-elicit-flat",
        "jsonrpc": "2.0",
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "功能测试: 您是否确认继续?",
            "elicitationId": "elicit-flat-1",
            "requestedSchema": {"type": "object", "properties": {"confirmed": {"type": "boolean"}}},
        },
        "_meta": {"toolCallId": "call_xxx", "toolCallIdInferred": True},
    }

    result = await acp_agent.ext_method("mcp/message", flat_params)

    # Should route to _handle_mcp_elicitation and return proper result
    assert result.get("action") == "accept"
    assert result.get("content") == {"confirmed": True}
    mock_input_provider.get_elicitation.assert_awaited_once()

    await conn.close()


async def test_elicitation_create_forwarded_to_acp_client(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """MCP server-initiated elicitation/create is forwarded to ACP client via mcp/message.

    This verifies the complete outgoing path:
    MCP server -> from_session -> forwarder -> send_to_client()
    -> _send_to_client({"connectionId": ..., "message": elicitation/create})
    -> ACP client receives the elicitation request
    """
    mock_send = AsyncMock(return_value={"action": "accept"})
    conn = AcpMcpConnection(
        connection_id="conn-elicit-forward",
        server_config=server_config,
        send_to_client=mock_send,
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-elicit-forward"] = conn

    transport = AcpMcpTransport(conn)

    # Simulate MCP server sending elicitation/create to from_session
    elicitation_msg = {
        "jsonrpc": "2.0",
        "id": "elicit-req-1",
        "method": "elicitation/create",
        "params": {
            "mode": "form",
            "message": "Do you accept?",
            "elicitationId": "elicit-req-1",
        },
    }

    async with transport.connect_session():
        # Write elicitation message as if fastmcp server sent it
        await conn.from_session.send(elicitation_msg)

        # Wait for forwarder to process and deliver to _send_to_client
        await anyio.sleep(0.2)

    # Verify _send_to_client was called with the wrapped mcp/message
    mock_send.assert_awaited()
    call_args = mock_send.call_args[0][0]

    # Verify wrapping structure
    assert call_args["connectionId"] == "conn-elicit-forward"
    assert "message" in call_args

    inner_message = call_args["message"]
    assert inner_message["method"] == "elicitation/create"
    assert inner_message["id"] == "elicit-req-1"
    assert inner_message["params"]["message"] == "Do you accept?"

    await conn.close()


async def test_ext_method_notification_fire_and_forget(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Messages without id return {} immediately without registry entry."""
    conn = AcpMcpConnection(
        connection_id="conn-notify",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-notify"] = conn

    message = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    # Should return immediately (not block)
    start = asyncio.get_event_loop().time()
    result = await acp_agent.ext_method(
        "mcp/message",
        {"connectionId": "conn-notify", "message": message},
    )
    elapsed = asyncio.get_event_loop().time() - start

    assert result == {}
    assert elapsed < 0.5, f"Notification took {elapsed}s, should return immediately"

    # No pending requests in registry
    assert len(conn._pending_client_requests) == 0

    # Give background task time to deliver
    await anyio.sleep(0.1)

    await conn.close()


async def test_ext_method_notification_with_null_id(
    acp_agent: AgentPoolACPAgent,
    server_config: AcpMcpServer,
) -> None:
    """Messages with id: null are treated as notifications (fire-and-forget)."""
    conn = AcpMcpConnection(
        connection_id="conn-null-id",
        server_config=server_config,
        send_to_client=AsyncMock(),
    )
    await conn.open()
    acp_agent._mcp_manager._connections["conn-null-id"] = conn

    message = {"jsonrpc": "2.0", "id": None, "method": "notifications/initialized"}

    result = await acp_agent.ext_method(
        "mcp/message",
        {"connectionId": "conn-null-id", "message": message},
    )

    assert result == {}
    assert len(conn._pending_client_requests) == 0

    await conn.close()
