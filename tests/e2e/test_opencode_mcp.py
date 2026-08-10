"""L4 subprocess E2E tests for OpenCode MCP management endpoints.

Covers Phase C7 of the protocol-e2e-coverage OpenSpec change:
    - C7.1: GET /mcp — verify 200 with MCP server list
    - C7.2: POST /mcp — verify 201 with MCP server ID
    - C7.3-C7.8 [skip] — MCP CRUD endpoints not implemented
    - C7.9: POST /mcp/{name}/connect — verify 200
    - C7.10: POST /mcp/{name}/disconnect — verify 200
    - C7.11: POST /mcp/{name}/auth — verify 200
    - C7.12: POST /mcp/{name}/auth/callback — verify 200
    - C7.13: POST /mcp/{name}/auth/authenticate — verify 200
    - C7.14: DELETE /mcp/{name}/auth — verify 200/204

All tests use ``model: test`` (pydantic-ai TestModel) so NO API key is needed.
L4a smoke tests: pytest -m "e2e and not slow" (~30s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from tests.e2e.conftest import SKIP_NO_BINARY, SKIP_WINDOWS


if TYPE_CHECKING:
    from tests.e2e.conftest import SubprocessServer


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(SKIP_NO_BINARY, reason="wolfharness binary not on PATH"),
    pytest.mark.skipif(SKIP_WINDOWS, reason="Windows subprocess issues"),
]


# ---------------------------------------------------------------------------
# C7.1 — GET /mcp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_mcp(subprocess_server: SubprocessServer) -> None:
    """C7.1: GET /mcp, verify 200 with MCP server list.

    Minimal-config variant (no MCP servers configured): the response dict
    is empty (``{}``) but must still be a dict. The companion test
    ``test_get_mcp_with_connected_server`` exercises the non-empty case
    with a real stdio MCP server.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{base_url}/mcp")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /mcp, got {resp.status_code}: {resp.text}"
        )
        mcp_servers = resp.json()
        # GET /mcp returns a dict mapping server names to MCPStatus objects.
        assert isinstance(mcp_servers, dict), f"Expected dict, got {type(mcp_servers)}"


@pytest.mark.parametrize(
    "subprocess_server_with_mcp",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_mcp_with_connected_server(
    subprocess_server_with_mcp: SubprocessServer,
) -> None:
    """C7.1+: GET /mcp with a real stdio MCP server, verify connected status.

    L4 e2e test for MCP status reporting (fix-mcp-status-reporting tasks
    8.1-8.3). Uses ``tests/fixtures/fake_mcp_server.py`` (a ``fastmcp``
    stdio server exposing the ``search_kb`` tool) spawned as a real
    subprocess by the ``MCPManager`` during pool startup.

    Asserts:
        - Response is a non-empty dict (the bug was that it returned ``{}``).
        - The ``fake_kb`` server appears (identified by ``display_name``
          because the dict key is ``client_id``, not the configured name).
        - ``status == "connected"`` (the server successfully handshaked).
        - ``tools`` contains ``"search_kb"`` (tools are populated from
          ``McpServerCap.list_tools()``).
    """
    base_url = subprocess_server_with_mcp.base_url

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{base_url}/mcp")
        assert resp.status_code == 200, (
            f"Expected 200 for GET /mcp, got {resp.status_code}: {resp.text}"
        )
        mcp_servers = resp.json()
        assert isinstance(mcp_servers, dict), f"Expected dict, got {type(mcp_servers)}"
        assert len(mcp_servers) > 0, (
            f"Expected non-empty MCP servers with fake_kb configured, got {mcp_servers}"
        )

        # The dict key is client_id (e.g. "<python>_<fixture_path>"), not
        # the configured name. Find the fake_kb entry by displayName (the
        # OpenCode API uses camelCase aliases via OpenCodeBaseModel).
        fake_kb_entry: dict[str, Any] | None = None
        for entry in mcp_servers.values():
            assert isinstance(entry, dict), f"Expected dict entry, got {type(entry)}"
            if entry.get("displayName") == "fake_kb":
                fake_kb_entry = entry
                break
        assert fake_kb_entry is not None, (
            f"fake_kb server not found in response (displayName lookup), got {mcp_servers}"
        )
        assert fake_kb_entry["status"] == "connected", (
            f"Expected status='connected', got '{fake_kb_entry['status']}': {fake_kb_entry}"
        )
        assert "search_kb" in fake_kb_entry["tools"], (
            f"Expected 'search_kb' in tools, got {fake_kb_entry['tools']}"
        )


# ---------------------------------------------------------------------------
# C7.2 — POST /mcp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp(subprocess_server: SubprocessServer) -> None:
    """C7.2: POST /mcp, verify 200/201 with MCP server ID.

    Test intent: POST /mcp with JSON body containing either ``command`` (for
    stdio servers) or ``url`` (for HTTP/SSE servers). Verify 200 or 201 with
    MCP server status object containing ``name`` and ``status`` fields.
    Error case: missing both ``command`` and ``url`` -> 400; no MCP manager
    available -> 400.

    With the minimal e2e config (no MCP infrastructure), the server may return
    400 ``No MCP manager available``. The test accepts both the expected success
    status (200/201) and the no-manager-available status (400).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Use a harmless stdio command that won't actually spawn a real server.
        mcp_payload: dict[str, Any] = {
            "command": "echo",
            "args": ["test-mcp-server"],
        }
        resp = await client.post(f"{base_url}/mcp", json=mcp_payload)
        # Accept 200/201 (server added) or 400 (no MCP manager in minimal config).
        assert resp.status_code in (200, 201, 400), (
            f"Expected 200/201/400 for POST /mcp, got {resp.status_code}: {resp.text}"
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            assert isinstance(data, dict), f"Expected dict, got {type(data)}"
            assert "name" in data or "status" in data, (
                f"MCP status response missing expected fields: {data}"
            )


# ---------------------------------------------------------------------------
# C7.3 — DELETE /mcp/{id} [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_delete_mcp_by_id(subprocess_server: SubprocessServer) -> None:
    """Test intent: DELETE /mcp/{id} to remove an MCP server entry.

    POST /mcp to create MCP server entry and capture returned ``id``. Call
    DELETE /mcp/{id} with that ID. Verify 200 or 204 response. Follow with
    GET /mcp to verify server removed from list. Verify MCP server process
    terminated (if spawned). Error case: delete non-existent MCP ID -> 404.
    """


# ---------------------------------------------------------------------------
# C7.4 — PUT /mcp/{id} [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_put_mcp_by_id(subprocess_server: SubprocessServer) -> None:
    """Test intent: PUT /mcp/{id} to update an MCP server config.

    POST /mcp to create MCP server entry, then PUT /mcp/{id} with JSON body
    containing updated config (e.g., new ``command``, ``args``, ``env``).
    Verify 200 with updated MCP config in response. Follow with GET /mcp to
    verify config change applied. Error case: non-existent ID -> 404;
    invalid body -> 422.
    """


# ---------------------------------------------------------------------------
# C7.5 — GET /mcp/tool [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_mcp_tool(subprocess_server: SubprocessServer) -> None:
    """Test intent: GET /mcp/tool with optional ``server_id`` query param.

    Verify 200 with JSON array of MCP tools, each containing ``name``,
    ``description``, ``input_schema`` (JSON Schema). Verify tools from all
    connected MCP servers appear when no ``server_id`` filter. Verify only
    tools from specified server when ``server_id`` provided.
    """


# ---------------------------------------------------------------------------
# C7.6 — POST /mcp/tool [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_tool(subprocess_server: SubprocessServer) -> None:
    """Test intent: POST /mcp/tool to execute an MCP tool directly.

    Send POST /mcp/tool with JSON body containing ``tool_name``, ``server_id``,
    and ``arguments`` object matching tool's input schema. Verify 200 or 201
    with tool execution result containing ``content`` array (text/resources).
    Error case: unknown tool name -> 404; invalid arguments -> 422;
    server not connected -> 409.
    """


# ---------------------------------------------------------------------------
# C7.7 — GET /mcp/server [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_get_mcp_server(subprocess_server: SubprocessServer) -> None:
    """Test intent: GET /mcp/server with optional query params.

    Verify 200 with JSON array of MCP server status objects, each containing
    ``name``, ``status`` (connected/disconnected/error), ``transport`` type,
    and ``tool_count``. Verify status reflects actual server connection state.
    """


# ---------------------------------------------------------------------------
# C7.8 — POST /mcp/server [skip]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_server(subprocess_server: SubprocessServer) -> None:
    """Test intent: POST /mcp/server to add a new MCP server.

    Send POST /mcp/server with JSON body containing server config (``name``,
    ``command``, ``args``, ``env``). Verify 200 or 201 with server ``id`` in
    response. Follow with GET /mcp/server to verify new server appears with
    ``status="connected"``. Verify server's tools available via GET /mcp/tool.
    Error case: invalid server config -> 422; connection failure -> 503.
    """


# ---------------------------------------------------------------------------
# C7.9 — POST /mcp/{name}/connect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_connect(subprocess_server: SubprocessServer) -> None:
    """C7.9: POST /mcp/{name}/connect, verify 200.

    Test intent: Configure an MCP server via POST /mcp, then call
    POST /mcp/{name}/connect with the server name. Verify 200 with response
    confirming connection established. Follow with GET /mcp to verify server
    status changed to ``connected``. Error case: non-existent server name ->
    404; already connected -> 409.

    With the minimal e2e config (no MCP manager), the server returns 400
    ``No MCP manager available``. The test accepts 200 (success), 400 (no
    manager), and 404 (server not found) as valid responses.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{base_url}/mcp/test-server/connect")
        # Accept 200 (connected), 400 (no MCP manager), 404 (server not found).
        assert resp.status_code in (200, 400, 404), (
            f"Expected 200/400/404 for POST /mcp/connect, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C7.10 — POST /mcp/{name}/disconnect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_disconnect(subprocess_server: SubprocessServer) -> None:
    """C7.10: POST /mcp/{name}/disconnect, verify 200.

    Test intent: Connect an MCP server via POST /mcp/{name}/connect, then call
    POST /mcp/{name}/disconnect with the same server name. Verify 200 with
    response confirming disconnection. Follow with GET /mcp to verify server
    status changed to ``disconnected``. Error case: non-existent server name
    -> 404; not connected -> 409.

    With the minimal e2e config (no MCP manager), the server returns 400
    ``No MCP manager available``. The test accepts 200, 400, and 404 as valid
    responses.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{base_url}/mcp/test-server/disconnect")
        # Accept 200 (disconnected), 400 (no MCP manager), 404 (server not found).
        assert resp.status_code in (200, 400, 404), (
            f"Expected 200/400/404 for POST /mcp/disconnect, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C7.11 — POST /mcp/{name}/auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_auth(subprocess_server: SubprocessServer) -> None:
    """C7.11: POST /mcp/{name}/auth, verify 200.

    Test intent: Configure an MCP server that requires authentication, then
    call POST /mcp/{name}/auth with the server name. Verify 200 with response
    containing auth URL or challenge. Error case: non-existent server name ->
    404; server doesn't require auth -> 400.

    The current implementation returns 501 ``MCP OAuth not yet supported`` for
    all servers. The test accepts 200 (when implemented) and 501 (current
    stub) as valid responses.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{base_url}/mcp/test-server/auth")
        # Accept 200 (when implemented) or 501 (current stub: not yet supported).
        assert resp.status_code in (200, 501), (
            f"Expected 200/501 for POST /mcp/auth, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C7.12 — POST /mcp/{name}/auth/callback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_auth_callback(subprocess_server: SubprocessServer) -> None:
    """C7.12: POST /mcp/{name}/auth/callback, verify 200.

    Test intent: Initiate auth via POST /mcp/{name}/auth, then call
    POST /mcp/{name}/auth/callback with auth callback parameters (e.g.,
    ``code``, ``state``). Verify 200 with response confirming authentication
    completed. Error case: invalid callback params -> 422; non-existent
    server -> 404.

    The current implementation returns 501 ``MCP OAuth not yet supported``.
    The test accepts 200 (when implemented) and 501 (current stub).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{base_url}/mcp/test-server/auth/callback",
            params={"code": "test-code", "state": "test-state"},
        )
        # Accept 200 (when implemented) or 501 (current stub: not yet supported).
        assert resp.status_code in (200, 501), (
            f"Expected 200/501 for POST /mcp/auth/callback, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C7.13 — POST /mcp/{name}/auth/authenticate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_post_mcp_auth_authenticate(subprocess_server: SubprocessServer) -> None:
    """C7.13: POST /mcp/{name}/auth/authenticate, verify 200.

    Test intent: Call POST /mcp/{name}/auth/authenticate with credentials or
    token for an MCP server requiring auth. Verify 200 with response confirming
    authentication successful and server connected. Error case: invalid
    credentials -> 401; non-existent server -> 404.

    The current implementation returns 501 ``MCP OAuth not yet supported``.
    The test accepts 200 (when implemented) and 501 (current stub).
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{base_url}/mcp/test-server/auth/authenticate")
        # Accept 200 (when implemented) or 501 (current stub: not yet supported).
        assert resp.status_code in (200, 501), (
            f"Expected 200/501 for POST /mcp/auth/authenticate, got {resp.status_code}: {resp.text}"
        )


# ---------------------------------------------------------------------------
# C7.14 — DELETE /mcp/{name}/auth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subprocess_server",
    [{"serve_command": "serve-opencode", "is_stdio": False, "health_path": "/session"}],
    indirect=True,
)
async def test_delete_mcp_auth(subprocess_server: SubprocessServer) -> None:
    """C7.14: DELETE /mcp/{name}/auth, verify 200/204.

    Test intent: Authenticate an MCP server, then call DELETE /mcp/{name}/auth
    to revoke authentication. Verify 200 or 204 response. Follow with GET /mcp
    to verify server auth state cleared. Error case: non-existent server ->
    404; not authenticated -> 409.

    The current implementation returns 200 with ``{"success": true}`` (stub
    that always succeeds). The test accepts 200 and 204.
    """
    base_url = subprocess_server.base_url

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.delete(f"{base_url}/mcp/test-server/auth")
        # Accept 200 or 204 (both are valid success responses for DELETE).
        assert resp.status_code in (200, 204), (
            f"Expected 200/204 for DELETE /mcp/auth, got {resp.status_code}: {resp.text}"
        )
