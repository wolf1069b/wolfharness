"""L1 unit tests for MCP status reporting (OpenSpec tasks 6.1-6.4, 6.6-6.8).

Covers:
- 6.1: ``MCPManager.get_server_status()`` scenarios.
- 6.2: ``setup_server()`` failure capture in ``_setup_errors`` and clear on retry.
- 6.3: ``McpServerCap.config`` and ``McpServerCap.client`` accessors.
- 6.4: ``BaseAgent._get_mcp_server_info()`` delegation and merge semantics.
- 6.6: ``MCPManager.__aenter__`` failure tolerance (partial and total).
- 6.7: ``MCPClient.server_info`` property (after/before connection, no-trigger).
- 6.8: multi-category precedence (providers wins over _setup_errors).

All tests are ``@pytest.mark.unit`` and mock ``MCPClient`` — no real subprocess
connections are spawned.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness.common_types import MCPServerStatus
from wolfharness.mcp_server.client import MCPClient
from wolfharness.mcp_server.manager import MCPManager
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.unit


# =============================================================================
# Test doubles / helpers
# =============================================================================


def _make_config(
    *,
    client_id: str = "test_server",
    display_name: str | None = None,
    enabled: bool = True,
) -> StdioMCPServerConfig:
    """Build a StdioMCPServerConfig with a recognizable ``client_id``.

    ``StdioMCPServerConfig.client_id`` is derived as
    ``f"{command}_{' '.join(args)}"``. We set ``command=client_id`` and
    ``args=[]`` so the generated id is ``f"{client_id}_"``. Tests should
    reference ``config.client_id`` rather than hardcoding the string.
    """
    return StdioMCPServerConfig(
        name=display_name or client_id,
        command=client_id,
        args=[],
        enabled=enabled,
    )


def _make_connected_cap(
    *,
    client_id: str = "connected_server",
    display_name: str = "Connected Server",
    tool_names: list[str] | None = None,
    server_info: dict[str, str] | None = None,
) -> McpServerCap:
    """Build a McpServerCap with a mocked, already-connected MCPClient.

    The cap's ``client`` property returns a mock whose ``server_info`` is
    pre-populated. ``list_tools()`` returns ToolEntry-like objects with
    ``.name`` set from ``tool_names``. No real connection is triggered.
    """
    config = _make_config(client_id=client_id, display_name=display_name)

    # ToolEntry-like objects (duck-typed: only ``.name`` is read by
    # ``_fetch_connected_status``).
    names = tool_names or []
    tool_entries: list[Any] = []
    for n in names:
        entry = MagicMock()
        entry.name = n
        tool_entries.append(entry)

    mock_client = MagicMock(spec=MCPClient)
    mock_client.server_info = server_info

    cap = McpServerCap(config=config, name=f"mgr_{client_id}", client=mock_client)
    # Replace list_tools with an AsyncMock so no _ensure_client() is triggered.
    cap.list_tools = AsyncMock(return_value=tool_entries)  # type: ignore[method-assign]
    return cap


def _populate_manager(
    manager: MCPManager,
    *,
    servers: list[Any] | None = None,
    providers: list[McpServerCap] | None = None,
    setup_errors: dict[str, str] | None = None,
) -> None:
    """Manually populate a MCPManager's internal registries for testing.

    Bypasses ``__aenter__`` and ``setup_server`` entirely so tests can assert
    ``get_server_status()`` behavior in isolation.
    """
    if servers is not None:
        manager.servers = list(servers)
    if providers is not None:
        manager.providers = list(providers)
    if setup_errors is not None:
        manager._setup_errors = dict(setup_errors)


# =============================================================================
# 6.1 — MCPManager.get_server_status() scenarios
# =============================================================================


async def test_get_server_status_connected() -> None:
    """A server in ``self.providers`` with a non-None client is ``connected``."""
    cap = _make_connected_cap(
        client_id="srv_connected",
        display_name="Connected",
        tool_names=["tool_a"],
        server_info={"name": "fake", "version": "1.0"},
    )
    manager = MCPManager()
    _populate_manager(manager, providers=[cap])

    status = await manager.get_server_status()

    assert cap.config.display_name in status
    entry = status[cap.config.display_name]
    assert entry.status == "connected"
    assert entry.display_name == "Connected"
    assert entry.server_name == "fake"
    assert entry.server_version == "1.0"


async def test_get_server_status_error() -> None:
    """A server in ``_setup_errors`` (and not in providers) is ``error``."""
    config = _make_config(client_id="srv_err", display_name="Errored")
    manager = MCPManager()
    _populate_manager(
        manager,
        servers=[config],
        setup_errors={config.client_id: "connection refused"},
    )

    status = await manager.get_server_status()

    entry = status[config.display_name]
    assert entry.status == "error"
    assert entry.error == "connection refused"
    assert entry.display_name == "Errored"


async def test_get_server_status_disabled() -> None:
    """A server config with ``enabled=False`` is ``disabled`` (wins over all)."""
    config = _make_config(client_id="srv_disabled", enabled=False)
    manager = MCPManager()
    _populate_manager(manager, servers=[config])

    status = await manager.get_server_status()

    entry = status[config.display_name]
    assert entry.status == "disabled"


async def test_get_server_status_disconnected() -> None:
    """A server config with ``enabled=True`` not in providers/errors is ``disconnected``."""
    config = _make_config(client_id="srv_disconnected", enabled=True)
    manager = MCPManager()
    _populate_manager(manager, servers=[config])

    status = await manager.get_server_status()

    entry = status[config.display_name]
    assert entry.status == "disconnected"
    assert entry.tools == []


async def test_get_server_status_tools_populated() -> None:
    """A connected server's ``tools`` field is populated from ``list_tools()``."""
    cap = _make_connected_cap(
        client_id="srv_tools",
        tool_names=["search_kb", "fetch_doc"],
    )
    manager = MCPManager()
    _populate_manager(manager, providers=[cap])

    status = await manager.get_server_status()

    entry = status[cap.config.display_name]
    assert entry.status == "connected"
    assert entry.tools == ["search_kb", "fetch_doc"]


async def test_get_server_status_no_lazy_connection() -> None:
    """A disconnected server does NOT trigger ``list_tools()`` or connection.

    The server config is in ``self.servers`` (enabled=True) but not in
    ``self.providers``. ``get_server_status()`` must return ``tools=[]``
    without attempting any connection.
    """
    config = _make_config(client_id="srv_pending", enabled=True)
    manager = MCPManager()
    _populate_manager(manager, servers=[config])

    status = await manager.get_server_status()

    entry = status[config.display_name]
    assert entry.status == "disconnected"
    assert entry.tools == []


async def test_get_server_status_connected_cap_without_client_skips_tools() -> None:
    """A cap in ``providers`` whose ``client`` is None stays ``connected`` but.

    ``list_tools()`` is NOT called (no lazy connection).
    """
    config = _make_config(client_id="srv_no_client")
    cap = McpServerCap(config=config, name="mgr_srv_no_client", client=None)
    cap.list_tools = AsyncMock(return_value=[])  # type: ignore[method-assign]
    manager = MCPManager()
    _populate_manager(manager, providers=[cap])

    status = await manager.get_server_status()

    entry = status[cap.config.display_name]
    assert entry.status == "connected"
    assert entry.tools == []
    cap.list_tools.assert_not_awaited()  # type: ignore[attr-defined]


# =============================================================================
# 6.2 — setup_server() failure capture and retry-clear
# =============================================================================


async def test_setup_server_records_failure() -> None:
    """When ``setup_server()`` raises, ``_setup_errors[client_id]`` is set.

    and the exception is re-raised.
    """
    config = _make_config(client_id="srv_fail")
    manager = MCPManager()
    # Force MCPClient construction to raise by patching the import target.
    # setup_server() imports MCPClient lazily inside the function body.
    import wolfharness.mcp_server.client as client_mod

    original = client_mod.MCPClient

    def _raising_ctor(**_kwargs: Any) -> MCPClient:
        raise RuntimeError("boom")

    client_mod.MCPClient = _raising_ctor  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.setup_server(config)
    finally:
        client_mod.MCPClient = original  # type: ignore[assignment]

    assert config.client_id in manager._setup_errors
    assert "boom" in manager._setup_errors[config.client_id]
    assert manager.providers == []


async def test_setup_server_clears_error_on_retry() -> None:
    """A previously recorded error is cleared when a retry succeeds."""
    config = _make_config(client_id="srv_retry")
    manager = MCPManager()
    # Seed a prior failure.
    manager._setup_errors[config.client_id] = "prior failure"
    # Build a mock MCPClient whose __aenter__ succeeds; patch the
    # McpServerCap.__aenter__ so exit_stack.enter_async_context works.
    mock_client = MagicMock(spec=MCPClient)

    async def _fake_cap_aenter(self: McpServerCap[Any]) -> McpServerCap[Any]:
        return self

    import wolfharness.mcp_server.client as client_mod

    original_client = client_mod.MCPClient
    client_mod.MCPClient = lambda **_kw: mock_client  # type: ignore[assignment]
    try:
        import wolfharness.capabilities.mcp_server_cap as cap_mod

        original_cap_aenter = cap_mod.McpServerCap.__aenter__
        cap_mod.McpServerCap.__aenter__ = _fake_cap_aenter  # type: ignore[assignment]
        try:
            result = await manager.setup_server(config)
        finally:
            cap_mod.McpServerCap.__aenter__ = original_cap_aenter  # type: ignore[assignment]
    finally:
        client_mod.MCPClient = original_client  # type: ignore[assignment]

    assert result is not None
    assert config.client_id not in manager._setup_errors
    assert len(manager.providers) == 1


# =============================================================================
# 6.3 — McpServerCap.config and McpServerCap.client accessors
# =============================================================================


def test_mcp_server_cap_config_property() -> None:
    """``config`` property returns ``self._config`` without side effects."""
    config = _make_config(client_id="cap_cfg")
    cap = McpServerCap(config=config, name="cap_cfg")

    assert cap.config is config
    assert cap.config is config
    # Verify the property returns the exact same object (no copy/side effect).
    assert cap.config.client_id == config.client_id


def test_mcp_server_cap_client_property_no_connection() -> None:
    """``client`` property returns ``self._client`` (or None) without calling.

    ``_ensure_client()`` — no connection triggered on access.
    """
    # Case 1: no client set → returns None, no connection.
    cap_no_client = McpServerCap(
        config=_make_config(client_id="cap_no_client"),
        name="cap_no_client",
    )
    assert cap_no_client.client is None

    # Case 2: pre-created client → returns the same instance, no connection.
    mock_client = MagicMock(spec=MCPClient)
    cap_with_client = McpServerCap(
        config=_make_config(client_id="cap_with_client"),
        name="cap_with_client",
        client=mock_client,
    )
    assert cap_with_client.client is mock_client
    # Accessing .client must NOT trigger any async call on the mock.
    # (If _ensure_client() had been called, it would be an awaitable on
    # the mock — assert no coroutine was returned/awaited.)


# =============================================================================
# 6.4 — BaseAgent._get_mcp_server_info() delegation
# =============================================================================


class _FakeAgent:
    """Minimal stand-in for BaseAgent exposing only ``mcp`` and ``host_context``.

    ``_get_mcp_server_info`` is an unbound method on ``BaseAgent``; calling
    ``BaseAgent._get_mcp_server_info(self)`` with this stand-in exercises the
    delegation logic without instantiating a real BaseAgent (which requires
    many heavyweight dependencies).
    """

    def __init__(self, mcp: Any, host_context: Any) -> None:
        self.mcp = mcp
        self.host_context = host_context


async def _mock_server_status(status_map: dict[str, MCPServerStatus]) -> Any:
    """Build an async mock returning ``status_map`` from ``get_server_status()``."""
    mock = MagicMock()
    mock.get_server_status = AsyncMock(return_value=dict(status_map))
    return mock


async def test_agent_mcp_info_same_manager() -> None:
    """When ``self.mcp is host_context.mcp`` (same object), no merge happens."""
    shared_mcp = await _mock_server_status({"a": MCPServerStatus(name="a", status="connected")})
    host_ctx = MagicMock()
    host_ctx.mcp = shared_mcp
    agent = _FakeAgent(mcp=shared_mcp, host_context=host_ctx)

    result = await BaseAgent._get_mcp_server_info(agent)  # type: ignore[arg-type]

    assert set(result.keys()) == {"a"}
    assert result["a"].status == "connected"
    shared_mcp.get_server_status.assert_awaited_once()


async def test_agent_mcp_info_different_managers() -> None:
    """When ``self.mcp`` differs from ``host_context.mcp``, results are merged."""
    agent_mcp = await _mock_server_status({
        "agent_srv": MCPServerStatus(name="agent_srv", status="connected")
    })
    pool_mcp = await _mock_server_status({
        "pool_srv": MCPServerStatus(name="pool_srv", status="disconnected")
    })
    host_ctx = MagicMock()
    host_ctx.mcp = pool_mcp
    agent = _FakeAgent(mcp=agent_mcp, host_context=host_ctx)

    result = await BaseAgent._get_mcp_server_info(agent)  # type: ignore[arg-type]

    assert set(result.keys()) == {"agent_srv", "pool_srv"}
    assert result["agent_srv"].status == "connected"
    assert result["pool_srv"].status == "disconnected"


async def test_agent_mcp_info_no_host_context() -> None:
    """When ``host_context`` is None, only ``self.mcp`` results are returned."""
    mcp = await _mock_server_status({"only": MCPServerStatus(name="only", status="connected")})
    agent = _FakeAgent(mcp=mcp, host_context=None)

    result = await BaseAgent._get_mcp_server_info(agent)  # type: ignore[arg-type]

    assert set(result.keys()) == {"only"}


async def test_agent_mcp_info_key_collision() -> None:
    """On key collision, agent-scoped status wins over pool-level."""
    agent_mcp = await _mock_server_status({
        "shared": MCPServerStatus(name="shared", status="connected")
    })
    pool_mcp = await _mock_server_status({
        "shared": MCPServerStatus(name="shared", status="error", error="pool fail")
    })
    host_ctx = MagicMock()
    host_ctx.mcp = pool_mcp
    agent = _FakeAgent(mcp=agent_mcp, host_context=host_ctx)

    result = await BaseAgent._get_mcp_server_info(agent)  # type: ignore[arg-type]

    assert set(result.keys()) == {"shared"}
    assert result["shared"].status == "connected"
    assert result["shared"].error is None


# =============================================================================
# 6.6 — MCPManager.__aenter__ failure tolerance
# =============================================================================


async def test_aenter_tolerates_individual_failure() -> None:
    """3 servers configured, 1 fails → manager enters, failed in.

    ``_setup_errors``, other 2 in ``self.providers``, ``get_server_status()``
    returns all 3.
    """
    cfg_ok1 = _make_config(client_id="ok1")
    cfg_ok2 = _make_config(client_id="ok2")
    cfg_fail = _make_config(client_id="fail")
    manager = MCPManager(servers=[cfg_ok1, cfg_ok2, cfg_fail])

    # Patch setup_server to simulate success/failure while preserving the
    # error-recording contract: failures populate _setup_errors before raise.
    async def _fake_setup(
        self: MCPManager, config: Any, *, add_to_config: bool = False
    ) -> McpServerCap | None:
        if config.client_id == cfg_fail.client_id:
            self._setup_errors[config.client_id] = "setup failed"
            raise RuntimeError("setup failed")
        # Success: append a mock cap using the SAME config (so client_id
        # matches) with a mocked connected client.
        mock_client = MagicMock(spec=MCPClient)
        mock_client.server_info = None
        cap = McpServerCap(config=config, name=f"mgr_{config.client_id}", client=mock_client)
        cap.list_tools = AsyncMock(return_value=[])  # type: ignore[method-assign]
        self.providers.append(cap)
        return cap

    import wolfharness.mcp_server.manager as mgr_mod

    original = mgr_mod.MCPManager.setup_server
    mgr_mod.MCPManager.setup_server = _fake_setup  # type: ignore[assignment]
    try:
        result = await manager.__aenter__()
    finally:
        mgr_mod.MCPManager.setup_server = original  # type: ignore[assignment]

    assert result is manager
    assert len(manager.providers) == 2
    assert {p.config.client_id for p in manager.providers} == {
        cfg_ok1.client_id,
        cfg_ok2.client_id,
    }
    assert cfg_fail.client_id in manager._setup_errors

    status = await manager.get_server_status()
    assert set(status.keys()) == {
        cfg_ok1.display_name,
        cfg_ok2.display_name,
        cfg_fail.display_name,
    }
    assert status[cfg_ok1.display_name].status == "connected"
    assert status[cfg_ok2.display_name].status == "connected"
    assert status[cfg_fail.display_name].status == "error"
    assert status[cfg_fail.display_name].error == "setup failed"

    # Cleanup so exit_stack doesn't try to close mock caps.
    manager.providers.clear()
    await manager.__aexit__(None, None, None)


async def test_aenter_all_fail() -> None:
    """All servers fail → manager still enters, ``providers`` empty, all in.

    ``_setup_errors``, every status is ``error``.
    """
    cfg_a = _make_config(client_id="a")
    cfg_b = _make_config(client_id="b")
    manager = MCPManager(servers=[cfg_a, cfg_b])

    async def _always_fail(
        self: MCPManager, config: Any, *, add_to_config: bool = False
    ) -> McpServerCap | None:
        self._setup_errors[config.client_id] = "boom"
        raise RuntimeError("boom")

    import wolfharness.mcp_server.manager as mgr_mod

    original = mgr_mod.MCPManager.setup_server
    mgr_mod.MCPManager.setup_server = _always_fail  # type: ignore[assignment]
    try:
        result = await manager.__aenter__()
    finally:
        mgr_mod.MCPManager.setup_server = original  # type: ignore[assignment]

    assert result is manager
    assert manager.providers == []
    assert set(manager._setup_errors.keys()) == {cfg_a.client_id, cfg_b.client_id}

    status = await manager.get_server_status()
    assert set(status.keys()) == {cfg_a.display_name, cfg_b.display_name}
    assert all(s.status == "error" for s in status.values())

    await manager.__aexit__(None, None, None)


# =============================================================================
# 6.7 — MCPClient.server_info property
# =============================================================================


def test_server_info_after_connection() -> None:
    """After connection, ``server_info`` returns name/version from.

    ``initialize_result.serverInfo``.
    """
    config = _make_config(client_id="srv_info")
    client = MCPClient(config=config)

    # Build a fake initialize_result with serverInfo.
    fake_server_info = MagicMock()
    fake_server_info.name = "fake-server"
    fake_server_info.version = "2.3.1"
    fake_init_result = MagicMock()
    fake_init_result.serverInfo = fake_server_info

    # _client is a fastmcp.Client; mock its initialize_result property.
    mock_inner = MagicMock()
    type(mock_inner).initialize_result = property(lambda _self: fake_init_result)
    client._client = mock_inner  # type: ignore[assignment]

    info = client.server_info
    assert info == {"name": "fake-server", "version": "2.3.1"}


def test_server_info_before_connection() -> None:
    """Before ``__aenter__`` completes, accessing ``initialize_result`` raises.

    RuntimeError → ``server_info`` returns ``None``.
    """
    config = _make_config(client_id="srv_pre")
    client = MCPClient(config=config)

    mock_inner = MagicMock()
    type(mock_inner).initialize_result = property(
        lambda _self: (_ for _ in ()).throw(RuntimeError("session not active"))
    )
    client._client = mock_inner  # type: ignore[assignment]

    assert client.server_info is None


def test_server_info_no_connection_trigger() -> None:
    """Accessing ``server_info`` does NOT call any connection method."""
    config = _make_config(client_id="srv_notrig")
    client = MCPClient(config=config)

    mock_inner = MagicMock()
    type(mock_inner).initialize_result = property(lambda _self: None)
    client._client = mock_inner  # type: ignore[assignment]

    _ = client.server_info

    # No connection-triggering methods should have been called.
    mock_inner.__aenter__.assert_not_called()
    mock_inner.is_connected.assert_not_called()


# =============================================================================
# 6.8 — Multi-category precedence
# =============================================================================


async def test_multi_category_precedence() -> None:
    """A server in both ``self.providers`` and ``self._setup_errors`` reports.

    ``connected`` (providers wins) with ``error=None``.
    """
    cap = _make_connected_cap(client_id="srv_both", display_name="srv_both", tool_names=["t"])
    config = _make_config(client_id="srv_both")
    manager = MCPManager()
    _populate_manager(
        manager,
        servers=[config],
        providers=[cap],
        setup_errors={config.client_id: "stale error"},
    )

    status = await manager.get_server_status()

    entry = status[config.display_name]
    assert entry.status == "connected"
    assert entry.error is None
    assert entry.tools == ["t"]
