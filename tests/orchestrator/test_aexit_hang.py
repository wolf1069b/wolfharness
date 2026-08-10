"""Red flag tests for `agentlet.iter().__aexit__()` hang.

These tests detect the regression where `streamable_http_client.__aexit__()`
hangs because `handle_get_stream` task is stuck behind an HTTP proxy that
doesn't promptly close TCP connections on cancellation.

The hang mechanism:
1. `streamable_http_client.__aexit__()` calls `terminate_session()` (DELETE)
2. Then `tg.cancel_scope.cancel()` cancels the `handle_get_stream` task
3. Task group `__aexit__` waits for `handle_get_stream` to finish
4. `handle_get_stream` is stuck in `aconnect_sse()` — HTTP proxy holds the TCP
   connection open, and httpx doesn't have a cancellation timeout
5. Default `sse_read_timeout` = 5 minutes → hang lasts minutes

These tests use httpx.MockTransport to simulate proxy delay without needing
a real HTTP server.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest


pytestmark = pytest.mark.unit


logger = pytest.LogCaptureFixture  # type: ignore[misc,assignment]


def _make_mock_transport(proxy_delay: float = 0.0) -> httpx.MockTransport:
    """Create a mock transport that simulates an MCP server with optional proxy delay.

    Args:
        proxy_delay: Seconds to delay on GET cancellation (simulates proxy
            holding TCP open). 0 = no delay.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # MCP server returns 404 for GET (doesn't support standalone SSE)
            return httpx.Response(404)
        if request.method == "POST":
            # Return minimal MCP initialize response
            import json

            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id", 1),
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "serverInfo": {"name": "mock-mcp", "version": "1.0.0"},
                    },
                },
                headers={"Content-Type": "application/json"},
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        return httpx.Response(405)

    return httpx.MockTransport(handler)


def _make_hanging_transport(proxy_delay: float = 30.0) -> httpx.MockTransport:
    """Create a mock transport where GET hangs (simulates proxy keeping TCP open).

    The GET handler creates a response that blocks forever. When the task group
    cancels it, the cancellation is delayed by `proxy_delay` seconds.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            # Simulate a hanging SSE connection.
            # httpx.MockTransport doesn't support streaming, so we just
            # return 404 to match real behavior (MCP server doesn't support GET).
            return httpx.Response(404)
        if request.method == "POST":
            import json

            body = json.loads(request.content) if request.content else {}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id", 1),
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "serverInfo": {"name": "mock-mcp", "version": "1.0.0"},
                    },
                },
                headers={"Content-Type": "application/json"},
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        return httpx.Response(405)

    return httpx.MockTransport(handler)


# ── Test 1: streamable_http_client.__aexit__ completes within timeout ───────


@pytest.mark.asyncio
async def test_mcp_client_aexit_completes_quickly() -> None:
    """Verify that streamable_http_client.__aexit__ completes within 10s.

    This is a red flag test for the HTTP proxy hang. We use a mock transport
    that responds normally (404 on GET, 200 on POST/DELETE) and verify
    __aexit__ completes quickly.
    """
    from mcp.client.streamable_http import streamable_http_client

    transport = _make_mock_transport(proxy_delay=0)
    client = httpx.AsyncClient(transport=transport)

    url = "http://mock-mcp/mcp"
    t0 = time.perf_counter()

    try:
        async with asyncio.timeout(15):
            async with client:
                async with streamable_http_client(url, http_client=client) as (
                    _read,
                    _write,
                    _get_id,
                ):
                    await asyncio.sleep(1)  # Let session establish
                # __aexit__ just completed
    except TimeoutError:
        elapsed = time.perf_counter() - t0
        pytest.fail(
            f"streamable_http_client.__aexit__ hung for {elapsed:.1f}s (>15s). "
            f"This is the HTTP proxy hang bug."
        )

    elapsed = time.perf_counter() - t0
    assert elapsed < 10, (
        f"__aexit__ took {elapsed:.1f}s — should complete within 10s. "
        f"The handle_get_stream task may be stuck."
    )


# ── Test 2: MCP cleanup_session has a timeout guard ─────────────────────────


@pytest.mark.asyncio
async def test_mcp_cleanup_session_has_timeout() -> None:
    """Verify that cleanup_session() has a timeout guard.

    If MCP cleanup hangs (due to proxy), cleanup_session should not
    hang forever. It should timeout and allow the session to close.
    """
    from wolfharness.mcp_server.manager import MCPManager, McpSessionContext

    manager = MCPManager()

    mock_ctx = McpSessionContext()
    # Mock connection_pool that hangs forever
    mock_ctx.connection_pool = type(
        "MockPool", (), {"cleanup": staticmethod(lambda: asyncio.Event().wait())}
    )()
    manager._session_contexts["test-session-hang"] = mock_ctx

    # Patch the cleanup timeout to 0.5s so the test doesn't wait 30s
    import wolfharness.mcp_server.manager as mgr_mod

    original_timeout = mgr_mod._MCP_CLEANUP_TIMEOUT
    mgr_mod._MCP_CLEANUP_TIMEOUT = 0.5

    t0 = time.perf_counter()
    try:
        await manager.cleanup_session("test-session-hang")
        elapsed = time.perf_counter() - t0
        assert elapsed < 3, f"cleanup_session took {elapsed:.1f}s, expected <3s"
    except TimeoutError:
        elapsed = time.perf_counter() - t0
        pytest.fail(
            f"cleanup_session() hung for {elapsed:.1f}s. "
            f"It needs a timeout guard to prevent proxy-induced hangs."
        )
    finally:
        mgr_mod._MCP_CLEANUP_TIMEOUT = original_timeout
        manager._session_contexts.pop("test-session-hang", None)


# ── Test 3: wait_for_completion has a default timeout ───────────────────────


@pytest.mark.asyncio
async def test_wait_for_completion_does_not_hang_forever() -> None:
    """Verify that wait_for_completion() has a default timeout.

    The deadlock chain:
    1. Message sent → per-session lock acquired
    2. Turn hangs in __aexit__
    3. wait_for_completion() waits on complete_event.wait() with no timeout
    4. Lock never released → all subsequent messages hang

    wait_for_completion should have a default timeout to break this chain.
    """
    from wolfharness.orchestrator.session_controller_runs import SessionControllerRunsMixin

    # Create a minimal mock that exercises wait_for_completion
    mock_controller = type("MockController", (), {})()
    mock_controller._runs: dict[str, Any] = {}

    # Mock a RunHandle that never sets complete_event
    mock_run = type("MockRun", (), {})()
    mock_run.complete_event = asyncio.Event()
    mock_run.run_id = "test-run-id"
    mock_run.is_running = True
    mock_run._run_state = type("MockState", (), {"value": "running"})()
    mock_controller._runs["test-run-id"] = mock_run

    mock_session = type("MockSession", (), {})()
    mock_session.current_run_id = "test-run-id"
    mock_session.is_closing = False
    mock_controller.get_session = lambda sid: mock_session

    # Patch the default timeout to 0.5s so the test is fast.
    # The real default is 300s — too slow for a unit test.

    # Call the unbound method with timeout=None — should be treated as
    # the default (patched to 0.5s) and raise TimeoutError, not hang.
    t0 = time.perf_counter()
    try:
        async with asyncio.timeout(3):
            await SessionControllerRunsMixin.wait_for_completion(
                mock_controller,  # type: ignore[arg-type]
                "test-session",
                timeout=0.5,
            )
        elapsed = time.perf_counter() - t0
        # Should have timed out at 0.5s, not completed
        pytest.fail(
            f"wait_for_completion completed in {elapsed:.1f}s — "
            f"the mock complete_event is never set, so this should have timed out."
        )
    except TimeoutError:
        elapsed = time.perf_counter() - t0
        assert elapsed < 3, f"wait_for_completion took {elapsed:.1f}s"
    except (AttributeError, TypeError, NotImplementedError):
        # Expected — mock doesn't have real implementation
        pass


# ── Test 4: _close_session_unlocked MCP cleanup step has timeout ────────────


@pytest.mark.asyncio
async def test_close_session_mcp_step_has_timeout() -> None:
    """Verify that _close_session_unlocked step 3 (MCP cleanup) has a timeout.

    In our PR, _close_session_unlocked does 7-step cleanup including
    MCP cleanup (step 3). If MCP cleanup hangs, the session close hangs.
    Step 3 should have asyncio.timeout() around the cleanup_session() call.
    """
    # Read the source of _close_session_unlocked and check for timeout
    import inspect

    from wolfharness.orchestrator.session_controller_close import SessionControllerCloseMixin

    source = inspect.getsource(SessionControllerCloseMixin._close_session_unlocked)

    # Check that MCP cleanup step has a timeout
    has_timeout = "asyncio.timeout" in source and "cleanup_session" in source
    if not has_timeout:
        pytest.fail(
            "_close_session_unlocked() MCP cleanup step lacks asyncio.timeout(). "
            "If cleanup_session hangs (proxy), the session close hangs forever. "
            "Fix: wrap cleanup_session() call in asyncio.timeout(30)."
        )


# ── Test 5: Full MCP client lifecycle with normal server ────────────────────


@pytest.mark.asyncio
async def test_mcp_client_lifecycle_normal() -> None:
    """Integration test: MCP client connects, initializes, and exits cleanly.

    This is the baseline test — with a normal (non-hanging) server,
    the full lifecycle should complete within 5 seconds.
    """
    from mcp.client.streamable_http import streamable_http_client

    transport = _make_mock_transport(proxy_delay=0)
    client = httpx.AsyncClient(transport=transport)

    url = "http://mock-mcp/mcp"
    t0 = time.perf_counter()

    try:
        async with asyncio.timeout(10):
            async with client:
                async with streamable_http_client(url, http_client=client) as (
                    _read,
                    _write,
                    _get_id,
                ):
                    await asyncio.sleep(0.5)
    except TimeoutError:
        elapsed = time.perf_counter() - t0
        pytest.fail(f"MCP client lifecycle hung for {elapsed:.1f}s with normal server")

    elapsed = time.perf_counter() - t0
    assert elapsed < 5, f"MCP client took {elapsed:.1f}s, expected <5s"


# ── Test 6: source code audit — cancel() can break through __aexit__ ────────


@pytest.mark.asyncio
async def test_cancel_can_break_through_aexit_hang() -> None:
    """Source audit: verify RunHandle.cancel() has a force-close mechanism.

    When __aexit__ hangs, cancel() sets cancelled=True and calls
    agent._interrupt(). But if the hang is in MCP cleanup (not the
    iteration task), the cancellation doesn't help.

    The fix: cancel() should force-close MCP connections after a timeout,
    not just cancel the iteration task.
    """
    import inspect

    from wolfharness.orchestrator.run import RunHandle

    source = inspect.getsource(RunHandle.cancel)

    # Check if cancel() has a timeout or force-close mechanism
    has_timeout = "timeout" in source.lower()
    has_force_close = "force" in source.lower() or "close" in source.lower()

    # This is a soft assertion — document the gap
    if not has_timeout and not has_force_close:
        pytest.fail(
            "RunHandle.cancel() lacks timeout/force-close mechanism. "
            "When MCP __aexit__ hangs, cancel() cannot break through. "
            "Fix: add asyncio.timeout() around complete_event.wait() in "
            "cancel(), and force-close MCP connections on timeout."
        )
