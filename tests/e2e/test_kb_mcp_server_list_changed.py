"""E2E: kb_mcp_server_example.py emits resources/list_changed -> McpServerCap ChangeEvent.

L4 subprocess test (``@pytest.mark.e2e``): spins up the real example MCP server
as a subprocess pointed at a temp ``--kb-dir``, connects to it through
``McpServerCap`` (via ``SessionConnectionPool``), then adds a file on disk and
asserts the capability surfaces a ``ChangeEvent(kind="resource_list_changed")``.

This proves the full best-practice chain works:
server scan loop -> notifications/resources/list_changed -> MCPMessageHandler ->
MCPClient.set_notification_callbacks -> McpServerCap.on_change().
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import pytest

from wolfharness.capabilities.mcp_server_cap import McpServerCap
from wolfharness.mcp_server.session_pool import SessionConnectionPool
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.e2e

_SERVER_PATH = (
    Path(__file__).parent / ".." / ".." / "examples" / "kb_mcp_server_example.py"
).resolve()
_SCAN_INTERVAL = 1.0


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """A temp knowledge base seeded with one doc and one image namespace."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "docs" / "intro.md").write_text("# Intro\nhello kb", encoding="utf-8")
    return tmp_path


def _server_config(kb_dir: Path) -> StdioMCPServerConfig:
    return StdioMCPServerConfig(
        name="kb-server",
        command=sys.executable,
        args=[
            str(_SERVER_PATH),
            "--kb-dir",
            str(kb_dir),
            "--scan-interval",
            str(_SCAN_INTERVAL),
        ],
    )


@pytest.mark.asyncio
async def test_list_changed_event_flows_to_mcp_server_cap(kb_dir: Path) -> None:
    """Adding a file to kb_data/ yields resource_list_changed on McpServerCap."""
    pool = SessionConnectionPool(session_id="kb-list-changed-test")
    cap = McpServerCap(config=_server_config(kb_dir), name="kb-server", session_pool=pool)

    # Trigger lazy connection so the notification callbacks are wired.
    resources = await cap.list_resources()
    assert any(str(r.uri).startswith("kb://docs/") for r in resources)

    stream = cap.on_change()
    assert stream is not None

    try:
        # Add a file: the server's scan loop detects it, broadcasts
        # notifications/resources/list_changed, and the capability emits a
        # ChangeEvent.
        (kb_dir / "docs" / "new-doc.md").write_text("# New doc", encoding="utf-8")
        event = await asyncio.wait_for(stream.__anext__(), timeout=20.0)
    finally:
        await stream.aclose()

    assert event.kind == "resource_list_changed"
    assert event.capability_name == "kb-server"
    assert event.source_uri == "mcp://kb-server"
