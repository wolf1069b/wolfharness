"""L1 unit tests for ``to_mcp_status`` converter (OpenSpec task 6.5).

Verifies the sync ``to_mcp_status()`` converter reads ``tools`` from
``MCPServerStatus`` and maps ``disabled`` → ``disconnected`` per the
``mcp-status-reporting`` spec.
"""

from __future__ import annotations

import pytest

from wolfharness.common_types import MCPServerStatus
from wolfharness_server.opencode_server.converters import to_mcp_status


pytestmark = pytest.mark.unit


def test_to_mcp_status_reads_tools() -> None:
    """``MCPServerStatus.tools`` is propagated to ``MCPStatus.tools``."""
    status = MCPServerStatus(
        name="kb",
        status="connected",
        display_name="Knowledge Base",
        tools=["search_kb", "fetch_doc"],
    )

    result = to_mcp_status(status)

    assert result.name == "kb"
    assert result.display_name == "Knowledge Base"
    assert result.status == "connected"
    assert result.tools == ["search_kb", "fetch_doc"]
    assert result.error is None


def test_to_mcp_status_disabled_maps_to_disconnected() -> None:
    """Internal ``disabled`` status maps to ``disconnected`` in the OpenCode API."""
    status = MCPServerStatus(name="kb", status="disabled", display_name="KB")

    result = to_mcp_status(status)

    assert result.status == "disconnected"


def test_to_mcp_status_error_propagates_message() -> None:
    """``error`` status surfaces its message in ``MCPStatus.error``."""
    status = MCPServerStatus(
        name="kb",
        status="error",
        display_name="KB",
        error="connection refused",
    )

    result = to_mcp_status(status)

    assert result.status == "error"
    assert result.error == "connection refused"
    assert result.tools == []


def test_to_mcp_status_falls_back_display_name() -> None:
    """When ``display_name`` is None, ``name`` is used as ``display_name``."""
    status = MCPServerStatus(name="fallback", status="connected")

    result = to_mcp_status(status)

    assert result.display_name == "fallback"
