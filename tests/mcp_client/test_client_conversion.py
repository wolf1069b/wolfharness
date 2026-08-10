"""Integration test for MCP client conversion functionality.

Tests that our MCP client properly converts FastMCP server responses
to PydanticAI-compatible return types without mocks.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
from mcp.types import TextContent
from pydantic_ai import BinaryContent, RunContext, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
import pytest

from wolfharness.mcp_server import MCPClient
from wolfharness_config.mcp_server import StdioMCPServerConfig


pytestmark = pytest.mark.unit


@pytest.fixture
async def mcp_client():
    """Create MCP client connected to test server."""
    server_path = Path(__file__).parent / ".." / "mcp_server" / "server.py"

    config = StdioMCPServerConfig(
        name="test_server",
        command="uv",
        args=["run", str(server_path)],
    )

    client = MCPClient(config)
    async with client:
        # Wait for server to be ready
        await anyio.sleep(0.5)
        yield client


async def test_rich_content_image(mcp_client: MCPClient):
    """Test that FastMCP Image content is converted to PydanticAI types."""
    ctx = RunContext(
        tool_call_id="test-call-123",
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
    )
    result = await mcp_client.call_tool(
        "test_rich_content",
        run_context=ctx,
        arguments={"content_type": "image"},
    )
    assert isinstance(result, ToolReturn)
    assert isinstance(result.return_value, list)
    assert isinstance(result.return_value[0], BinaryContent)
    result = await mcp_client.call_tool(
        "test_rich_content",
        run_context=ctx,
        arguments={"content_type": "audio"},
    )
    assert isinstance(result, ToolReturn)
    assert isinstance(result.return_value, list)
    assert isinstance(result.return_value[0], BinaryContent)
    assert result.return_value[0].media_type == "audio/wav"
    result = await mcp_client.call_tool(
        "test_rich_content",
        run_context=ctx,
        arguments={"content_type": "file"},
    )
    assert result is not None
    result = await mcp_client.call_tool(
        "test_rich_content",
        run_context=ctx,
        arguments={"content_type": "mixed"},
    )
    assert result is not None


async def test_structured_mcp_result_preserves_text_as_tool_return() -> None:
    """MCP text content must remain visible as the tool result when metadata exists."""
    ctx = RunContext(
        tool_call_id="test-call-structured",
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
    )
    client = MCPClient(StdioMCPServerConfig(name="test_server", command="uv", args=["--version"]))
    client._client = MagicMock()
    client._client.is_connected.return_value = True
    client._client.call_tool = AsyncMock(
        return_value=SimpleNamespace(
            is_error=False,
            content=[TextContent(type="text", text="<file>\n00001| report body\n</file>")],
            data={"file_path": "reports/case.md", "total_lines": 1},
        )
    )

    result = await client.call_tool(
        "read",
        run_context=ctx,
        arguments={"file_path": "reports/case.md"},
    )

    assert isinstance(result, ToolReturn)
    assert result.return_value == "<file>\n00001| report body\n</file>"
    assert result.metadata == {"file_path": "reports/case.md", "total_lines": 1}


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
