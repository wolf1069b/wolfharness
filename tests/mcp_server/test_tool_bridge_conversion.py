from __future__ import annotations

from pydantic_ai import ToolReturn
import pytest

from wolfharness.mcp_server.tool_bridge import _convert_to_tool_result


pytestmark = pytest.mark.unit


def test_bridge_preserves_pydantic_tool_return_content() -> None:
    """`ToolReturn` content remains visible after MCP bridge conversion."""
    converted = _convert_to_tool_result(
        ToolReturn(
            return_value="<file>\n00001| graph TD\n00002| A --> B\n</file>",
            metadata={"file_path": "scratchpad://report.md", "read_lines": 2},
        )
    )

    assert converted.content
    assert converted.content[0].text.startswith("<file>\n00001| graph TD")
    assert converted.structured_content == {
        "file_path": "scratchpad://report.md",
        "read_lines": 2,
    }
    assert converted.meta == {"file_path": "scratchpad://report.md", "read_lines": 2}
