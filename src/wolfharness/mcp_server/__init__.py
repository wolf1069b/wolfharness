"""MCP server integration for AgentPool."""

import pydantic_ai.mcp as _pai_mcp
from wolfharness.mcp_server.client import MCPClient
from wolfharness.mcp_server.tool_bridge import ToolManagerBridge

# --- monkey-patch: prefer content[] over structuredContent for non-JSON MCP results ---
# Some MCP servers (e.g. fixmaster KB) put metadata in `structuredContent` and full
# markdown/XML body in `content[]`. pydantic-ai discards `content` when `structuredContent`
# is present and all parts are TextContent (MCP spec says content should be the JSON text
# form of structuredContent). This patch checks if content text is actually JSON — if not,
# it carries richer information and should be preferred.
_orig = _pai_mcp._map_mcp_call_tool_result


def _patched_map_mcp_call_tool_result(result, *, prefer_structured: bool = False):
    if (
        not prefer_structured
        and result.structured_content is not None
        and result.content
        and all(getattr(p, 'type', None) == 'text' for p in result.content)
        and result.content[0].text.lstrip()[:1] not in ('[', '{')
    ):
        return _pai_mcp._map_mcp_tool_results(result.content)
    return _orig(result, prefer_structured=prefer_structured)


_pai_mcp._map_mcp_call_tool_result = _patched_map_mcp_call_tool_result
# --- end monkey-patch ---

__all__ = ["MCPClient", "ToolManagerBridge"]
