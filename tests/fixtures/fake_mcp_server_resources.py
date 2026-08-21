"""Fake MCP server exposing resources for AgentPool L4 tests."""

from __future__ import annotations

from fastmcp import FastMCP


mcp = FastMCP("fake-resource-server", version="0.1.0")


@mcp.tool()
def search_kb(query: str) -> str:
    """Return a deterministic search result for the requested query."""
    return f"Results for: {query}"


@mcp.resource("file:///shared.txt", mime_type="text/plain")
def shared_text() -> str:
    """Return the shared text resource."""
    return "<div>shared resource text</div>"


@mcp.resource("file:///image.png", mime_type="image/png")
def image_resource() -> bytes:
    """Return a small PNG-shaped binary resource."""
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


@mcp.resource("file:///{path}", mime_type="text/plain")
def templated_file(path: str) -> str:
    """Return a deterministic dynamic resource."""
    return f"template:{path}"


if __name__ == "__main__":
    mcp.run()
