"""Pool server configuration."""

from __future__ import annotations

from typing import Annotated, Literal, assert_never
import warnings

from pydantic import ConfigDict, Field, SecretStr, field_validator
from schemez import Schema


TransportType = Literal["stdio", "sse", "streamable-http"]


class BasePoolServerConfig(Schema):
    """Base configuration for pool servers."""

    type: str = Field(title="Server type")
    """Type discriminator for server configurations."""

    enabled: bool = Field(default=False, title="Server enabled")
    """Whether this server is currently enabled."""


class MCPPoolServerConfig(BasePoolServerConfig):
    """Configuration for pool-based MCP server."""

    type: Literal["mcp"] = Field("mcp", init=False)
    """MCP server type."""

    # Resource exposure control
    serve_nodes: list[str] | bool = Field(
        default=True,
        title="Serve nodes",
        examples=[["node1", "node2"], ["analysis", "transform"]],
    )
    """Which nodes to expose as tools:
    - True: All nodes
    - False: No nodes
    - list[str]: Specific node names
    """

    serve_prompts: list[str] | bool = Field(
        default=True,
        title="Serve prompts",
        examples=[["prompt1", "prompt2"], ["system", "user"]],
    )
    """Which prompts to expose:
    - True: All prompts from manifest
    - False: No prompts
    - list[str]: Specific prompt names
    """

    transport: TransportType = Field(
        default="stdio",
        title="Transport type",
        examples=["stdio", "sse", "streamable-http"],
    )
    """Transport type to use."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to (SSE / Streamable-HTTP only)."""

    port: int = Field(
        default=3001,
        gt=0,
        title="Server port",
        examples=[3001, 8080, 9000],
    )
    """Port to listen on (SSE / Streamable-HTTP only)."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS origins",
        examples=[["*"], ["https://example.com", "https://app.com"]],
    )
    """Allowed CORS origins (SSE / Streamable-HTTP only)."""

    zed_mode: bool = Field(default=False, title="Zed editor mode")
    """Enable Zed editor compatibility mode."""

    model_config = ConfigDict(frozen=True)

    def should_serve_node(self, name: str) -> bool:
        """Check if a node should be exposed."""
        match self.serve_nodes:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_nodes
            case _ as unreachable:
                assert_never(unreachable)

    def should_serve_prompt(self, name: str) -> bool:
        """Check if a prompt should be exposed."""
        match self.serve_prompts:
            case True:
                return True
            case False:
                return False
            case list():
                return name in self.serve_prompts
            case _ as unreachable:
                assert_never(unreachable)


class A2APoolServerConfig(BasePoolServerConfig):
    """Configuration for A2A (Agent-to-Agent) server."""

    type: Literal["a2a"] = Field("a2a", init=False)
    """A2A server type."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to."""

    port: int = Field(
        default=8001,
        gt=0,
        title="Server port",
        examples=[8001, 8080, 9000],
    )
    """Port to listen on."""

    raise_exceptions: bool = Field(
        default=False,
        title="Raise exceptions",
    )
    """Whether to raise exceptions during server start."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS origins",
        examples=[["*"], ["https://example.com", "https://app.com"]],
    )
    """Allowed CORS origins."""

    serve_docs: bool = Field(default=True, title="Serve documentation")
    """Whether to serve API documentation endpoints."""


class ACPPoolServerConfig(BasePoolServerConfig):
    """Configuration for ACP (Agent Client Protocol) server."""

    type: Literal["acp"] = Field("acp", init=False)
    """ACP server type."""

    file_access: bool = Field(
        default=True,
        title="File access",
    )
    """Whether to allow file system access."""

    terminal_access: bool = Field(
        default=True,
        title="Terminal access",
    )
    """Whether to allow terminal/shell access."""

    raise_exceptions: bool = Field(
        default=False,
        title="Raise exceptions",
    )
    """Whether to raise exceptions during server start."""

    subagent_display_mode: Literal["legacy", "zed", "qwen"] = Field(
        default="legacy",
        title="Subagent display mode",
    )
    """How to display nested agent output in ACP clients:
    - "legacy": Original display mode (backward compat for "inline"/"tool_box")
    - "zed": Zed editor optimized display mode
    - "qwen": Qwen compatible display mode
    """

    raw_input_mode: Literal["dict", "skip", "json_str"] = Field(
        default="dict",
        title="Raw input mode",
    )
    """How to emit tool call raw_input in ACP session updates:
    - "dict": Parse args as dict (default; partial JSON returns empty dict)
    - "skip": Omit raw_input until the tool call is complete
    - "json_str": Emit raw_input as a JSON string instead of a dict
    """

    @field_validator("subagent_display_mode", mode="before")
    @classmethod
    def _coerce_legacy_mode(cls, v: str) -> str:
        """Coerce deprecated values to new values with warning."""
        if v in ("inline", "tool_box"):
            warnings.warn(
                f"subagent_display_mode='{v}' is deprecated, use 'legacy' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return "legacy"
        return v

    transport: Literal["stdio", "streamable-http"] = Field(
        default="stdio",
        title="Transport type",
        examples=["stdio", "streamable-http"],
    )
    """Transport type to use."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to (streamable-http only)."""

    port: int = Field(
        default=8080,
        gt=0,
        title="Server port",
        examples=[8080, 9000],
    )
    """Port to listen on (streamable-http only)."""

    websocket_ping_interval: float | None = Field(
        default=60.0,
        gt=0,
        title="WebSocket ping interval",
    )
    """Seconds between server-initiated WebSocket ping frames.

    Set to null to disable server-side WebSocket heartbeat.
    """

    websocket_pong_timeout: float = Field(
        default=30.0,
        gt=0,
        title="WebSocket pong timeout",
    )
    """Seconds to wait for each expected WebSocket pong."""

    websocket_max_missed_pongs: int = Field(
        default=3,
        gt=0,
        title="WebSocket max missed pongs",
    )
    """Consecutive missed WebSocket pongs to tolerate before disconnecting."""

    model_config = ConfigDict(frozen=True)


class AGUIPoolServerConfig(BasePoolServerConfig):
    """Configuration for AGUI (AG-UI) server."""

    type: Literal["agui"] = Field("agui", init=False)
    """AGUI server type."""

    host: str = Field(
        default="localhost",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to."""

    port: int = Field(
        default=8002,
        gt=0,
        title="Server port",
        examples=[8002, 8080, 9000],
    )
    """Port to listen on."""

    raise_exceptions: bool = Field(
        default=False,
        title="Raise exceptions",
    )
    """Whether to raise exceptions during server start."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS origins",
        examples=[["*"], ["https://example.com", "https://app.com"]],
    )
    """Allowed CORS origins."""

    serve_docs: bool = Field(default=True, title="Serve documentation")
    """Whether to serve API documentation endpoints."""


class OpenAIAPIPoolServerConfig(BasePoolServerConfig):
    """Configuration for OpenAI-compatible API server."""

    type: Literal["openai-api"] = Field("openai-api", init=False)
    """OpenAI API server type."""

    host: str = Field(
        default="0.0.0.0",
        title="Server host",
        examples=["localhost", "0.0.0.0", "127.0.0.1"],
    )
    """Host to bind server to."""

    port: int = Field(
        default=8000,
        gt=0,
        title="Server port",
        examples=[8000, 8080, 9000],
    )
    """Port to listen on."""

    cors: bool = Field(default=True, title="Enable CORS")
    """Whether to enable CORS middleware."""

    docs: bool = Field(default=True, title="Enable API docs")
    """Whether to enable API documentation endpoints."""

    api_key: SecretStr | None = Field(
        default=None,
        title="API key",
    )
    """Optional API key for authentication."""

    raise_exceptions: bool = Field(
        default=False,
        title="Raise exceptions",
    )
    """Whether to raise exceptions during server start."""

    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        title="CORS origins",
        examples=[["*"], ["https://example.com", "https://app.com"]],
    )
    """Allowed CORS origins."""


PoolServerConfig = Annotated[
    MCPPoolServerConfig
    | A2APoolServerConfig
    | ACPPoolServerConfig
    | AGUIPoolServerConfig
    | OpenAIAPIPoolServerConfig,
    Field(discriminator="type"),
]
