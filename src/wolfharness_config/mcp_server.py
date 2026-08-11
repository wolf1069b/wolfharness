"""MCP server configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

import httpx
from pydantic import ConfigDict, Field, HttpUrl, model_validator
from schemez import Schema


if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp.client import ClientTransport


#: Read timeout (seconds) for MCP HTTP/SSE clients. When the remote server
#: or an intermediate proxy goes silent, the read timeout fires and raises
#: ``httpx.ReadTimeout``, unblocking the task group and preventing deadlock.
_MCP_HTTP_READ_TIMEOUT: float = 60.0


def _expand_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Expand ``${VAR}`` environment variables in all HTTP header values.

    Mirrors the env expansion already performed for skill ``mcp.json``
    companion files (``wolfharness.skills.skill._expand_env_vars_in_value``).
    Without this, header values such as ``Bearer ${API_TOKEN}`` are sent to
    the MCP server as literals, causing auth failures (e.g. ``401``).

    Args:
        headers: Header dict whose values may contain ``${VAR}`` placeholders.

    Returns:
        A new dict with every header value passed through
        ``os.path.expandvars``, or ``None`` if ``headers`` is ``None``.
    """
    if headers is None:
        return None
    return {key: os.path.expandvars(value) for key, value in headers.items()}


def make_mcp_httpx_client_factory(
    read_timeout: float = _MCP_HTTP_READ_TIMEOUT,
) -> Callable[..., httpx.AsyncClient]:
    """Create an httpx client factory for MCP HTTP/SSE transports.

    The factory is compatible with the ``httpx_client_factory`` parameter
    of ``StreamableHttpTransport`` and ``SSETransport``. It wraps the MCP
    library's ``create_mcp_http_client`` but overrides the read timeout
    to ``read_timeout`` seconds (default 60s), which is shorter than the
    MCP default of 300s. This ensures that a silent proxy or unresponsive
    server triggers a ``ReadTimeout`` rather than hanging indefinitely.

    When created from ``BaseMCPServerConfig.to_transport()``, the
    ``read_timeout`` is set to the server's configured ``timeout`` value
    (default 600s), allowing long-running tool calls (e.g., elicitation
    flows that wait for user input) to complete without premature timeout.

    Args:
        read_timeout: Read timeout in seconds.

    Returns:
        A factory callable suitable for ``httpx_client_factory=``.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        if timeout is None:
            timeout = httpx.Timeout(30.0, read=read_timeout)
        else:
            timeout = httpx.Timeout(
                timeout.connect,
                read=read_timeout,
                write=timeout.write,
                pool=timeout.pool,
            )
        # fastmcp passes follow_redirects=True via kwargs; don't duplicate it.
        kwargs.setdefault("follow_redirects", True)
        return httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            auth=auth,
            **kwargs,
        )

    return factory


class MCPServerAuthSettings(Schema):
    """Represents authentication configuration for a server.

    Minimal OAuth v2.1 support with sensible defaults.
    """

    oauth: bool = Field(default=False, title="Enable OAuth")

    # Local callback server configuration
    redirect_port: int = Field(
        default=3030, ge=1, lt=65536, examples=[3030, 8080, 9000], title="Redirect port"
    )
    redirect_path: str = Field(
        default="/callback",
        examples=["/callback", "/auth/callback", "/oauth"],
        title="Redirect path",
    )

    # Optional scope override. If set to a list, values are space-joined.
    scope: str | list[str] | None = Field(
        default=None,
        examples=["read write", ["read", "write"], "admin"],
        title="OAuth scope",
    )

    # Token persistence: use OS keychain via 'keyring' by default; fallback to 'memory'.
    persist: Literal["keyring", "memory"] = Field(
        default="keyring",
        examples=["keyring", "memory"],
        title="Token persistence",
    )


class BaseMCPServerConfig(Schema):
    """Base model for MCP server configuration."""

    type: str = Field(title="Server type")
    """Type discriminator for MCP server configurations."""

    name: str | None = Field(
        default=None,
        examples=["my_server", "api_connector", "file_handler"],
        title="Server name",
    )
    """Optional name for referencing the server."""

    enabled: bool = Field(default=True, title="Server enabled")
    """Whether this server is currently enabled."""

    lazy: bool = Field(default=False, title="Lazy connection")
    """If True, defer MCP server connection until first access (tool/skill/prompt/resource).
    Reduces pool startup time when many MCP servers are configured."""

    env: dict[str, str] | None = Field(default=None, title="Environment variables")
    """Environment variables to pass to the server process."""

    timeout: float = Field(
        default=600.0,
        gt=0,
        examples=[30.0, 60.0, 300.0, 600.0],
        title="Server timeout",
    )
    """Timeout in seconds for both the MCP initialization handshake and per-request
    read timeout (tool calls, elicitation, etc.)."""

    enabled_tools: list[str] | None = Field(
        default=None,
        examples=[["read_file", "list_directory"], ["search", "fetch"]],
        title="Enabled tools",
    )
    """If set, only these tools will be available (whitelist).
    Mutually exclusive with disabled_tools."""

    disabled_tools: list[str] | None = Field(
        default=None,
        examples=[["delete_file", "write_file"], ["dangerous_tool"]],
        title="Disabled tools",
    )
    """Tools to exclude from this server (blacklist). Mutually exclusive with enabled_tools."""

    @model_validator(mode="after")
    def _validate_tool_filters(self) -> Self:
        """Validate that enabled_tools and disabled_tools are mutually exclusive."""
        if self.enabled_tools is not None and self.disabled_tools is not None:
            raise ValueError("Cannot specify both 'enabled_tools' and 'disabled_tools'")
        return self

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed based on enabled/disabled lists.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if the tool is allowed, False otherwise
        """
        if self.enabled_tools is not None:
            return tool_name in self.enabled_tools
        if self.disabled_tools is not None:
            return tool_name not in self.disabled_tools
        return True

    def needs_tool_filtering(self) -> bool:
        """Check if this config has tool filtering configured."""
        return self.enabled_tools is not None or self.disabled_tools is not None

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this MCP server with mcp-filter for tool filtering.

        Creates a new StdioMCPServerConfig that runs mcp-filter as a proxy,
        applying the configured enabled_tools/disabled_tools filtering.

        Returns:
            A new StdioMCPServerConfig that wraps the original server with mcp-filter

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables for the server process."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def to_transport(self, force_oauth: bool = False) -> ClientTransport:
        """Convert to a FastMCP ClientTransport instance.

        Subclasses override this to return the appropriate transport.

        Args:
            force_oauth: If True, force OAuth authentication flow.
        """
        raise NotImplementedError

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this server configuration."""
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        """Return a display name for this server configuration.

        Returns the configured name (stripped of whitespace) if available,
        otherwise falls back to the generated client_id.

        Returns:
            The display name to use for this server.
        """
        return self.name.strip() if self.name and self.name.strip() else self.client_id

    @classmethod
    def from_string(cls, text: str) -> MCPServerConfig:
        """Create a MCPServerConfig from a string."""
        text = text.strip()
        if text.startswith(("http://", "https://")) and text.endswith("/sse"):
            return SSEMCPServerConfig(url=HttpUrl(text))
        if text.startswith(("http://", "https://")):
            return StreamableHTTPMCPServerConfig(url=HttpUrl(text))
        return StdioMCPServerConfig.from_string(text)


class StdioMCPServerConfig(BaseMCPServerConfig):
    """MCP server started via stdio.

    Uses subprocess communication through standard input/output streams.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Stdio MCP Server"})

    type: Literal["stdio"] = Field("stdio", init=False)
    """Stdio server coniguration."""

    command: str = Field(
        examples=["python", "node", "pipx", "uvx"],
        title="Command to execute",
    )
    """Command to execute (e.g. "pipx", "python", "node")."""

    args: list[str] = Field(
        default_factory=list,
        examples=[["run", "mcp-server"], ["-m", "my_mcp_server"], ["--debug"]],
        title="Command arguments",
    )
    """Command arguments (e.g. ["run", "some-server", "--debug"])."""

    @classmethod
    def from_string(cls, text: str) -> Self:
        """Create a MCP server from a command string."""
        parts = text.split(maxsplit=1)
        cmd = parts[0]
        args = parts[1].split() if len(parts) > 1 else []
        return cls(command=cmd, args=args)

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this stdio server configuration."""
        return f"{self.command}_{' '.join(self.args)}"

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this stdio MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "stdio", "--stdio-command", self.command]

        # Add original args as a single --stdio-arg
        if self.args:
            filter_args.extend(["--stdio-arg", " ".join(self.args)])

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            env=self.env,
            timeout=self.timeout,
        )

    def to_transport(self, force_oauth: bool = False) -> ClientTransport:
        """Convert to a FastMCP StdioTransport instance.

        Args:
            force_oauth: If True, raise ValueError since OAuth is not
                supported for stdio transport.

        Raises:
            ValueError: If force_oauth is True.
        """
        if force_oauth:
            raise ValueError("OAuth is not supported for StdioMCPServerConfig")
        from fastmcp.client.transports import StdioTransport

        return StdioTransport(command=self.command, args=self.args, env=self.get_env_vars())


class SSEMCPServerConfig(BaseMCPServerConfig):
    """MCP server using Server-Sent Events transport.

    Connects to a server over HTTP with SSE for real-time communication.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "SSE MCP Server"})

    type: Literal["sse"] = Field("sse", init=False)
    """SSE server configuration."""

    url: HttpUrl = Field(
        examples=["https://api.example.com/sse", "http://localhost:8080/events"],
        title="SSE endpoint URL",
    )
    """URL of the SSE server endpoint."""

    headers: dict[str, str] | None = Field(default=None, title="HTTP headers")
    """Headers to send with the SSE request."""

    auth: MCPServerAuthSettings = Field(
        default_factory=MCPServerAuthSettings,
        title="Authentication settings",
    )
    """OAuth settings for the SSE server."""

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this SSE server configuration."""
        return f"sse_{self.url}"

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this SSE MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "http", "--http-url", str(self.url)]

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            timeout=self.timeout,
        )

    def to_transport(self, force_oauth: bool = False) -> ClientTransport:
        """Convert to a FastMCP SSETransport instance.

        Args:
            force_oauth: Accepted for API compatibility; auth is applied at
                the MCPToolset/Client level, not at the transport level.
        """
        from fastmcp.client import SSETransport

        return SSETransport(
            url=str(self.url),
            headers=_expand_headers(self.headers),
            httpx_client_factory=make_mcp_httpx_client_factory(read_timeout=self.timeout),
        )


class StreamableHTTPMCPServerConfig(BaseMCPServerConfig):
    """MCP server using StreamableHttp.

    Connects to a server over HTTP with streamable HTTP.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "Streamable HTTP MCP Server"})

    type: Literal["streamable-http"] = Field("streamable-http", init=False)
    """HTTP server configuration."""

    url: HttpUrl = Field(
        examples=["https://api.example.com/mcp", "http://localhost:8080/stream"],
        title="HTTP endpoint URL",
    )
    """URL of the HTTP server endpoint."""

    headers: dict[str, str] | None = Field(default=None, title="HTTP headers")
    """Headers to send with the HTTP request."""

    auth: MCPServerAuthSettings = Field(
        default_factory=MCPServerAuthSettings,
        title="Authentication settings",
    )
    """OAuth settings for the HTTP server."""

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this streamable HTTP server configuration."""
        return f"streamable_http_{self.url}"

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this HTTP MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "http", "--http-url", str(self.url)]

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            timeout=self.timeout,
        )

    def to_transport(self, force_oauth: bool = False) -> ClientTransport:
        """Convert to a FastMCP StreamableHttpTransport instance.

        Args:
            force_oauth: Accepted for API compatibility; auth is applied at
                the MCPToolset/Client level, not at the transport level.
        """
        from fastmcp.client import StreamableHttpTransport

        return StreamableHttpTransport(
            url=str(self.url),
            headers=_expand_headers(self.headers),
            httpx_client_factory=make_mcp_httpx_client_factory(read_timeout=self.timeout),
        )


class AcpMCPServerConfig(BaseMCPServerConfig):
    """MCP server using ACP channel transport.

    Connects to a server over the existing ACP connection.
    """

    model_config = ConfigDict(json_schema_extra={"x-doc-title": "ACP MCP Server"})

    type: Literal["acp"] = Field("acp", init=False)
    """ACP server configuration."""

    acp_id: str = Field(
        examples=["uuid-xxx", "server-123"],
        title="ACP server ID",
    )
    """Unique identifier for the ACP-transport MCP server."""

    @property
    def client_id(self) -> str:
        """Generate a unique client ID for this ACP server configuration."""
        return f"acp_{self.acp_id}"

    def wrap_with_mcp_filter(self) -> StdioMCPServerConfig:
        """Wrap this ACP MCP server with mcp-filter for tool filtering.

        Returns:
            A new StdioMCPServerConfig that wraps this server with mcp-filter
        """
        filter_args = ["mcp-filter", "run", "-t", "acp", "--acp-id", self.acp_id]

        # Add allowlist (exact tool names)
        if self.enabled_tools:
            filter_args.extend(["-a", ",".join(self.enabled_tools)])

        # Add denylist (regex patterns)
        if self.disabled_tools:
            filter_args.extend(["-d", ",".join(self.disabled_tools)])

        return StdioMCPServerConfig(
            name=self.name,
            command="uvx",
            args=filter_args,
            timeout=self.timeout,
        )


MCPServerConfig = Annotated[
    StdioMCPServerConfig | SSEMCPServerConfig | StreamableHTTPMCPServerConfig | AcpMCPServerConfig,
    Field(discriminator="type"),
]


def parse_mcp_servers_json(data: dict[str, object]) -> list[MCPServerConfig]:
    """Parse MCP servers from JSON format used by clients (e.g., Zed).

    Expected format:
        {
            "mcpServers": {
                "server_name": {
                    "url": "https://...",
                    "transport": "sse" | "http"  # optional, defaults to http
                },
                "stdio_server": {
                    "command": "python",
                    "args": ["-m", "my_server"]
                },
                ...
            }
        }

    Args:
        data: JSON data containing mcpServers key

    Returns:
        List of parsed MCPServerConfig instances

    Raises:
        ValueError: If data format is invalid or transport type unsupported
    """
    if "mcpServers" not in data:
        raise ValueError("MCP config must contain 'mcpServers' key")

    servers: list[MCPServerConfig] = []
    mcp_servers = data["mcpServers"]
    if not isinstance(mcp_servers, dict):
        raise TypeError("'mcpServers' must be an object")
    for server_name, server_cfg in mcp_servers.items():
        assert isinstance(server_name, str)
        assert isinstance(server_cfg, dict)
        match server_cfg:
            case {"command": str(command), **rest}:
                server: MCPServerConfig = StdioMCPServerConfig(
                    name=server_name,
                    command=command,
                    args=rest.get("args", []),
                    env=rest.get("env"),
                )
            case {"transport": "sse", "url": url}:
                server = SSEMCPServerConfig(name=server_name, url=url)
            case {"transport": "http", "url": url} | {"url": url}:  # Default to HTTP
                server = StreamableHTTPMCPServerConfig(name=server_name, url=url)
            case {"transport": "acp", "id": acp_id} | {"id": acp_id}:  # ACP transport requires id
                server = AcpMCPServerConfig(name=server_name, acp_id=acp_id)
            case {"transport": unknown}:
                raise ValueError(
                    f"Unsupported transport type for '{server_name}': {unknown}. "
                    f"Supported transports: stdio, sse, http, acp"
                )
            case _:
                raise ValueError(f"Invalid config for MCP server '{server_name}': {server_cfg}")

        servers.append(server)

    return servers
