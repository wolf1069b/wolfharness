"""MCP server bridge for exposing ToolManager tools to ACP agents.

This module provides a bridge that exposes a ToolManager's tools as an MCP server
using HTTP transport. This allows ACP agents (external agents like Claude Code,
Gemini CLI, etc.) to use our internal toolsets like SubagentTools,
AgentManagementTools, etc.

The bridge runs in-process on the same event loop, providing direct access to
the pool and avoiding IPC serialization overhead.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
import inspect
import time
from typing import TYPE_CHECKING, Any, Self, get_args, get_origin
from uuid import uuid4

import anyio
from pydantic import BaseModel
from pydantic_ai import RunContext, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from wolfharness.agents import Agent
from wolfharness.capabilities.change_event import ChangeEvent
from wolfharness.log import get_logger
from wolfharness.utils.signatures import filter_schema_params, get_params_matching_predicate


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from fastmcp import Context, FastMCP
    from fastmcp.tools.tool import ToolResult as FastMCPToolResult
    from pydantic_ai.messages import UserContent
    from uvicorn import Server

    from wolfharness.agents import AgentContext
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.tools.base import Tool
_ = ChangeEvent  # Used at runtime in method signature


logger = get_logger(__name__)


def _is_annotation_of_type(annotation: Any, type_name: str) -> bool:
    """Check if annotation matches the given type name.

    Handles direct types, generics, forward references, and unions.
    """
    if annotation is None or annotation is inspect.Parameter.empty:
        return False
    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        base_name = annotation.split("[")[0].strip()
        return base_name == type_name
    # Check direct class match by name
    if isinstance(annotation, type) and annotation.__name__ == type_name:
        return True
    # Check generic origin (e.g., SomeType[T])
    origin = get_origin(annotation)
    if origin is not None:
        if isinstance(origin, type) and origin.__name__ == type_name:
            return True
        # Handle Union types (e.g., SomeType | None)
        if origin is type(None) or str(origin) in ("typing.Union", "types.UnionType"):
            return any(_is_annotation_of_type(arg, type_name) for arg in get_args(annotation))
    return False


def _get_context_param_names(fn: Callable[..., Any], type_name: str) -> set[str]:
    """Get parameter names that are AgentContext (to be auto-injected)."""
    return get_params_matching_predicate(
        fn, lambda p: _is_annotation_of_type(p.annotation, type_name)
    )


def _create_stub_run_context(
    ctx: AgentContext[Any],
    prompt: str | Sequence[UserContent] | None = None,
) -> RunContext[Any]:
    """Create a stub RunContext from AgentContext for MCP bridge tool invocations.

    This provides a best-effort RunContext for tools that require it when
    invoked via MCP bridge. Not all fields can be populated accurately since
    we're outside of pydantic-ai's normal execution flow.

    Args:
        ctx: The AgentContext available in the bridge
        prompt: The current prompt being processed

    Returns:
        A RunContext with available information populated
    """
    match ctx.agent:
        case Agent():
            model = ctx.agent._model or TestModel()
        # case ACPAgent():
        #     try:
        #         model = infer_model(ctx.agent.model_name or "test")
        #     except Exception:
        #         model = TestModel()
        case _:
            model = TestModel()
    # Create a minimal usage object
    return RunContext(
        deps=ctx.data,
        model=model,
        usage=RunUsage(),
        prompt=prompt,
        messages=[],
        tool_name=ctx.tool_name,
        tool_call_id=ctx.tool_call_id,
    )


def _convert_to_tool_result(result: Any) -> FastMCPToolResult:
    """Convert a tool's return value to a FastMCP ToolResult.

    Handles different result types appropriately:
    - FastMCP ToolResult: Pass through unchanged
    - AgentPool ToolResult: Convert to FastMCP format
    - Pydantic AI ToolReturn: Preserve the model-visible return value and metadata
    - dict: Use as structured_content (enables programmatic access by clients)
    - Pydantic models: Serialize to dict for structured_content
    - Other types: Pass to ToolResult(content=...) which handles conversion internally
    """
    from fastmcp.tools.tool import ToolResult as FastMCPToolResult

    from wolfharness.tools.base import ToolResult as AgentPoolToolResult

    match result:
        case FastMCPToolResult():
            return result
        case AgentPoolToolResult():
            return FastMCPToolResult(
                content=result.content,
                structured_content=result.structured_content,
                meta=result.metadata,
            )
        case ToolReturn():
            metadata = result.metadata if isinstance(result.metadata, dict) else None
            return FastMCPToolResult(
                content=result.return_value if result.return_value is not None else "",
                structured_content=metadata,
                meta=metadata,
            )
        case dict():
            return FastMCPToolResult(structured_content=result)
        case BaseModel():
            return FastMCPToolResult(structured_content=result.model_dump(mode="json"))
        case _:
            return FastMCPToolResult(content=result if result is not None else "")


def _append_injection_to_result(result: Any, injection: str) -> Any:
    """Append an injected message to a tool result.

    The injection is already wrapped in XML tags by the PromptInjectionManager.

    Handles different result types:
    - str: Append with newline separator
    - AgentPool ToolResult: Modify content field
    - dict: Add 'injected_context' key
    - Other: Wrap in dict with original and injection
    """
    from wolfharness.tools.base import ToolResult as AgentPoolToolResult

    match result:
        case str():
            return f"{result}\n\n{injection}"
        case AgentPoolToolResult(content=None):
            return replace(result, content=injection)
        case AgentPoolToolResult(content=str() as existing):
            return replace(result, content=f"{existing}\n\n{injection}")
        case AgentPoolToolResult(content=list() as existing):
            return replace(result, content=[*existing, f"\n\n{injection}"])
        case dict():
            return {**result, "injected_context": injection}
        case _:
            return (result, {"injected_context": injection})


def _extract_tool_call_id(context: Context | None) -> str:
    """Extract Claude's original tool_call_id from request metadata.

    Claude Code passes the tool_use_id via the _meta field as
    'claudecode/toolUseId'. This allows us to maintain consistent
    tool_call_ids across ToolCallStartEvent and ToolCallCompleteEvent.

    Falls back to generating a UUID if not available.
    """
    if context and (request_ctx := context.request_context) and request_ctx.meta:
        # Access extra fields on the Meta object (extra="allow" in pydantic)
        meta_dict = request_ctx.meta.model_dump()
        claude_tool_id = meta_dict.get("claudecode/toolUseId")
        if isinstance(claude_tool_id, str):
            logger.debug("Extracted Claude tool_call_id", tool_call_id=claude_tool_id)
            return claude_tool_id
    return str(uuid4())  # Generate fallback UUID if no tool_call_id found in meta


@dataclass
class ToolManagerBridge:
    """Exposes a node's tools as an MCP server for ACP agents.

    This bridge allows external ACP agents to access our internal toolsets
    (SubagentTools, AgentManagementTools, etc.) via HTTP MCP transport.

    The node's existing context is used for tool invocations, providing
    pool access and proper configuration without reconstruction.

    Example:
        ```python
        bridge = ToolManagerBridge(node=agent)
        async with bridge.set_run_context(...):
            # Bridge is running, get MCP config for ACP agent
            mcp_config = bridge.get_mcp_server_config()
            # Pass to ACP agent...
        ```
    """

    node: BaseAgent[Any, Any]
    """The node whose tools to expose."""

    server_name: str | None = None
    """Name for the MCP server."""

    _current_context: AgentContext[Any] | None = field(default=None, init=False, repr=False)
    """Current run-scoped context (set by set_run_context, read by WrappedTool.run)."""

    _current_prompt: str | Sequence[UserContent] | None = field(
        default=None, init=False, repr=False
    )
    """Current prompt for tool invocations (needed for stub RunContext)."""

    _mcp: FastMCP | None = field(default=None, init=False, repr=False)
    """FastMCP server instance."""

    _server: Server | None = field(default=None, init=False, repr=False)
    """Uvicorn server instance."""

    _server_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    """Background task running the server."""

    _actual_port: int | None = field(default=None, init=False, repr=False)
    """Actual port the server is bound to."""

    tool_metadata: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    """Metadata from tool results, keyed by tool_call_id.

    Populated by WrappedTool.run() when a tool returns metadata.
    Agents read this to enrich ToolCallCompleteEvent with metadata that
    the SDK would otherwise strip (e.g. MCP _meta fields).
    Cleared at the start of each run via set_run_context().
    """

    async def __aenter__(self) -> Self:
        """Start the MCP server."""
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Stop the MCP server."""
        await self.stop()

    async def start(self) -> None:
        """Start the HTTP MCP server in the background."""
        from fastmcp import FastMCP

        self._mcp = FastMCP(name=self.resolved_server_name)
        await self._register_tools()
        self._subscribe_to_tool_changes()
        await self._start_server()

    async def stop(self) -> None:
        """Stop the HTTP MCP server."""
        # Unsubscribe from tool changes
        self._unsubscribe_from_tool_changes()

        if self._server:
            self._server.should_exit = True
            if self._server_task:
                try:
                    await asyncio.wait_for(self._server_task, timeout=5.0)
                except TimeoutError:
                    self._server_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._server_task
            self._server = None
            self._server_task = None
        self._mcp = None
        self._actual_port = None
        logger.info("ToolManagerBridge stopped")

    def _subscribe_to_tool_changes(self) -> None:
        """Subscribe to tool changes from all observable capabilities."""
        from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

        node = self.node
        host_ctx = node.host_context
        registry = host_ctx.extension_registry if host_ctx is not None else None
        if registry is not None:
            session_id = node.session_id or ""
            if session_id:
                scope = Scope(
                    level=ScopeLevel.SESSION,
                    agent_name=node.name,
                    session_id=session_id,
                )
            else:
                scope = Scope(level=ScopeLevel.AGENT, agent_name=node.name)
            _providers = registry.get_observable_capabilities(scope)
        else:
            _providers = node._all_capabilities

        for _provider in _providers:
            on_change = getattr(_provider, "on_change", None)
            if callable(on_change):
                # on_change returns an async iterator or None
                # We subscribe to the tools_changed signal if available
                pass  # Signal-based change notification handled per-capability

    def _unsubscribe_from_tool_changes(self) -> None:
        """Disconnect from tool change signals on all observable capabilities."""
        from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

        node = self.node
        host_ctx = node.host_context
        registry = host_ctx.extension_registry if host_ctx is not None else None
        if registry is not None:
            session_id = node.session_id or ""
            if session_id:
                scope = Scope(
                    level=ScopeLevel.SESSION,
                    agent_name=node.name,
                    session_id=session_id,
                )
            else:
                scope = Scope(level=ScopeLevel.AGENT, agent_name=node.name)
            _providers = registry.get_observable_capabilities(scope)
        else:
            _providers = node._all_capabilities

        for _provider in _providers:
            pass  # No-op: signal subscriptions are managed per-capability

    async def _on_tools_changed(self, event: ChangeEvent) -> None:
        """Handle tool changes from a provider."""
        logger.info(
            "Tools changed in provider, refreshing MCP tools",
            capability_name=event.capability_name,
            kind=event.kind,
        )
        if self._mcp:
            await self._refresh_tools()

    async def _refresh_tools(self) -> None:
        """Refresh tools registered with the MCP server.

        Uses FastMCP's add_tool/remove_tool API which automatically sends
        ToolListChanged notifications when called within a request context.

        Note: FastMCP only sends notifications when inside a request context
        (ContextVar-based). Outside of requests, tools are updated but clients
        won't receive a push notification - they'll see changes on next list_tools.

        Future improvement: Access StreamableHTTPSessionManager._server_instances
        to broadcast ToolListChanged to all connected sessions regardless of context.
        """
        if not self._mcp:
            return

        # Get current and new tool sets
        # Support both old and new FastMCP API

        # New API (git): tools stored in _local_provider._components
        # Keys are prefixed with 'tool:', e.g., 'tool:bash'
        current_names = {
            key.removeprefix("tool:")
            for key in self._mcp.local_provider._components
            if key.startswith("tool:")
        }
        new_tools = await self.node._get_all_tools()
        new_names = {t.name for t in new_tools}
        # Remove tools that are no longer present
        for name in current_names - new_names:
            with suppress(Exception):
                self._mcp.remove_tool(name)

        # Add/update tools
        for tool in new_tools:
            if tool.name in current_names:
                # Remove and re-add to update
                with suppress(Exception):
                    self._mcp.remove_tool(tool.name)
            self._register_single_tool(tool)

    @property
    def port(self) -> int:
        """Get the actual port the server is running on."""
        if self._actual_port is None:
            raise RuntimeError("Server not started")
        return self._actual_port

    @property
    def url(self) -> str:
        """Get the server URL."""
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def resolved_server_name(self) -> str:
        """Get the server name."""
        return self.server_name or f"wolfharness-{self.node.name}-tools"

    async def _register_tools(self) -> None:
        """Register all node tools with the FastMCP server."""
        if not self._mcp:
            return

        tools = await self.node._get_all_tools()
        enabled_tools = [t for t in tools if t.enabled]
        for tool in enabled_tools:
            self._register_single_tool(tool)
        logger.info("Registered tools with MCP bridge", tools=[t.name for t in enabled_tools])

    def _register_single_tool(self, tool: Tool) -> None:
        """Register a single tool with the FastMCP server."""
        if not self._mcp:
            return
        from fastmcp.tools import Tool as FastMCPTool

        class _BridgeTool(FastMCPTool):
            """Custom FastMCP Tool that wraps a wolfharness Tool.

            This allows us to use our own schema and invoke tools with AgentContext.
            """

            def __init__(self, tool: Tool, bridge: ToolManagerBridge) -> None:
                # Get input schema from our tool
                schema = tool.schema["function"]
                input_schema = schema.get("parameters", {"type": "object", "properties": {}})
                # Filter out context parameters - they're auto-injected by the bridge
                fn = tool.get_callable()
                context_params = _get_context_param_names(fn, "AgentContext")
                run_context_params = _get_context_param_names(fn, "RunContext")
                all_context_params = context_params | run_context_params
                filtered_schema = filter_schema_params(dict(input_schema), all_context_params)
                desc = tool.description or "No description"
                super().__init__(
                    name=tool.name,
                    description=desc,
                    parameters=filtered_schema,
                    annotations=tool.get_mcp_tool_annotations(),
                    # output_schema=...,
                )
                # Set these AFTER super().__init__() to avoid being overwritten
                self._tool = tool
                self._bridge = bridge

            async def run(self, arguments: dict[str, Any]) -> FastMCPToolResult:
                """Execute the wrapped tool with context bridging."""
                from fastmcp.server.dependencies import get_context

                from wolfharness.tools.base import ToolResult as AgentPoolToolResult

                args = arguments.copy()
                # Validate args against tool's schema
                # try:
                #     param_model = self._tool.schema_obj.create_parameter_model()
                #     param_model.model_validate(args)
                # except pydantic.ValidationError as e:
                #     error_msg = (
                #         f"Tool '{self._tool.name}' called with invalid args. "
                #         f"Ensure args match the schema.\n\nValidation errors:\n{e}"
                #     )
                #     return ToolResult(content=[TextContent(type="text", text=error_msg)])

                # Get FastMCP context from context variable (not passed as parameter)
                try:
                    mcp_context: Context | None = get_context()
                except LookupError:
                    mcp_context = None
                # Try to get Claude's original tool_call_id from request metadata
                tc_id = _extract_tool_call_id(mcp_context)
                # Derive per-call context from the run-scoped base context
                base = self._bridge._current_context or self._bridge.node.get_context()
                args_ = args.copy()
                ctx = replace(base, tool_name=self._tool.name, tool_call_id=tc_id, tool_input=args_)
                # Invoke with context - copy args since invoke_tool_with_context
                # modifies kwargs in-place to inject context parameters
                result = await self._bridge.invoke_tool_with_context(self._tool, ctx, args)
                # Store metadata for agent to correlate with ToolCallCompleteEvent
                # (works around Claude SDK stripping MCP _meta field)
                if isinstance(result, AgentPoolToolResult) and result.metadata:
                    logger.info("Storing tool result metadata", tool_call_id=tc_id)
                    self._bridge.tool_metadata[tc_id] = result.metadata
                if isinstance(result, ToolReturn) and isinstance(result.metadata, dict):
                    logger.info("Storing tool return metadata", tool_call_id=tc_id)
                    self._bridge.tool_metadata[tc_id] = result.metadata

                # Consume pending injection from node's run context (isolated per-call)
                # Use get_active_run_context() for ContextVar + SessionPool fallback.
                run_ctx = self._bridge.node.get_active_run_context()
                injection_manager = run_ctx.injection_manager if run_ctx else None
                if injection_manager and (injection := await injection_manager.consume()):
                    result = _append_injection_to_result(result, injection)

                return _convert_to_tool_result(result)

        # Create a custom FastMCP Tool that wraps our tool
        bridge_tool = _BridgeTool(tool=tool, bridge=self)
        self._mcp.add_tool(bridge_tool)

    @asynccontextmanager
    async def set_run_context(
        self,
        context: AgentContext[Any],
        prompt: str | Sequence[UserContent] | None = None,
    ) -> AsyncIterator[Self]:
        """Context manager for setting run-scoped state.

        Stores the AgentContext for the duration of the run so that
        WrappedTool.run() can derive per-call contexts via replace().

        Args:
            context: Base AgentContext for this run (deps, input_provider, pool, etc.)
            prompt: Current prompt being processed (needed for stub RunContext)
        """
        self._current_context = context
        self._current_prompt = prompt
        self.tool_metadata.clear()
        try:
            yield self
        finally:
            self._current_context = None
            self._current_prompt = None

    async def invoke_tool_with_context(
        self,
        tool: Tool,
        ctx: AgentContext[Any],
        kwargs: dict[str, Any],
    ) -> Any:
        """Invoke a tool with proper context injection and hooks.

        Handles tools that expect AgentContext, RunContext, or neither.
        Runs pre/post tool hooks if configured on the node.
        """
        from wolfharness.tasks import ToolSkippedError

        if hooks := self.node.hooks:
            pre_result = await hooks.run_pre_tool_hooks(
                agent_name=ctx.node_name,
                tool_name=tool.name,
                tool_input=kwargs,
                session_id=None,
                env=self.node.env,
            )
            if pre_result.get("decision") == "deny":
                reason = pre_result.get("reason", "Blocked by pre-tool hook")
                raise ToolSkippedError(f"Tool {tool.name} blocked: {reason}")
            # Apply modified input if provided
            if modified := pre_result.get("modified_input"):
                kwargs.update(modified)

        fn = tool.get_callable()
        # Inject AgentContext parameters
        context_param_names = _get_context_param_names(fn, "AgentContext")
        for param_name in context_param_names:
            if param_name not in kwargs:
                kwargs[param_name] = ctx
        # Inject RunContext parameters (as stub since we're outside pydantic-ai)
        run_context_param_names = _get_context_param_names(fn, "RunContext")
        if run_context_param_names:
            stub_run_ctx = _create_stub_run_context(ctx, prompt=self._current_prompt)
            for param_name in run_context_param_names:
                if param_name not in kwargs:
                    kwargs[param_name] = stub_run_ctx

        start_time = time.perf_counter()
        result = fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        duration_ms = (time.perf_counter() - start_time) * 1000
        if hooks:
            await hooks.run_post_tool_hooks(
                agent_name=ctx.node_name,
                tool_name=tool.name,
                tool_input=kwargs,
                tool_output=result,
                duration_ms=duration_ms,
                session_id=None,
                env=self.node.env,
            )

        return result

    async def _start_server(self) -> None:
        """Start the uvicorn server in the background."""
        import socket

        import uvicorn

        if not self._mcp:
            raise RuntimeError("MCP server not initialized")

        # Auto-select an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._actual_port = port
        app = self._mcp.http_app(transport="http")  # Create the ASGI app
        cfg = uvicorn.Config(
            app=app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio"
        )
        self._server = uvicorn.Server(cfg)
        # Start server in background task
        name = f"mcp-bridge-{self.resolved_server_name}"
        self._server_task = asyncio.create_task(self._server.serve(), name=name)
        await anyio.sleep(0.1)  # Wait briefly for server to start
        logger.info("ToolManagerBridge started", url=self.url)
