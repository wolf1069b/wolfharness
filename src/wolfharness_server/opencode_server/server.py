"""OpenCode-compatible FastAPI server.

This server implements the OpenCode API endpoints to allow OpenCode SDK clients
to interact with AgentPool agents.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from wolfharness import log
from wolfharness_server.opencode_server.routes import (
    agent_router,
    app_router,
    config_router,
    file_router,
    global_router,
    lsp_router,
    message_router,
    permission_router,
    pty_router,
    question_router,
    session_router,
    tui_router,
)
from wolfharness_server.opencode_server.state import ServerState
from wolfharness_server.opencode_server.todo_utils import build_opencode_todos


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Set as AbstractSet

    from httpx import Headers

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.storage.manager import SessionMetadataGeneratedEvent
    from wolfharness.utils.todos import TodoTracker


VERSION = "0.1.0"
logger = log.get_logger(__name__)


def filter_headers(headers: Headers) -> dict[str, str]:
    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in excluded_headers}


PROXY_BLOCKED_PREFIXES = (
    "api/",
    "session/",
    "config/",
    "agent/",
    "model/",
    "provider/",
    "command/",
    "skill/",
    "location/",
    "integration/",
    "file/",
    "todo/",
    "diff/",
    "snapshot/",
    "v1/",
    "experimental/",
)
"""Unmatched API path prefixes that must NOT be forwarded to the hosted Web UI.

Requests falling under these prefixes return 404 so the OpenCode SDK/TUI learns
the endpoint doesn't exist instead of receiving a `text/html` SPA page, which
the client would mis-parse and crash on (see `fs.find`).
"""


def is_proxy_path_blocked(path: str) -> bool:
    """Return whether an unmatched path should 404 instead of proxying.

    Any route under ``api/`` (or a known API prefix) is treated as a missing
    endpoint so the OpenCode client gets a structured error rather than the
    hosted Web UI's ``index.html`` being forwarded back and crashing the TUI.
    """
    return any(path.startswith(prefix) for prefix in PROXY_BLOCKED_PREFIXES)


class OpenCodeJSONResponse(JSONResponse):
    """Custom JSON response that excludes None values (like OpenCode does)."""

    def render(self, content: Any) -> bytes:
        from fastapi.encoders import jsonable_encoder

        return super().render(jsonable_encoder(content, exclude_none=True))


async def check_pypi_version(package: str = "wolfharness") -> str | None:
    """Check PyPI for the latest version of a package.

    Args:
        package: Package name to check

    Returns:
        Latest version string, or None if check fails
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"https://pypi.org/pypi/{package}/json")
            if response.status_code == 200:  # noqa: PLR2004
                data: dict[str, Any] = response.json()
                info: dict[str, Any] = data.get("info", {})
                version: str | None = info.get("version")
                return version
    except Exception:  # noqa: BLE001
        pass
    return None


def compare_versions(current: str, latest: str) -> bool:
    """Check if latest version is newer than current."""
    from packaging.version import Version

    try:
        return Version(latest) > Version(current)
    except Exception:  # noqa: BLE001
        return False


def create_app(*, agent: BaseAgent[Any, Any], working_dir: str | None = None) -> FastAPI:  # noqa: PLR0915
    """Create the FastAPI application.

    Args:
        agent: The agent to use for handling messages. Must have agent_pool set.
        working_dir: Working directory for file operations. Defaults to cwd.

    Returns:
        Configured FastAPI application.

    Raises:
        ValueError: If agent has no agent_pool set.
    """
    import logfire

    ctx = agent.host_context
    if ctx is None:
        msg = "Agent must have host_context set"
        raise ValueError(msg)

    session_controller = None
    if ctx.session_pool is not None:
        session_controller = ctx.session_pool.sessions

    state = ServerState(
        working_dir=working_dir or str(Path.cwd()),
        agent=agent,
        session_controller=session_controller,
    )

    # Set up SessionPool integration for session-scoped event consumption
    if state.pool.session_pool is not None:
        from wolfharness_server.opencode_server.session_pool_integration import (
            OpenCodeSessionPoolIntegration,
        )

        state.session_pool_integration = OpenCodeSessionPoolIntegration(
            session_pool=state.pool.session_pool,
            server_state=state,
        )

    # Set up skill command bridge and CommandStore for slash commands.
    # Builds SkillCommand objects from pool.skills and bridges them to
    # slashed Commands for OpenCode's slash command execution endpoint.
    if state.pool.skills is not None:
        from slashed import CommandStore

        from wolfharness.skills.command import SkillCommand
        from wolfharness_server.opencode_server.skill_bridge import OpenCodeSkillBridge

        def _build_skill_commands() -> list[SkillCommand]:
            """Build SkillCommand list from current pool skills."""
            cmds: list[SkillCommand] = []
            for skill in state.pool.skills.list_skills():
                if not skill.user_invocable:
                    continue
                cmds.append(
                    SkillCommand(
                        name=skill.name,
                        description=skill.description,
                        skill=skill,
                        skill_uri=f"skill://{skill.name}",
                    )
                )
            return cmds

        def _sync_command_store(
            bridge: OpenCodeSkillBridge,
            store: CommandStore,
            cmds: list[SkillCommand],
        ) -> None:
            """Reconcile bridge and store with the latest skill commands."""
            new_names = {cmd.name for cmd in cmds}
            old_names = {sc.name for sc in bridge.get_skill_commands()}

            # Remove stale commands from bridge and store.
            for stale in old_names - new_names:
                bridge.handle_change(stale, None)
                store.unregister_command(stale)

            # Add/update commands in bridge and store.
            for cmd in cmds:
                bridge.handle_change(cmd.name, cmd)

            # Register all current commands in the store.
            for slashed_cmd in bridge.get_commands():
                store.register_command(slashed_cmd, replace=True)

        # Use SkillManagerCap for skill instruction loading (D8 consolidation).
        skill_cap = state.pool.skill_capabilities[0] if state.pool.skill_capabilities else None
        bridge = OpenCodeSkillBridge(skill_provider=skill_cap)
        initial_cmds = _build_skill_commands()
        for cmd in initial_cmds:
            bridge.handle_change(cmd.name, cmd)

        state.skill_bridge = bridge
        state.command_store = CommandStore(commands=bridge.get_commands())
        logger.debug(
            "OpenCode skill bridge setup complete",
            command_count=len(initial_cmds),
        )

        # Subscribe to ChangeEvent stream for dynamic skill updates.
        # ExtensionRegistry.merge_change_streams() yields ChangeEvent with
        # kind="skills_changed" when capabilities emit change notifications.
        extension_registry = state.pool.extension_registry
        if extension_registry is not None:
            from wolfharness.capabilities.extension_registry import (
                Scope,
                ScopeLevel,
            )

            async def _watch_skill_changes() -> None:
                """Watch for skill change events and update CommandStore."""
                stream = extension_registry.merge_change_streams(Scope(level=ScopeLevel.POOL))
                if stream is None:
                    return
                async for event in stream:
                    if event.kind != "skills_changed":
                        continue
                    logger.info("Skill change detected, rebuilding CommandStore")
                    try:
                        new_cmds = _build_skill_commands()
                        if state.command_store is not None and state.skill_bridge is not None:
                            _sync_command_store(
                                state.skill_bridge,
                                state.command_store,
                                new_cmds,
                            )
                            logger.debug(
                                "CommandStore rebuilt",
                                command_count=len(new_cmds),
                            )
                    except Exception:
                        logger.exception("Failed to rebuild CommandStore after skill change")

            state._skill_change_task = asyncio.create_task(_watch_skill_changes())

            # Watch for MCP tool list changes and broadcast McpToolsChangedEvent.
            # McpServerCap.on_change() yields ChangeEvent(kind="tools_changed")
            # when an MCP server's tool list changes (via notifications/tools/list_changed).
            # This watcher converts that to McpToolsChangedEvent and broadcasts it
            # so connected OpenCode clients can refresh their tool lists.
            from wolfharness_server.opencode_server.event_processor import EventProcessor

            processor = EventProcessor()

            async def _watch_mcp_tool_changes() -> None:
                """Watch for MCP tool change events and broadcast notifications."""
                stream = extension_registry.merge_change_streams(Scope(level=ScopeLevel.POOL))
                if stream is None:
                    return
                async for event in stream:
                    if event.kind != "tools_changed":
                        continue
                    logger.info(
                        "MCP tools changed, broadcasting notification",
                        server=event.capability_name,
                    )
                    try:
                        oc_event = processor.create_mcp_tools_changed_event(
                            server=event.capability_name,
                        )
                        await state.broadcast_event(oc_event)
                    except Exception:
                        logger.exception("Failed to broadcast McpToolsChangedEvent")

            state._mcp_tool_change_task = asyncio.create_task(_watch_mcp_tool_changes())

    # Set up todo change callback to broadcast events
    async def on_todo_change(tracker: TodoTracker) -> None:
        """Broadcast todo updates to all active sessions."""
        from wolfharness_server.opencode_server.models.events import Todo, TodoUpdatedEvent

        # Convert tracker entries to OpenCode Todo models.
        todos = build_opencode_todos(tracker, Todo)
        # Broadcast to all active sessions
        for session_id in state.sessions:
            event = TodoUpdatedEvent.create(session_id=session_id, todos=todos)
            await state.broadcast_event(event)

    state.pool.todos.on_change = on_todo_change
    # Set up title generation callback to update OpenCode sessions

    async def on_title_generated(event: SessionMetadataGeneratedEvent) -> None:
        """Update session when metadata is generated by StorageManager."""
        from wolfharness_server.opencode_server.converters import opencode_to_session_data
        from wolfharness_server.opencode_server.models.events import SessionUpdatedEvent

        logger.info("on_title_generated called", session_id=event.session_id, data=event.metadata)
        session_id = event.session_id
        if session_id in state.sessions:
            # Update in-memory session
            session = state.sessions[session_id]
            updated_session = session.model_copy(update={"title": event.metadata.title})
            state.sessions[session_id] = updated_session
            # Persist to storage
            session_data = opencode_to_session_data(
                updated_session,
                agent_name=state.agent.name,
                pool_id=state.pool.manifest.config_file_path,
            )
            if state.pool.session_pool and state.pool.session_pool.sessions.store:
                await state.pool.session_pool.sessions.store.save_session(session_data)
            # Broadcast session update to UI
            await state.broadcast_event(SessionUpdatedEvent.create(updated_session))
        else:
            logger.warning("Session not found in state.sessions", session_id=session_id)

    # Connect to storage manager's metadata_generated signal
    if state.storage:
        state.storage.metadata_generated.connect(on_title_generated)

    # Watchers for VCS and file events
    branch_watcher: Any = None
    project_file_watcher: Any = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: PLR0915
        nonlocal branch_watcher, project_file_watcher

        from watchfiles import Change

        from wolfharness.utils.file_watcher import FileWatcher, GitBranchWatcher
        from wolfharness_server.opencode_server.models import (
            FileWatcherUpdatedEvent,
            VcsBranchUpdatedEvent,
        )

        # --- Git branch watcher ---
        async def on_branch_change(branch: str | None) -> None:
            """Broadcast branch change to all subscribers."""
            logger.info("Broadcasting vcs.branch.updated event", branch=branch)
            event = VcsBranchUpdatedEvent.create(branch=branch)
            await state.broadcast_event(event)

        logger.info("Setting up GitBranchWatcher", working_dir=state.working_dir)
        branch_watcher = GitBranchWatcher(repo_path=state.working_dir, callback=on_branch_change)
        await branch_watcher.start()
        logger.info("GitBranchWatcher started", current_branch=branch_watcher.current_branch)
        # --- Project file watcher ---
        # Map watchfiles Change types to OpenCode event types
        change_type_map: dict[Change, str] = {
            Change.added: "add",
            Change.modified: "change",
            Change.deleted: "unlink",
        }

        # Get ignore patterns from config
        ignore_patterns: list[str] = []
        if state.config and state.config.watcher and state.config.watcher.ignore:
            ignore_patterns = state.config.watcher.ignore

        def should_ignore(file_path: str) -> bool:
            """Check if a file path should be ignored."""
            import fnmatch

            # Always ignore .git
            if "/.git/" in file_path or file_path.endswith("/.git"):
                return True
            # Check user-configured patterns
            rel_path = file_path
            if state.working_dir and file_path.startswith(state.working_dir):
                rel_path = file_path[len(state.working_dir) :].lstrip("/")
            return any(fnmatch.fnmatch(rel_path, pat) for pat in ignore_patterns)

        async def on_file_change(changes: AbstractSet[tuple[Change, str]]) -> None:
            """Broadcast file changes to all subscribers."""
            for change_type, file_path in changes:
                if should_ignore(file_path):
                    continue
                event_type = change_type_map.get(change_type, "change")
                logger.info(
                    "Broadcasting file.watcher.updated", event_type=event_type, path=file_path
                )
                event = FileWatcherUpdatedEvent.create(file=file_path, event=event_type)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                await state.broadcast_event(event)

        logger.info("Setting up project FileWatcher", working_dir=state.working_dir)
        project_file_watcher = FileWatcher(
            paths=[state.working_dir],
            callback=on_file_change,
            debounce=500,  # 500ms debounce to batch rapid changes
        )
        await project_file_watcher.start()
        logger.info("Project FileWatcher started")

        # --- Version update check (triggered when first client connects) ---
        async def check_for_updates() -> None:
            """Check PyPI for updates and notify via toast."""
            from wolfharness import __version__ as current_version
            from wolfharness_server.opencode_server.models.events import TuiToastShowEvent

            latest = await check_pypi_version("wolfharness")
            if latest and compare_versions(current_version, latest):
                logger.info("Update available", current_version=current_version, latest=latest)
                event = TuiToastShowEvent.create(
                    title="Update Available",
                    message=f"wolfharness {latest} is available (current: {current_version})",
                    variant="info",
                    duration=10000,
                )
                await state.broadcast_event(event)

        # Register callback to run when first SSE client connects
        state.on_first_subscriber = check_for_updates
        # Pool context is managed externally (by the caller)
        yield
        # Shutdown - clean up session pool integration first
        if state.session_pool_integration is not None:
            await state.session_pool_integration.shutdown()
        # Cancel skill change watcher
        if state._skill_change_task is not None:
            state._skill_change_task.cancel()
            try:
                await state._skill_change_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error during skill change task cleanup")
            state._skill_change_task = None
        # Cancel MCP tool change watcher
        if state._mcp_tool_change_task is not None:
            state._mcp_tool_change_task.cancel()
            try:
                await state._mcp_tool_change_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Error during MCP tool change task cleanup")
            state._mcp_tool_change_task = None
        # Then clean up background tasks
        await state.cleanup_tasks()
        # Then tear down watchers and shared infrastructure
        state.pool.todos.on_change = None
        if branch_watcher:
            await branch_watcher.stop()
        if project_file_watcher:
            await project_file_watcher.stop()
        # Clean up LSP servers
        await state.lsp_manager.stop_all()

    app = FastAPI(
        title="OpenCode-Compatible API",
        description="AgentPool server with OpenCode API compatibility",
        version=VERSION,
        lifespan=lifespan,
        default_response_class=OpenCodeJSONResponse,
    )

    # Add CORS middleware (required for OpenCode TUI)
    app.add_middleware(
        CORSMiddleware,  # ty: ignore[invalid-argument-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state on app for access in routes
    app.state.server_state = state

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = await request.body()
        print(f"Validation error for {request.url}")
        print(f"Body: {body.decode()}")
        print(f"Errors: {exc.errors()}")
        content = {"detail": exc.errors(), "body": body.decode()}
        return JSONResponse(status_code=422, content=content)

    # Register routers
    app.include_router(global_router)
    app.include_router(app_router)
    app.include_router(config_router)
    app.include_router(session_router)
    app.include_router(message_router)
    app.include_router(file_router)
    app.include_router(agent_router)
    app.include_router(permission_router)
    app.include_router(question_router)
    app.include_router(pty_router)
    app.include_router(tui_router)
    app.include_router(lsp_router)

    # OpenAPI doc redirect
    @app.get("/doc")
    async def get_doc() -> RedirectResponse:
        """Redirect to OpenAPI docs."""
        return RedirectResponse(url="/docs")

    # OTLP telemetry sink endpoints (compatibility for OpenCode 1.4.4+)
    # Must be registered BEFORE the catch-all proxy so POST /v1/metrics etc.
    # don't fall through to a GET/HEAD/OPTIONS-only route (→ 405).
    @app.post("/v1/metrics")
    async def otlp_metrics(request: Request) -> Response:
        """Accept OTLP metrics payloads and discard them."""
        return Response(status_code=204)

    @app.post("/v1/traces")
    async def otlp_traces(request: Request) -> Response:
        """Accept OTLP traces payloads and discard them."""
        return Response(status_code=204)

    @app.post("/v1/logs")
    async def otlp_logs(request: Request) -> Response:
        """Accept OTLP logs payloads and discard them."""
        return Response(status_code=204)

    # Proxy catch-all for OpenCode's hosted web UI
    # This must be registered LAST so it doesn't catch API routes
    @app.api_route("/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
    async def proxy_web_ui(request: Request, path: str) -> Response:
        """Proxy unmatched GET requests to OpenCode's hosted web UI.

        This allows users to open http://localhost:4096 in a browser and get
        the full OpenCode web interface, which then makes API calls back to
        this local server for all data operations.

        API paths (``session/``, ``config/``, ``agent/``, etc.) are excluded
        from the proxy so that unmatched API routes return 404 instead of
        silently forwarding to the cloud (which would return empty data and
        break the TUI's history loading).
        """
        # Don't proxy API paths — return 404 so the TUI knows the endpoint
        # doesn't exist rather than receiving cloud data for local sessions.
        if is_proxy_path_blocked(path):
            raise HTTPException(status_code=404, detail=f"Endpoint not found: /{path}")

        import httpx

        # Build target URL
        url = f"https://app.opencode.ai/{path}"
        if request.url.query:
            url += f"?{request.url.query}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Forward the request
            response = await client.request(
                method=request.method,
                url=url,
                headers={"host": "app.opencode.ai"},
                follow_redirects=True,
            )
            # Filter out hop-by-hop headers that shouldn't be forwarded
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=filter_headers(response.headers),
                media_type=response.headers.get("content-type"),
            )

    if os.environ.get("LOGFIRE_DISABLE", "").lower() != "true":
        try:
            from wolfharness_server.opencode_server.otel_fastapi_patch import (
                patch_otel_fastapi_route_details,
            )

            # FastAPI >= 0.136 wraps included routers in _IncludedRouter nodes
            # without a `.path` attribute; OTel's _get_route_details crashes on
            # these for partial matches (e.g. /session/{id}/message), turning
            # valid requests into 500s. Guard before instrumenting.
            patch_otel_fastapi_route_details()
            logfire.instrument_fastapi(app)
        except Exception:
            logger.warning("Failed to instrument FastAPI app with Logfire", exc_info=True)
    return app


@dataclass
class OpenCodeServer:
    """OpenCode-compatible server wrapper.

    Provides a convenient interface for running the server.
    """

    agent: BaseAgent[Any, Any]
    """The agent to use for handling messages. Must have agent_pool set."""
    host: str = "127.0.0.1"
    """Host to bind to."""
    port: int = 4096
    """Port to listen on."""
    working_dir: str | None = None
    """Working directory for file operations."""
    _app: FastAPI | None = field(default=None, init=False, repr=False)

    @property
    def app(self) -> FastAPI:
        """Get or create the FastAPI application."""
        if self._app is None:
            self._app = create_app(agent=self.agent, working_dir=self.working_dir)
        return self._app

    def run(self) -> None:
        """Run the server (blocking)."""
        import uvicorn

        uvicorn.run(self.app, host=self.host, port=self.port)

    async def run_async(self) -> None:
        """Run the server asynchronously."""
        import uvicorn

        config = uvicorn.Config(self.app, host=self.host, port=self.port, ws="websockets-sansio")
        server = uvicorn.Server(config)
        await server.serve()


def run_server(
    agent: BaseAgent[Any, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 4096,
    working_dir: str | None = None,
) -> None:
    """Run the OpenCode-compatible server.

    Args:
        agent: The agent to use for handling messages. Must have agent_pool set.
        host: Host to bind to.
        port: Port to listen on.
        working_dir: Working directory for file operations.
    """
    server = OpenCodeServer(agent, host=host, port=port, working_dir=working_dir)
    server.run()


if __name__ == "__main__":
    import asyncio

    from wolfharness import AgentPool, config_resources

    async def main() -> None:
        pool = AgentPool(config_resources.ACP_ASSISTANT)
        async with pool:
            assert pool.session_pool is not None
            agent = await pool.session_pool.sessions.get_or_create_session_agent(
                "opencode-main", pool.main_agent_name
            )
            run_server(agent)

    asyncio.run(main())
