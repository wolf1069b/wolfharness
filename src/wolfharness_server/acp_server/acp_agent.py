"""ACP (Agent Client Protocol) Agent implementation."""

from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast as _cast

import anyio
import logfire
from opentelemetry.context import attach, detach
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from acp import Agent as ACPAgent
from acp.exceptions import RequestError
from acp.schema import (
    CloseSessionResponse,
    DisableProvidersRequest,
    DisableProvidersResponse,
    ForkSessionResponse,
    InitializeResponse,
    ListProvidersRequest,
    ListProvidersResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    ModelInfo as ACPModelInfo,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionConfigOption,
    SessionConfigSelectOption,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SetProvidersRequest,
    SetProvidersResponse,
    SetSessionConfigOptionResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
)
from wolfharness.log import get_logger
from wolfharness.utils.task_group import ManagedTaskGroup
from wolfharness_server.acp_server.acp_mcp_manager import AcpMcpConnectionManager
from wolfharness_server.acp_server.converters import to_session_config_option, to_session_info
from wolfharness_server.acp_server.provider_router import ProviderRouter
from wolfharness_server.acp_server.session_manager import ACPSessionManager


if TYPE_CHECKING:
    from pydantic_ai import ModelMessage

    from acp import Client
    from acp.schema import (
        AuthenticateRequest,
        CancelNotification,
        ClientCapabilities,
        CloseSessionRequest,
        ForkSessionRequest,
        Implementation,
        InitializeRequest,
        ListSessionsRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
        ResumeSessionRequest,
        SetSessionConfigOptionRequest,
        SetSessionModelRequest,
        SetSessionModeRequest,
    )
    from acp.schema.mcp import AcpMcpServer
    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.host.context import HostContext
    from wolfharness.models.agents import NativeAgentConfig
    from wolfharness.storage.manager import SessionMetadataGeneratedEvent
    from wolfharness_server.acp_server.handler import ACPProtocolHandler
    from wolfharness_server.acp_server.server import ACPServer

logger = get_logger(__name__)


async def get_session_model_state(
    agent: BaseAgent,
    provider_router: ProviderRouter | None = None,
) -> SessionModelState | None:
    """Get SessionModelState from an agent, including configured variants.

    Uses configured-first logic via build_model_state_for_acp(),
    falling back to tokonomics discovery if no configured variants exist.

    Args:
        agent: Any agent with get_available_models() method
        provider_router: Optional ProviderRouter for disable filtering

    Returns:
        SessionModelState with all available models, None if no models available
    """
    from wolfharness_server.shared.model_utils import build_model_state_for_acp

    try:
        state = await build_model_state_for_acp(agent, provider_router)
    except Exception:
        logger.exception("Failed to build model state for ACP")
        return None

    if state is not None:
        return state

    # Final fallback: ensure current model is represented
    current_model = agent.model_name
    if current_model:
        desc = "Currently configured model"
        model_info = ACPModelInfo(model_id=current_model, name=current_model, description=desc)
        return SessionModelState(available_models=[model_info], current_model_id=current_model)

    return None


async def get_session_mode_state(agent: BaseAgent) -> SessionModeState | None:
    """Get SessionModeState from an agent using its get_modes() method.

    Converts wolfharness ModeCategory to ACP SessionModeState format.
    Uses the first category that looks like permissions (not model).

    Args:
        agent: Any agent with get_modes() method

    Returns:
        SessionModeState from agent's modes, None if no modes available
    """
    try:
        mode_categories = await agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        return None
    if not mode_categories:
        return None
    # Find the permissions category (not model)
    category = next((c for c in mode_categories if c.id == "mode"), None)
    if not category:
        return None
    acp_modes = [  # Convert ModeInfo to ACP SessionMode
        SessionMode(id=mode.id, name=mode.name, description=mode.description)
        for mode in category.available_modes
    ]
    return SessionModeState(available_modes=acp_modes, current_mode_id=category.current_mode_id)


async def get_session_config_options(agent: BaseAgent) -> list[SessionConfigOption]:
    """Get SessionConfigOptions from an agent using its get_modes() method."""
    try:
        mode_categories = await agent.get_modes()
    except Exception:
        logger.exception("Failed to get modes from agent")
        return []
    options = [to_session_config_option(category) for category in mode_categories]
    # Append agent_role config option if pool has multiple agents
    if agent_role_opt := get_agent_role_config_option(agent):
        options.append(agent_role_opt)
    return options


def get_agent_role_config_option(agent: BaseAgent[Any, Any]) -> SessionConfigOption | None:
    """Build agent_role config option if pool has more than one agent.

    Args:
        agent: The agent to check pool membership for.

    Returns:
        SessionConfigOption for agent_role, or None if pool has <= 1 agents.
    """
    pool = agent.host_context
    if pool is None or len(pool.manifest.agents) <= 1:
        return None

    choices = [
        SessionConfigSelectOption(
            value=a.name or "",
            name=a.display_name
            if isinstance(a.display_name, str) and a.display_name
            else (a.name or ""),
            description=f"Switch to {a.name or ''} agent",
        )
        for a in pool.manifest.agents.values()
    ]
    return SessionConfigOption(
        id="agent_role",
        name="Agent Role",
        description="Switch between available agents",
        category="other",
        current_value=agent.name,
        options=choices,
    )


@dataclass
class AgentPoolACPAgent(ACPAgent):
    """Implementation of ACP Agent protocol interface for AgentPool.

    This class implements the external library's Agent protocol interface,
    bridging AgentPool with the standard ACP JSON-RPC protocol.
    """

    PROTOCOL_VERSION: ClassVar = 1

    client: Client
    """ACP connection for client communication."""

    default_agent: BaseAgent[Any, Any]
    """Default agent instance to use for new sessions.

    The agent carries its own pool reference via agent.host_context,
    which is used for pool-level operations and agent switching.
    """

    _: KW_ONLY

    debug_commands: bool = False
    """Whether to enable debug slash commands for testing."""

    load_skills: bool | None = None
    """Whether to load client-side skills from .claude/skills directory.

    If None (default), uses the manifest's skills.include_default setting.
    """

    server: ACPServer | None = field(default=None)
    """Reference to the ACPServer for pool hot-switching."""

    subagent_display_mode: Literal["legacy", "zed", "qwen"] = "legacy"
    """Display mode for subagent outputs ("legacy", "zed", or "qwen")."""

    raw_input_mode: Literal["dict", "skip", "json_str"] = "dict"
    """How to emit tool call raw_input ("dict", "skip", or "json_str")."""

    session_manager: ACPSessionManager = field(default=_cast(ACPSessionManager, None))
    """Shared session manager for tracking ACP sessions across connections.

    If None, a new ``ACPSessionManager`` is created per agent instance.
    When provided (e.g., from ``ACPServer``), enables cross-connection
    session tracking and WebSocket disconnect cleanup via
    ``close_all_sessions_for_connection()``.
    """

    _mcp_manager: AcpMcpConnectionManager = field(init=False)
    """Manager for MCP-over-ACP connection lifecycle."""

    _protocol_handler: ACPProtocolHandler | None = field(init=False, default=None)
    """SessionPool-backed protocol handler when ``acp.use_session_pool`` is enabled."""

    def __post_init__(self) -> None:
        """Initialize derived attributes and setup after field assignment."""
        self.client_capabilities: ClientCapabilities | None = None
        self.client_info: Implementation | None = None
        ctx = self.host_context
        pool = self.default_agent._agent_pool
        if pool is None:
            msg = "Default agent has no associated pool"
            raise RuntimeError(msg)
        if self.session_manager is None:
            self.session_manager = ACPSessionManager(pool=pool)
        self._task_group = ManagedTaskGroup()
        self._initialized = False
        self._sessions_cache: ListSessionsResponse | None = None
        self._sessions_cache_time: float = 0.0
        # Connect to title generation signal to notify clients of session updates
        if ctx is not None:
            ctx.storage.metadata_generated.connect(self._on_metadata_generated)

        # Initialize MCP-over-ACP connection manager
        self._mcp_manager = AcpMcpConnectionManager()
        # RFC-0034: Initialize provider router with None manifest (will be updated in initialize)
        self.provider_router = ProviderRouter(None)
        # NEW: Cache agent config for per-session creation (RFC-0031)
        from wolfharness.models.agents import NativeAgentConfig

        ctx = self.host_context
        if ctx is not None and ctx.main_agent_name and ctx.main_agent_name in ctx.manifest.agents:
            cfg = ctx.manifest.agents[ctx.main_agent_name]
            if isinstance(cfg, NativeAgentConfig):
                if cfg.name is None:
                    cfg = cfg.model_copy(update={"name": ctx.main_agent_name})
                self._agent_config = cfg

        # Initialize SessionPool-backed protocol handler if feature flag is enabled
        if self.host_context and (
            self.host_context.manifest.acp and self.host_context.manifest.acp.use_session_pool
        ):
            from wolfharness_server.acp_server.event_converter import ACPEventConverter
            from wolfharness_server.acp_server.handler import ACPProtocolHandler

            self._protocol_handler = ACPProtocolHandler(
                host_context=self.host_context,
                session_manager=self.session_manager,
                event_converter=ACPEventConverter(
                    subagent_display_mode=self.subagent_display_mode,
                    raw_input_mode=self.raw_input_mode,
                ),
                client=self.client,
                client_capabilities=self.client_capabilities,
            )
            logger.info("ACPProtocolHandler initialized for SessionPool mode")

    _agent_config: NativeAgentConfig | None = field(init=False, default=None)
    """Cached main-agent config used during pool swaps."""

    _session_agent_locks: dict[str, asyncio.Lock] = field(init=False, default_factory=dict)
    """Locks for serializing agent swaps per session."""

    _swap_in_progress: bool = field(init=False, default=False)
    """Flag to prevent concurrent session creation during pool swap."""

    # RFC-0034: Provider router for ACP providers/* protocol
    provider_router: ProviderRouter = field(init=False)
    """Router for LLM provider metadata and override tracking."""

    def _get_connection_id(self) -> str | None:
        """Get the WebSocket connection ID for session tracking.

        Returns:
            The connection_id if the client is an AgentSideConnection, else None.
        """
        from acp.agent.connection import AgentSideConnection

        if isinstance(self.client, AgentSideConnection):
            return self.client.connection_id
        return None

    async def _on_metadata_generated(self, event: SessionMetadataGeneratedEvent) -> None:
        """Handle metadata generation - notify active sessions of the update."""
        from acp.schema import SessionInfoUpdate, SessionNotification

        session = self.session_manager.get_session(event.session_id)
        if session is None:
            logger.debug("Metadata generated for inactive session", session_id=event.session_id)
            return

        # Send session info update to client
        update = SessionInfoUpdate(session_id=event.session_id, title=event.metadata.title)
        notification = SessionNotification(session_id=event.session_id, update=update)
        try:
            await session.client.session_update(notification)  # pyright: ignore[reportArgumentType]
            logger.info(
                "Sent session info update",
                session_id=event.session_id,
                title=event.metadata.title,
            )
        except Exception:
            logger.exception("Failed to send session info update", session_id=event.session_id)

    @property
    def host_context(self) -> HostContext | None:
        """Get the host context from the default agent."""
        return self.default_agent.host_context

        # Note: Tool registration happens after initialize() when we know client caps

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        """Initialize the agent and negotiate capabilities."""
        await self._task_group.__aenter__()
        version = min(params.protocol_version, self.PROTOCOL_VERSION)
        self.client_capabilities = params.client_capabilities
        self.client_info = params.client_info
        logger.info("Client info", request=params.model_dump_json())
        self._initialized = True
        # Forward client capabilities to the SessionPool protocol handler so
        # elicitation/create is used when the client supports it.
        if self._protocol_handler is not None:
            self._protocol_handler.client_capabilities = self.client_capabilities
        # Initialize provider router from current pool manifest
        ctx = self.host_context
        manifest = ctx.manifest if ctx else None
        self.provider_router = ProviderRouter(manifest)
        # Gate turn_complete advertisement on client's declared support
        client_caps = params.client_capabilities
        turn_complete = bool(client_caps.turn_complete) if client_caps is not None else False
        # Derive audio/image modality support from the default agent's model
        # capabilities. When capabilities are not declared (None), default to
        # True (optimistic) to preserve backward-compatible behavior.
        audio_prompts = True
        image_prompts = True
        from wolfharness.agents.native_agent.agent import Agent
        from wolfharness.models.model_configs import BaseModelConfig

        # Derive modality support from the default agent's model config.
        # When the agent or its config is unavailable (e.g. test mocks),
        # fall back to True (optimistic) to preserve backward-compatible behavior.
        if (
            self.default_agent is not None
            and isinstance(self.default_agent, Agent)
            and self.default_agent.config is not None
        ):
            model_config = self.default_agent.config.model
            if isinstance(model_config, BaseModelConfig) and model_config.capabilities is not None:
                caps = model_config.capabilities
                audio_prompts = caps.audio_input if caps.audio_input is not None else True
                image_prompts = caps.image_input if caps.image_input is not None else True
        return InitializeResponse.create(
            protocol_version=version,
            name="wolfharness",
            title="AgentPool",
            version=_version("wolfharness"),
            load_session=True,
            list_sessions=True,
            resume_session=True,
            close_session=True,
            fork_session=True,
            http_mcp_servers=True,
            sse_mcp_servers=True,
            acp_mcp_servers=True,
            audio_prompts=audio_prompts,
            embedded_context_prompts=True,
            image_prompts=image_prompts,
            providers=True,
            turn_complete=turn_complete,
        )

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        """Create a new session."""
        from wolfharness.agents.acp_agent import ACPAgent as ACPAgentClient

        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        logger.info("Creating new session", default_agent=self.default_agent.name)
        try:
            session_id = await self.session_manager.create_session(
                agent_name=self.default_agent.name,
                cwd=params.cwd,
                client=self.client,
                acp_agent=self,
                mcp_servers=params.mcp_servers,
                client_capabilities=self.client_capabilities,
                client_info=self.client_info,
                subagent_display_mode=self.subagent_display_mode,
                raw_input_mode=self.raw_input_mode,
                connection_id=self._get_connection_id(),
            )
            state: SessionModeState | None = None
            models: SessionModelState | None = None
            config_options: list[SessionConfigOption] = []

            if session := self.session_manager.get_session(session_id):
                if isinstance(session.agent, ACPAgentClient):
                    # Nested ACP agent - pass through its state directly
                    if session.agent._state:
                        models = session.agent._state.models
                        state = session.agent._state.modes
                    # Also get config_options from nested agent
                    config_options = await get_session_config_options(session.agent)
                else:
                    # Use unified helpers for all other agents
                    models = await get_session_model_state(
                        session.agent, provider_router=self.provider_router
                    )
                    state = await get_session_mode_state(session.agent)
                    config_options = await get_session_config_options(session.agent)
        except Exception:
            logger.exception("Failed to create new session")
            raise
        else:
            # Schedule available commands update after session response is returned
            if session := self.session_manager.get_session(session_id):
                # Schedule task to run after response is sent
                self._task_group.start_soon(session.send_available_commands_update)
                self._task_group.start_soon(session.agent.load_rules, session.cwd)
                self._task_group.start_soon(session._register_prompt_hub_commands)
                # Determine whether to load client skills
                # None means "use manifest's include_default setting"
                should_load_skills = self.load_skills
                if (
                    should_load_skills is None
                    and self.host_context
                    and self.host_context.manifest
                    and self.host_context.manifest.skills is not None
                ):
                    should_load_skills = self.host_context.manifest.skills.include_default
                elif should_load_skills is None:
                    should_load_skills = True  # Fallback default

                if should_load_skills:
                    self._task_group.start_soon(session.init_client_skills)
            logger.info("Created session", session_id=session_id)

            return NewSessionResponse(
                session_id=session_id,
                modes=state,
                models=models,
                config_options=config_options if config_options else None,
            )

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        """Load an existing session from storage.

        Delegates to the agent's load_session method, which populates agent.conversation.
        Then replays the conversation to the client via ACP notifications.
        """
        from wolfharness.agents.acp_agent import ACPAgent as ACPAgentClient

        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        try:
            # Get or resume session from storage
            session = self.session_manager.get_session(params.session_id)
            if not session:
                # Check if the session is closed in storage before attempting
                # to resume. A closed session should not be loadable via
                # session/load — clients must use session/resume instead.
                store = self.session_manager.session_store
                stored_data = None
                if store is not None:
                    try:
                        stored_data = await store.load_session(params.session_id)
                    except Exception:
                        logger.exception(
                            "Failed to check session status in store",
                            session_id=params.session_id,
                        )
                if stored_data is not None and stored_data.status == "closed":
                    # Allow loading closed sessions for read-only history
                    # replay (e.g. viewing team member conversations after
                    # team_delete).  resume_session will load the
                    # conversation history without starting a new run.
                    logger.info(
                        "Loading closed session for history replay",
                        session_id=params.session_id,
                    )
                session = await self.session_manager.resume_session(
                    session_id=params.session_id,
                    client=self.client,
                    acp_agent=self,
                    mcp_servers=params.mcp_servers,
                    client_capabilities=self.client_capabilities,
                    client_info=self.client_info,
                    subagent_display_mode=self.subagent_display_mode,
                    connection_id=self._get_connection_id(),
                )

            if not session:
                logger.error("Failed to load session")
                return LoadSessionResponse()

            # Replay loaded conversation to client via ACP notifications
            if msgs := session.agent.conversation.chat_messages:
                model_messages: list[ModelMessage] = []
                for chat_msg in msgs:
                    if chat_msg.messages:
                        model_messages.extend(chat_msg.messages)
                await session.notifications.replay(model_messages)
                logger.info(
                    "Conversation replayed",
                    session_id=params.session_id,
                    message_count=len(model_messages),
                )
            mode_state: SessionModeState | None = None
            models: SessionModelState | None = None
            if isinstance(session.agent, ACPAgentClient) and session.agent._state:
                mode_state = session.agent._state.modes
                models = session.agent._state.models
            config_opts = await get_session_config_options(session.agent)
            # Schedule post-load tasks
            self._task_group.start_soon(session.send_available_commands_update)
            self._task_group.start_soon(session.agent.load_rules, session.cwd)
            logger.info("Session loaded", session_id=params.session_id)
            return LoadSessionResponse(models=models, modes=mode_state, config_options=config_opts)
        except RequestError:
            raise
        except Exception:
            logger.exception("Failed to load session", session_id=params.session_id)
            return LoadSessionResponse()

    async def list_sessions(self, params: ListSessionsRequest) -> ListSessionsResponse:
        """List available sessions.

        Delegates to the current agent's list_sessions method which handles
        fetching sessions from storage with proper titles.

        Uses a short TTL cache to avoid redundant expensive storage reads
        when clients request the list multiple times in quick succession.
        """
        import time

        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        # Return cached result if fresh (within 10 seconds)
        cache_ttl = 10.0
        now = time.monotonic()
        if self._sessions_cache and (now - self._sessions_cache_time) < cache_ttl:
            logger.debug("Returning cached sessions list", count=len(self._sessions_cache.sessions))
            return self._sessions_cache

        # Get agent from first active session, or fall back to default
        first_session = next(iter(self.session_manager._acp_sessions.values()), None)
        agent = first_session.agent if first_session else self.default_agent
        try:
            logger.info("Listing sessions for agent", agent_name=agent.name)
            agent_sessions = await agent.list_sessions()
            logger.info("Agent returned sessions", count=len(agent_sessions))
            sessions = [to_session_info(s) for s in agent_sessions]
            logger.info("Listed sessions", count=len(sessions))
            response = ListSessionsResponse(sessions=sessions)
        except Exception:
            logger.exception("Failed to list sessions")
            return ListSessionsResponse(sessions=[])
        else:
            # Cache the result
            self._sessions_cache = response
            self._sessions_cache_time = now
            return response

    async def fork_session(self, params: ForkSessionRequest) -> ForkSessionResponse:
        """Fork an existing session.

        Creates a new session with the same state as the original.
        UNSTABLE: This feature is not part of the spec yet.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        logger.info("Forking session", session_id=params.session_id)
        # For now, just create a new session - full fork implementation would copy state
        session_id = await self.session_manager.create_session(
            agent_name=self.default_agent.name,
            cwd=params.cwd,
            client=self.client,
            acp_agent=self,
            mcp_servers=params.mcp_servers,
            client_capabilities=self.client_capabilities,
            client_info=self.client_info,
            subagent_display_mode=self.subagent_display_mode,
            raw_input_mode=self.raw_input_mode,
            connection_id=self._get_connection_id(),
        )
        return ForkSessionResponse(session_id=session_id)

    async def resume_session(self, params: ResumeSessionRequest) -> ResumeSessionResponse:
        """Resume an existing session without replaying history.

        Restores session context, creates a per-session agent with loaded
        conversation history, and returns session state.

        UNSTABLE: This feature is not part of the spec yet.
        """
        from wolfharness.agents.acp_agent import ACPAgent as ACPAgentClient

        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        try:
            session = self.session_manager.get_session(params.session_id)
            if not session:
                session = await self.session_manager.resume_session(
                    session_id=params.session_id,
                    client=self.client,
                    acp_agent=self,
                    mcp_servers=params.mcp_servers,
                    client_capabilities=self.client_capabilities,
                    client_info=self.client_info,
                    subagent_display_mode=self.subagent_display_mode,
                    connection_id=self._get_connection_id(),
                )

            if not session:
                logger.error("Failed to resume session")
                return ResumeSessionResponse()

            if self.host_context is None:
                logger.error("Agent pool not available")
                return ResumeSessionResponse()
            session_pool = self.host_context.session_pool
            if session_pool is not None:
                try:
                    await session_pool.create_session(
                        params.session_id, cwd=session.cwd or params.cwd
                    )
                    await session_pool.sessions.get_or_create_session_agent(params.session_id)
                    # MCP tools are handled via McpConfigSnapshot →
                    # get_capabilities() → MCPToolset, not through
                    # agent.tools.providers.
                except Exception:
                    logger.exception(
                        "Failed to create per-session agent during resume",
                        session_id=params.session_id,
                    )

            models: SessionModelState | None = None
            mode_state: SessionModeState | None = None
            config_options: list[SessionConfigOption] = []
            if isinstance(session.agent, ACPAgentClient) and session.agent._state:
                models = session.agent._state.models
                mode_state = session.agent._state.modes
            else:
                models = await get_session_model_state(
                    session.agent, provider_router=self.provider_router
                )
                mode_state = await get_session_mode_state(session.agent)
                config_options = await get_session_config_options(session.agent)

            self._task_group.start_soon(session.send_available_commands_update)
            self._task_group.start_soon(session.agent.load_rules, session.cwd)
            logger.info("Session resumed", session_id=params.session_id)
            return ResumeSessionResponse(
                models=models, modes=mode_state, config_options=config_options
            )

        except Exception:
            logger.exception("Failed to resume session", session_id=params.session_id)
            return ResumeSessionResponse()

    async def authenticate(self, params: AuthenticateRequest) -> None:
        """Authenticate with the agent."""
        logger.info("Authentication requested", method_id=params.method_id)

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        """Process a prompt request."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        # Delegate to SessionPool-backed handler when feature flag is enabled
        if self._protocol_handler is not None:
            # Extract W3C trace context from _meta to link spans to the client's trace
            meta = params.field_meta or {}
            context = TraceContextTextMapPropagator().extract(meta)
            token = attach(context)
            # Extract delivery hint from _meta — "steer" → asap, "followup" → when_idle
            delivery: str | None = None
            if isinstance(meta, dict):
                raw_delivery = meta.get("delivery")
                if isinstance(raw_delivery, str) and raw_delivery in ("steer", "followup"):
                    delivery = raw_delivery
            try:
                with logfire.span(
                    "acp.agent.handle_prompt",
                    session_id=params.session_id,
                ):
                    return await self._protocol_handler.handle_prompt(
                        params.session_id,
                        params.prompt,
                        delivery=delivery,
                    )
            finally:
                detach(token)
        raise RuntimeError("No protocol handler configured for prompt processing")

    async def close_session(self, params: CloseSessionRequest) -> CloseSessionResponse:
        """Stop an active session and free its resources.

        Cancels any ongoing work (like session/cancel) and then
        closes the session and releases all associated resources.
        """
        # Delegate to SessionPool-backed handler when feature flag is enabled
        if self._protocol_handler is not None:
            await self._protocol_handler.close_session(params.session_id)
            # Handler returns early when per-agent canary is off;
            # legacy cleanup below still runs for those agents.
            if self.default_agent.metadata.get("use_session_pool", False):
                return CloseSessionResponse()

        logger.info("Stopping session", session_id=params.session_id)
        try:
            # Cancel ongoing work first
            if session := self.session_manager.get_session(params.session_id):
                await session.cancel()
            # Close and release session resources
            await self.session_manager.close_session(params.session_id)
            logger.info("Session stopped", session_id=params.session_id)
        except Exception:
            logger.exception("Failed to stop session", session_id=params.session_id)
        return CloseSessionResponse()

    async def cancel(self, params: CancelNotification) -> None:
        """Cancel operations for a session."""
        logger.info("Cancelling session", session_id=params.session_id)
        try:
            if self._protocol_handler is not None:
                await self._protocol_handler.cancel_session(params.session_id)
            elif session := self.session_manager.get_session(params.session_id):
                await session.cancel()
            logger.info("Cancelled operations", session_id=params.session_id)
        except Exception:
            logger.exception("Failed to cancel session", session_id=params.session_id)

    async def list_providers(self, params: ListProvidersRequest) -> ListProvidersResponse:
        """List available LLM providers."""
        providers = self.provider_router.get_providers()
        return ListProvidersResponse(providers=providers)

    async def set_provider(self, params: SetProvidersRequest) -> SetProvidersResponse:
        """Configure an LLM provider."""
        try:
            await self.provider_router.set_provider_override(
                params.id, base_url=params.base_url, api_key_id=None
            )
        except ValueError as e:
            from acp.exceptions import RequestError

            raise RequestError.invalid_params({"id": params.id}) from e
        return SetProvidersResponse()

    async def disable_provider(self, params: DisableProvidersRequest) -> DisableProvidersResponse:
        """Disable an LLM provider."""
        try:
            await self.provider_router.disable_provider(params.id)
        except ValueError as e:
            from acp.exceptions import RequestError

            raise RequestError.invalid_params({"id": params.id}) from e
        return DisableProvidersResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Handle extension methods.

        Args:
            method: The extension method name.
            params: Method parameters.

        Returns:
            Response dictionary.
        """
        match method:
            case "mcp/message":
                connection_id = params.get("connectionId", "")
                conn = self._mcp_manager.get_connection(connection_id)
                if conn is not None:
                    # Pass the full flattened ACP params to handle_client_message.
                    # handle_client_message will reconstruct the inner JSON-RPC
                    # message from the flattened format (per MCP-over-ACP RFD).
                    self._task_group.start_soon(conn.handle_client_message, params)
                else:
                    logger.warning(
                        "Received MCP message for unknown connection",
                        connection_id=connection_id,
                    )
                return {}
            case _:
                return {}

    async def connect_acp_mcp_server(
        self,
        server: AcpMcpServer,
        session_id: str,
    ) -> tuple[str, int]:
        """Connect to an ACP-transport MCP server by requesting connection from client.

        Initiates mcp/connect to the client per ACP spec. The client returns a
        connectionId which is used to establish the local AcpMcpConnection.
        A per-session stream pair is registered on the connection, and the
        session-to-connection mapping is recorded in the connection manager's
        reverse index for cleanup tracking.

        Args:
            server: The ACP MCP server configuration.
            session_id: The ACP session ID requesting the connection.

        Returns:
            A tuple of ``(connection_id, session_key)`` where connection_id is
            the string returned by the client and session_key is the int key
            for the per-session stream pair registered on the AcpMcpConnection.

        Raises:
            ValueError: If the client does not return a connectionId.
            TimeoutError: If the client does not respond to mcp/connect within 300s.
        """
        params = {
            "server": server.model_dump(by_alias=True, exclude_none=True),
            "acpId": server.id,
        }
        with anyio.fail_after(300):
            response = await self.client.send_request("mcp/connect", params)
        connection_id = str(response.get("connectionId", ""))
        if not connection_id:
            msg = "Client did not return connectionId for mcp/connect"
            raise ValueError(msg)

        async def send_to_client(message: dict[str, Any]) -> Any:
            # message is already wrapped as {"connectionId": conn_id, "message": mcp_msg}
            # by AcpMcpConnection.send_to_client. Pass through directly.
            with anyio.fail_after(300):
                return await self.client.send_request("mcp/message", message)

        conn = await self._mcp_manager.create_connection(connection_id, server, send_to_client)
        _pair, session_key = conn.register_session()
        self._mcp_manager.register_session_connection(session_id, connection_id, session_key)
        logger.info(
            "ACP MCP server connected",
            server_name=server.name,
            connection_id=connection_id,
            session_id=session_id,
            session_key=session_key,
        )
        return connection_id, session_key

    async def disconnect_acp_mcp_server(self, connection_id: str) -> None:
        """Disconnect from an ACP-transport MCP server.

        Sends mcp/disconnect to the client and cleans up the local connection.

        Args:
            connection_id: The connection ID to disconnect.
        """
        try:
            await self.client.send_request("mcp/disconnect", {"connectionId": connection_id})
        except Exception:
            logger.exception(
                "Failed to send mcp/disconnect to client",
                connection_id=connection_id,
            )
        await self._mcp_manager.remove_connection(connection_id)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    async def close(self) -> None:
        """Close the agent and clean up all resources."""
        logger.info("Closing AgentPoolACPAgent")
        try:
            # Notify client to disconnect all ACP MCP servers before closing locally
            for conn_id in list(self._mcp_manager.get_connection_ids()):
                try:
                    await self.disconnect_acp_mcp_server(conn_id)
                except Exception:
                    logger.exception(
                        "Failed to disconnect ACP MCP server during shutdown",
                        connection_id=conn_id,
                    )
            await self._mcp_manager.close_all()
        except Exception:
            logger.exception("Failed to close MCP connections during agent shutdown")
        await self._task_group.close()

    async def set_session_mode(
        self, params: SetSessionModeRequest
    ) -> SetSessionModeResponse | None:
        """Set the session mode (change tool confirmation level).

        Calls set_mode directly on the agent with the mode_id, allowing the agent
        to handle mode-specific logic (e.g., acceptEdits auto-allowing edit tools).
        """
        from wolfharness.agents.acp_agent import ACPAgent as ACPAgentClient

        session = self.session_manager.get_session(params.session_id)
        if not session:
            logger.warning("Session not found for mode switch", session_id=params.session_id)
            return None
        try:
            # Call set_mode directly - agent handles mode-specific logic
            await session.agent.set_mode(params.mode_id, category_id="mode")
            # Update stored mode state for ACPAgent
            if (
                isinstance(session.agent, ACPAgentClient)
                and session.agent._state
                and session.agent._state.modes
            ):
                session.agent._state.modes.current_mode_id = params.mode_id

            logger.info("Set mode", mode_id=params.mode_id, session_id=params.session_id)
            return SetSessionModeResponse()

        except Exception:
            logger.exception("Failed to set session mode", session_id=params.session_id)
            return None

    async def set_session_model(
        self, params: SetSessionModelRequest
    ) -> SetSessionModelResponse | None:
        """Set the session model.

        Changes the model for the active agent in the session.
        Validates that the requested model is available via agent.get_available_models()
        OR is a configured variant in the manifest.
        """
        session = self.session_manager.get_session(params.session_id)
        if not session:
            msg = "Session not found for model switch"
            logger.warning(msg, session_id=params.session_id)
            return None
        try:
            # Build list of valid model IDs from both tokonomics and configured variants
            valid_model_ids: set[str] = set()

            # Get tokonomics models
            if toko_models := await session.agent.get_available_models():
                valid_model_ids.update(
                    m.id_override if m.id_override else m.id for m in toko_models
                )

            # Get configured variants from manifest
            ctx = session.agent.host_context
            manifest = ctx.manifest if ctx else None
            if manifest and manifest.model_variants:
                valid_model_ids.update(manifest.model_variants.keys())
                # Also include resolved identifiers from StringModelConfig variants,
                # since configOptions sends resolved identifiers (e.g., "openai-chat:svc/glm-4.7")
                # while model_variants.keys() are variant names (e.g., "glm47").
                from wolfharness.models.model_configs import StringModelConfig

                for config in manifest.model_variants.values():
                    if isinstance(config, StringModelConfig):
                        valid_model_ids.add(config.identifier)

            # Validate the requested model
            if params.model_id not in valid_model_ids:
                logger.warning(
                    "Model not in available models",
                    model_id=params.model_id,
                    available=list(valid_model_ids),
                )
                return None

            # Set the model on the agent (all agents now have async set_model)
            await session.agent.set_model(params.model_id)
            logger.info("Set model", model_id=params.model_id, session_id=params.session_id)
            return SetSessionModelResponse()
        except (AttributeError, NotImplementedError) as e:
            logger.warning(
                "Agent does not support model switching",
                error=str(e),
                session_id=params.session_id,
            )
            return None
        except Exception:
            logger.exception("Failed to set session model", session_id=params.session_id)
            return None

    async def _swap_session_agent(self, session_id: str, new_agent_name: str) -> dict[str, bool]:
        """Swap the active agent for a session.

        Acquires the session agent lock to prevent concurrent swaps,
        then delegates to the session's switch_active_agent method.

        Args:
            session_id: The session to swap agent for.
            new_agent_name: Name of the agent to switch to.

        Returns:
            Dict with success flag.

        Raises:
            RequestError: If swap fails (session not found, agent unknown,
                or prompt is active).
        """
        from acp.exceptions import RequestError

        session = self.session_manager.get_session(session_id)
        if not session:
            msg = {"session_id": session_id, "reason": "Session not found"}
            raise RequestError.invalid_params(msg)

        # Block swap during active prompt
        if session.is_busy:
            msg = {"session_id": session_id, "reason": "Prompt active"}
            raise RequestError.invalid_params(msg)

        # Ensure lock exists for this session
        if session_id not in self._session_agent_locks:
            self._session_agent_locks[session_id] = asyncio.Lock()

        async with self._session_agent_locks[session_id]:
            await session.switch_active_agent(new_agent_name)

        return {"success": True}

    async def set_session_config_option(
        self, params: SetSessionConfigOptionRequest
    ) -> SetSessionConfigOptionResponse | None:
        """Set a session config option.

        Forwards the config option change to the agent's set_mode method
        or handles agent_role swap, then returns the updated config options.
        """
        session = self.session_manager.get_session(params.session_id)
        if not session or not session.agent:
            msg = "Session not found for config option change"
            logger.warning(msg, session_id=params.session_id)
            return None
        logger.info(
            "Set config option",
            config_id=params.config_id,
            value=params.value,
            session_id=params.session_id,
        )
        try:
            if params.config_id == "agent_role":
                await self._set_agent_role(session, params.value)
            else:
                # Forward to agent's set_mode method
                # config_id maps to category_id, value maps to mode_id
                await session.agent.set_mode(params.value, category_id=params.config_id)
            # Return updated config options
            config_options = await get_session_config_options(session.agent)
            return SetSessionConfigOptionResponse(config_options=config_options)
        except Exception:
            logger.exception("Failed to set session config option", session_id=params.session_id)
            return None

    async def _set_agent_role(self, session: Any, agent_name: str) -> None:
        """Validate and swap to the requested agent role.

        Args:
            session: The active session.
            agent_name: Target agent name.

        Raises:
            RequestError: If agent name is not in pool.
        """
        from acp.exceptions import RequestError

        ctx = session.agent.host_context
        if ctx is None or agent_name not in ctx.manifest.agents:
            msg = {"agent_role": agent_name, "reason": "Unknown agent"}
            raise RequestError.invalid_params(msg)
        await self._swap_session_agent(session.session_id, agent_name)

    async def swap_pool(self, config_path: str, agent_name: str | None = None) -> list[str]:
        """Swap the agent pool with a new one from configuration.

        This coordinates the full pool swap:
        1. Acquires session_manager._lock to serialize with create_session()
        2. Clears all active sessions while holding the lock
        3. Closes all sessions (outside the lock)
        4. Cleans up all per-session agents
        5. Delegates to server.swap_pool() for pool lifecycle
        6. Updates internal references and cached agent config

        Args:
            config_path: Path to the new agent configuration file
            agent_name: Optional specific agent name to use as default

        Returns:
            List of agent names in the new pool

        Raises:
            RuntimeError: If server reference is not set
            ValueError: If config is invalid or agent not found
        """
        if not self.server:
            msg = "Server reference not set - cannot swap pool"
            raise RuntimeError(msg)

        logger.info("Swapping pool", config_path=config_path, agent=agent_name)

        # 1. Copy and clear all active sessions
        sessions = list(self.session_manager._acp_sessions.values())
        self.session_manager._acp_sessions.clear()
        # 2. Set swap flag to prevent new session creation during swap
        self._swap_in_progress = True

        # 3. Close all sessions (may take time)
        for session in sessions:
            try:
                await session.close()
            except Exception:
                logger.exception(
                    "Error closing session during pool swap",
                    session_id=session.session_id,
                )

        # 4. Disconnect all ACP MCP servers before cleaning up session agents
        try:
            for conn_id in list(self._mcp_manager.get_connection_ids()):
                try:
                    await self.disconnect_acp_mcp_server(conn_id)
                except Exception:
                    logger.exception(
                        "Failed to disconnect ACP MCP server during pool swap",
                        connection_id=conn_id,
                    )
            await self._mcp_manager.close_all()
        except Exception:
            logger.exception("Failed to close MCP connections during pool swap")

        try:
            # 5. Swap pool
            new_agent = await self.server.swap_pool(config_path, agent_name)

            # 6. Update cached agent config from new pool
            ctx = new_agent.host_context
            new_pool = new_agent._agent_pool
            if ctx is None or new_pool is None:
                msg = "New agent has no associated pool"
                raise RuntimeError(msg)

            # Re-resolve _agent_config from the new pool's manifest
            if ctx.main_agent_name and ctx.main_agent_name in ctx.manifest.agents:
                cfg = ctx.manifest.agents[ctx.main_agent_name]
                from wolfharness.models.agents import NativeAgentConfig

                if isinstance(cfg, NativeAgentConfig):
                    if cfg.name is None:
                        cfg = cfg.model_copy(update={"name": ctx.main_agent_name})
                    self._agent_config = cfg
            elif ctx.manifest.agents:
                cfg = next(iter(ctx.manifest.agents.values()))
                from wolfharness.models.agents import NativeAgentConfig

                if isinstance(cfg, NativeAgentConfig):
                    self._agent_config = cfg
            else:
                self._agent_config = None

            # 7. Update default_agent reference and pool
            self.default_agent = new_agent
            self.session_manager._pool = new_pool

            # 8. Invalidate sessions cache
            self._sessions_cache = None

            agent_names = list(ctx.manifest.agents.keys())
            logger.info("Pool swap complete", agent_names=agent_names)
            return agent_names
        finally:
            # 9. Clear swap flag - new sessions can now be created
            # This MUST be in finally to prevent permanent blocking on error
            self._swap_in_progress = False
