"""Server state management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from wolfharness import log
from wolfharness.diagnostics.lsp_manager import LSPManager
from wolfharness_server.opencode_server.models import SessionStatus
from wolfharness_server.opencode_server.provider_auth import (
    ProviderAuthService,
    create_default_auth_service,
)
from wolfharness_storage.opencode_provider import helpers


logger = log.get_logger(__name__)

if TYPE_CHECKING:
    from fsspec.asyn import AsyncFileSystem
    from slashed import CommandStore

    from wolfharness.agents.base_agent import BaseAgent
    from wolfharness.delegation import AgentPool
    from wolfharness.models.model_configs import AnyModelConfig
    from wolfharness.orchestrator.core import SessionController
    from wolfharness.storage import StorageManager
    from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider
    from wolfharness_server.opencode_server.models import (
        Config,
        Event,
        MessageWithParts,
        QuestionInfo,
        Session,
    )
    from wolfharness_server.opencode_server.models.question import QuestionToolInfo
    from wolfharness_server.opencode_server.routes.global_routes import GlobalEventFactory

# Type alias for async callback
OnFirstSubscriberCallback = Callable[[], Coroutine[Any, Any, None]]


@dataclass
class PendingQuestion:
    """Pending question awaiting user response."""

    session_id: str
    """Session that owns this question."""

    questions: list[QuestionInfo]
    """Questions to ask."""

    future: asyncio.Future[list[list[str]]]
    """Future that resolves when user answers."""

    tool: QuestionToolInfo | None = None
    """Optional tool context."""


@dataclass
class ServerState:
    """Shared state for the OpenCode server.

    Uses agent.host_context for session persistence and storage.
    In-memory state tracks active sessions and runtime data.
    """

    working_dir: str
    agent: BaseAgent[Any, Any]
    start_time: float = field(default_factory=time.time)
    config: Config | None = None
    sessions: dict[str, Session] = field(default_factory=dict)
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    agent_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reverted_messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)
    messages: dict[str, list[MessageWithParts]] = field(default_factory=dict)
    event_subscribers: list[asyncio.Queue[Event]] = field(default_factory=list)
    _event_factory: GlobalEventFactory | None = field(default=None, repr=False)
    on_first_subscriber: OnFirstSubscriberCallback | None = None
    _first_subscriber_triggered: bool = field(default=False, repr=False)
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _run_handles: dict[str, Any] = field(default_factory=dict)
    event_managers: dict[str, Any] = field(default_factory=dict)
    auth_service: ProviderAuthService = field(default_factory=create_default_auth_service)
    skill_bridge: Any = field(default=None)
    command_store: CommandStore | None = field(default=None)
    _skill_change_task: Any = field(default=None, repr=False)
    _mcp_tool_change_task: Any = field(default=None, repr=False)
    session_pool_integration: Any = field(default=None)
    session_controller: SessionController | None = field(default=None)
    event_bridge: Any = field(default=None, repr=False)
    _shell_env: Any = field(default=None, repr=False)

    @staticmethod
    def parse_model_info(model_name: str | None) -> tuple[str, str]:
        """Parse a model string into (model_id, provider_id).

        Splits on the first colon (e.g. ``"openai:gpt-4o"`` →
        ``("gpt-4o", "openai")``). Falls back to ``("default",
        "wolfharness")`` when the model name is ``None`` or missing a
        provider prefix.

        Args:
            model_name: The full model string (e.g. ``"openai:gpt-4o"``).

        Returns:
            Tuple of ``(model_id, provider_id)``.
        """
        if model_name and ":" in model_name:
            provider, model = model_name.split(":", 1)
            return model, provider
        return "default", "wolfharness"

    @property
    def model_variants(self) -> dict[str, AnyModelConfig]:
        """Return configured model variants from the pool manifest.

        Returns an empty dict when the pool or manifest is not available.
        """
        if self._pool is not None and self._pool.manifest is not None:
            return self._pool.manifest.model_variants
        return {}

    def resolve_default_model_info(self) -> tuple[str, str]:
        """Resolve default (model_id, provider_id) from the configured agent.

        Priority:
        1. If agent's resolved model matches a configured variant → use variant
           name as model_id and configured provider as provider_id.
        2. Parses ``self.agent.model_name`` (e.g. ``"openai:gpt-4o"``) by
           splitting on the first colon.
        3. Falls back to ``("default", "wolfharness")``.

        Returns:
            Tuple of ``(model_id, provider_id)``.
        """
        agent_model = self.agent.model_name
        # Try variant-aware resolution first
        if agent_model and self._pool is not None:
            manifest = self._pool.manifest
            if manifest and manifest.model_variants:
                from wolfharness_server.shared.model_utils import (
                    _extract_provider,
                    _find_variant_name,
                )

                matched = _find_variant_name(manifest.model_variants, agent_model)
                if matched:
                    config = manifest.model_variants[matched]
                    provider = _extract_provider(config)
                    return matched, provider

        return self.parse_model_info(agent_model)

    def __post_init__(self) -> None:
        """Initialize derived state."""
        self.lsp_manager = LSPManager(env=self.agent.env)
        self.lsp_manager.register_defaults()
        # Cache non-session-scoped dependencies directly so they remain
        # accessible even after the shared ``self.agent`` is removed in a
        # later migration step.
        self._pool: AgentPool[Any] | None = self.agent._agent_pool
        self._storage: StorageManager | None = self.agent.storage

        # Create a standalone execution environment for shell commands.
        # This preserves direct execution semantics (no SessionPool turn)
        # and avoids depending on the shared agent for shell operations.
        agent_env = self.agent.env
        match agent_env:
            case _ if hasattr(agent_env, "cwd"):
                from exxec import LocalExecutionEnvironment

                self._shell_env = LocalExecutionEnvironment(cwd=agent_env.cwd)
            case _:
                # Fallback: reference the same env (preserves remote env support)
                self._shell_env = agent_env

        # Instantiate the OpenCodeEventBridge when a SessionController is
        # available.  The bridge dual-publishes events to SSE subscribers
        # (backward compat) and the SessionPool EventBus.
        if self.session_controller is not None:
            event_bus = None
            if self._pool is not None:
                session_pool = getattr(self._pool, "session_pool", None)
                if session_pool is not None:
                    event_bus = getattr(session_pool, "event_bus", None)

            if event_bus is not None:
                from wolfharness_server.opencode_server.event_bridge import (
                    OpenCodeEventBridge,
                )

                self.event_bridge = OpenCodeEventBridge(self, event_bus)

    def get_event_factory(self) -> GlobalEventFactory:
        """Get or lazily create the GlobalEventFactory for event wrapping.

        The factory is created on first access using the working directory
        and computed project ID, then cached for the server's lifetime.
        Imports GlobalEventFactory locally to avoid circular imports.
        """
        from wolfharness_server.opencode_server.routes.global_routes import GlobalEventFactory

        if self._event_factory is None:
            directory = self.base_path
            project = helpers.compute_project_id(directory)
            self._event_factory = GlobalEventFactory(
                directory=directory,
                project=project,
            )
        return self._event_factory

    def ensure_runtime_session_state(self, session_id: str) -> None:
        """Ensure in-memory runtime buckets exist for a session.

        This is used both for brand-new sessions and for sessions reloaded from
        persisted storage after a server restart. Cold-start recovery should not
        depend on individual routes remembering to initialize each bucket.
        """
        self.reverted_messages.setdefault(session_id, [])
        self.messages.setdefault(session_id, [])

    @property
    def fs(self) -> AsyncFileSystem:
        """Get the fsspec filesystem from the agent's environment."""
        return self.agent.env.get_fs()

    @property
    def shell_env(self) -> Any:
        """Get the standalone execution environment for shell commands.

        Returns the cached execution environment that was created from
        ``self.agent.env`` during ``__post_init__``.  This avoids
        depending on the shared agent for shell execution.
        """
        return self._shell_env

    @property
    def base_path(self) -> str:
        """Get the resolved OpenCode project root for routing and file operations.

        OpenCode routes SSE events against the server/project directory the client
        attached to, not an agent-specific execution sandbox. Agent execution
        environments may override `env.cwd` for tool isolation, but routing
        metadata must remain anchored to the server's configured `working_dir`.
        """
        return str(Path(self.working_dir).resolve())

    @property
    def is_local_fs(self) -> bool:
        """Check if the filesystem is local."""
        from fsspec.implementations.local import LocalFileSystem

        return isinstance(self.fs, LocalFileSystem)

    @property
    def pool(self) -> AgentPool[Any]:
        """Get the agent pool.

        Returns the cached pool reference that was resolved from
        ``self.agent.host_context`` during ``__post_init__``.  This avoids
        depending on the shared agent for non-session-scoped access.

        Raises:
            AttributeError: If the pool was not set during ``__post_init__``
                (e.g. in test environments without a real AgentPool).
        """
        if self._pool is None:
            msg = "ServerState has no agent_pool set"
            raise AttributeError(msg)
        return self._pool

    @property
    def pool_or_none(self) -> AgentPool[Any] | None:
        """Get the agent pool, or ``None`` if not set.

        Use this in code paths that must gracefully handle the absence of
        a pool (e.g. test fixtures, optional features).  Production route
        handlers should use :attr:`pool` instead.
        """
        return self._pool

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for the given session.

        Per-session locks ensure that messages to the same session
        are processed sequentially, preventing race conditions and
        event interleaving.

        Args:
            session_id: The session ID to get the lock for.

        Returns:
            asyncio.Lock: The lock for the session.
        """
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]

    def ensure_input_provider(self, session_id: str) -> OpenCodeInputProvider:
        """Get or create the OpenCode input provider for a session.

        Stores the provider on SessionState (via SessionController) when available.
        """
        from wolfharness_server.opencode_server.input_provider import OpenCodeInputProvider

        input_provider = None
        if self.session_controller is not None:
            session = self.session_controller.get_session(session_id)
            if session is not None:
                input_provider = session.input_provider

        if input_provider is None:
            input_provider = OpenCodeInputProvider(self, session_id)
            if self.session_controller is not None:
                session = self.session_controller.get_session(session_id)
                if session is not None:
                    session.input_provider = input_provider
        return input_provider

    @property
    def storage(self) -> StorageManager:
        """Get the storage manager for session persistence.

        Returns the cached storage reference that was resolved from
        ``self.agent.storage`` during ``__post_init__``.  This avoids
        depending on the shared agent for non-session-scoped access.

        Returns:
            StorageManager: The storage manager for session persistence.

        Raises:
            RuntimeError: If agent storage is not initialized.
        """
        if self._storage is None:
            msg = "Agent storage is not initialized"
            raise RuntimeError(msg)
        return self._storage

    def create_background_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task[Any]:
        """Create and track a background task."""
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def cancel_session_pending_questions(self, session_id: str) -> list[str]:
        """Cancel pending questions for a specific session and return their IDs."""
        if self.session_controller is not None:
            return self.session_controller.cancel_session_pending_questions(session_id)
        return []

    def cancel_all_pending_questions(self) -> list[str]:
        """Cancel all pending questions and return their IDs."""
        if self.session_controller is not None:
            return self.session_controller.cancel_all_pending_questions()
        return []

    async def cleanup_tasks(self) -> None:
        """Cancel and wait for all background tasks."""
        for task in self.background_tasks:
            task.cancel()
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

    async def broadcast_event(self, event: Event) -> None:
        """Broadcast an event via the EventBus bridge.

        When :attr:`event_bridge` is present, delegates to the bridge which
        publishes the event to the SessionPool EventBus. Otherwise, the
        event is silently dropped (no event delivery path available).
        """
        if self.event_bridge is not None:
            await self.event_bridge.publish(event)
        else:
            logger.debug(
                "broadcast_event: no event_bridge, skipping event",
                event_type=getattr(event, "type", "unknown"),
            )

    async def mark_session_idle(self, session_id: str) -> None:
        """Mark a session idle and broadcast the matching status events."""
        from wolfharness_server.opencode_server.models import SessionIdleEvent, SessionStatusEvent
        from wolfharness_server.opencode_server.session_pool_integration import set_session_status

        status = SessionStatus(type="idle")
        await set_session_status(self, session_id, status)
        await self.broadcast_event(SessionStatusEvent.create(session_id, status))
        await self.broadcast_event(SessionIdleEvent.create(session_id))

    async def emit_session_turn_complete(self, session_id: str) -> None:
        """Broadcast the per-turn completion signal without changing busy state."""
        from wolfharness_server.opencode_server.models import SessionIdleEvent

        await self.broadcast_event(SessionIdleEvent.create(session_id))
