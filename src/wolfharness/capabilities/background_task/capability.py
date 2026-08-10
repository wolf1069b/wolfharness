"""Background task capability for async task delegation."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from uuid import uuid4
import weakref

import logfire
from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NativeTool,
    NodeResult,
    ProcessHistory,
)
from pydantic_ai.messages import ModelRequest, TextPartDelta, ThinkingPartDelta, UserPromptPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_graph import End

from wolfharness.agents.base_agent import BaseAgent
from wolfharness.agents.context import AgentContext
from wolfharness.agents.events import (
    PartDeltaEvent,
    RunErrorEvent,
    RunFailedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallStartEvent,
)
from wolfharness.capabilities.background_task.manager import TERMINAL_STATES, BackgroundTaskManager
from wolfharness.capabilities.background_task.notification import (
    NotificationBatcher,
    _format_duration,
)
from wolfharness.capabilities.background_task.types import BackgroundTask, SessionTaskState
from wolfharness.orchestrator.core import EventEnvelope
from wolfharness.skills.uri_resolver import ResolvedSkillURI
from wolfharness.tools.exceptions import ToolError
from wolfharness.utils.tool_schema import apply_params_schema, load_tool_schema
from wolfharness_config.context import get_config_dir
from wolfharness_toolsets.builtin.skills import load_skill_for_node


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.run import AgentRunResult
    from schemez.functionschema import OpenAIFunctionDefinition

    from wolfharness.agents.context import AgentRunContext
    from wolfharness.orchestrator.session_pool import SessionPool


MAX_DELEGATION_DEPTH = 5

logger = logging.getLogger(__name__)


class ForceRetrievalMode(enum.Enum):
    """How pending background tasks are retrieved before the agent run ends.

    Modes:
        disabled: No forced retrieval — agent can end with pending tasks.
        tool_choice: Force ``background_output`` via ``tool_choice`` parameter.
            Requires the model/API to support forced tool choice.
        directive: Inject a system-reminder prompt directing the agent to call
            ``background_output``. Works with any model, but not guaranteed.
    """

    disabled = "disabled"
    tool_choice = "tool_choice"
    directive = "directive"

    @classmethod
    def coerce(cls, value: bool | str | ForceRetrievalMode | None) -> ForceRetrievalMode:
        """Coerce legacy bool/str/None values to ForceRetrievalMode.

        Backward compatibility:
            True → tool_choice
            False / None → disabled
            str → match by name
        """
        if value is True:
            return cls.tool_choice
        if value is False or value is None:
            return cls.disabled
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        msg = f"Cannot coerce {value!r} to ForceRetrievalMode"
        raise ValueError(msg)


class DelegationDeps(TypedDict, total=False):
    """Dependencies passed to delegated subagent tasks.

    Known keys are declared explicitly; arbitrary keys from ``ctx.data``
    may also be present due to the spread merge pattern.
    """

    delegation_depth: int


def _generate_task_id(description: str) -> str:
    """Generate a unique task ID using a random hex prefix.

    Args:
        description: Short task description (unused, kept for API compatibility)

    Returns:
        Task ID in format: bg_XXXXXXXXXXXX (12 hex chars)
    """
    return f"bg_{uuid4().hex[:12]}"


class BackgroundTaskCapability(AbstractCapability[AgentContext]):
    """Capability providing background task tools with full lifecycle management.

    Provides:
    - Instructions via ``get_instructions`` (callable: capability desc + available agents,
      resolved per model request)
    - Tools via ``get_toolset`` (task, background_output, background_cancel)
    - Lifecycle via ``before_run`` (per-run init; EventBus cleanup via per-task try/finally
      in ``_run_and_stream``)
    - Ordering via ``get_ordering`` (wrapped_by ProcessHistory, NativeTool)
    """

    def __init__(
        self,
        schemas: dict[str, str] | None = None,
        enabled_tools: list[str] | None = None,
        *,
        force_retrieval: bool | str | ForceRetrievalMode | None = False,
        max_concurrent_tasks: int = 10,
        max_retrieval_retries: int = 3,
        notification_debounce_ms: float = 500,
        notification_deliver_timeout: float = 5.0,
    ) -> None:
        """Initialize the background task capability.

        Args:
            schemas: Optional dictionary mapping tool names to schema file paths.
                Expected keys: "task", "background_output", "background_cancel"
                Example: {"task": "/path/to/task.yaml"}
                Paths are resolved relative to config directory using CONFIG_DIR context.
            enabled_tools: Optional list of tools to enable. If None or empty, all
                tools are enabled.
                Expected values: "task", "background_output", "background_cancel"
                Example: ["task", "background_output"]
            force_retrieval: Controls how pending background tasks are retrieved
                before the agent run ends. Accepts a :class:`ForceRetrievalMode`
                value, a string name (``"disabled"``, ``"tool_choice"``,
                ``"directive"``), or a bool for backward compatibility
                (``True`` → ``"tool_choice"``, ``False`` → ``"disabled"``).
                - ``"disabled"``: No forced retrieval.
                - ``"tool_choice"``: Force ``background_output`` via ``tool_choice``
                  parameter. Requires model/API support for forced tool choice.
                - ``"directive"``: Inject a system-reminder prompt directing the
                  agent to call ``background_output``. Works with any model.
            max_concurrent_tasks: Maximum number of background tasks that may run
                concurrently. Additional tasks queue until a slot is released.
            max_retrieval_retries: Maximum number of times ``after_node_run``
                will intercept ``End`` to inject a retrieval prompt before
                allowing the run to terminate. Prevents infinite loops when
                the model ignores the directive or ``tool_choice`` is not
                honored by the API.
            notification_debounce_ms: Debounce window in milliseconds for
                batching background-task completion notifications.
                Default 500ms.
            notification_deliver_timeout: Timeout in seconds for delivering a
                batched notification. Default 5.0s.
        """
        self._schemas = schemas or {}
        self._enabled_tools = enabled_tools or [
            "task",
            "background_output",
            "background_cancel",
            "steer_task",
        ]
        self._force_retrieval = ForceRetrievalMode.coerce(force_retrieval)
        self._max_retrieval_retries = max_retrieval_retries
        self._notification_debounce_ms = notification_debounce_ms
        self._notification_deliver_timeout = notification_deliver_timeout

        # Schema loading with fail-fast behavior
        self._task_schema: OpenAIFunctionDefinition | None = None
        self._background_output_schema: OpenAIFunctionDefinition | None = None
        self._background_cancel_schema: OpenAIFunctionDefinition | None = None
        self._steer_task_schema: OpenAIFunctionDefinition | None = None

        if schemas:
            if (task_schema_path := schemas.get("task")) is not None:
                self._task_schema = self._resolve_and_load_schema(task_schema_path)

            if (background_output_schema_path := schemas.get("background_output")) is not None:
                self._background_output_schema = self._resolve_and_load_schema(
                    background_output_schema_path
                )

            if (background_cancel_schema_path := schemas.get("background_cancel")) is not None:
                self._background_cancel_schema = self._resolve_and_load_schema(
                    background_cancel_schema_path
                )

            if (steer_task_schema_path := schemas.get("steer_task")) is not None:
                self._steer_task_schema = self._resolve_and_load_schema(steer_task_schema_path)

        # Resolve the actual background_output tool name from schema (or default)
        self._output_tool_name = (
            self._background_output_schema.get("name") if self._background_output_schema else None
        ) or "background_output"

        # Store for per-session state creation
        self._max_concurrent_tasks = max_concurrent_tasks

        # Per-session state store, keyed by run_ctx.run_id (stable UUID string).
        # AgentRunContext is a dataclass (unhashable by default), so we cannot
        # use WeakKeyDictionary.  Instead we key by run_id and rely on
        # before_run() to create fresh state per run.
        self._session_states: dict[str, SessionTaskState] = {}

        # Ephemeral state fallback for contexts where run_ctx is None
        # (e.g. during instruction resolution before a run starts).
        # Keyed by id(ctx) to avoid strong references.
        self._ephemeral_states: dict[int, SessionTaskState] = {}

    @staticmethod
    def _resolve_and_load_schema(schema_path_str: str) -> OpenAIFunctionDefinition:
        """Resolve a schema path relative to CONFIG_DIR and load it.

        Args:
            schema_path_str: The schema file path (absolute or relative to CONFIG_DIR).

        Returns:
            The loaded schema as an OpenAIFunctionDefinition.

        Raises:
            FileNotFoundError: If the schema file doesn't exist.
            ValueError: If the schema file can't be parsed.
        """
        schema_path = Path(schema_path_str)
        if not schema_path.is_absolute():
            config_dir = get_config_dir()
            if config_dir is not None:
                schema_path = Path(str(config_dir)) / schema_path
        result = load_tool_schema(str(schema_path))
        if result is None:
            msg = f"Tool schema at {schema_path} loaded as None"
            raise ValueError(msg)
        return result

    # ---- Session state management ----

    def _get_session_state(self, ctx: RunContext[AgentContext] | AgentContext) -> SessionTaskState:
        """Get or create per-session ``SessionTaskState``.

        Extracts ``AgentRunContext`` from ``ctx`` and uses its ``run_id``
        (a stable UUID string) as the key in ``_session_states``.  When
        ``run_ctx`` is ``None`` (e.g. during instruction resolution before
        a run starts), falls back to an ephemeral state keyed by ``id(ctx)``.

        Args:
            ctx: Either a ``RunContext[AgentContext]`` (tool methods,
                lifecycle hooks) or an ``AgentContext`` (``_task_async``).

        Returns:
            The ``SessionTaskState`` for this session.
        """
        # Extract AgentRunContext from the appropriate ctx type
        if isinstance(ctx, RunContext):
            agent_ctx = ctx.deps
            run_ctx = agent_ctx.run_ctx
            id_key = id(ctx.deps)
        else:
            agent_ctx = ctx
            run_ctx = agent_ctx.run_ctx
            id_key = id(ctx)

        # Normal path: run_ctx is available
        if run_ctx is not None:
            session_key = run_ctx.run_id
            state = self._session_states.get(session_key)
            if state is not None:
                return state

            # Clean up any ephemeral state for this context
            self._ephemeral_states.pop(id_key, None)

            # Extract session_pool for fallback notification delivery
            session_pool = agent_ctx.pool.session_pool if agent_ctx.pool is not None else None

            state = self._create_session_state(run_ctx=run_ctx, session_pool=session_pool)
            self._session_states[session_key] = state
            return state

        # Ephemeral path: run_ctx is None — use id-keyed fallback
        state = self._ephemeral_states.get(id_key)
        if state is not None:
            return state

        state = self._create_session_state(run_ctx=None)
        self._ephemeral_states[id_key] = state
        return state

    def _create_session_state(
        self,
        run_ctx: AgentRunContext | None,
        session_pool: SessionPool | None = None,
    ) -> SessionTaskState:
        """Create a new ``SessionTaskState`` with task manager and batcher.

        Args:
            run_ctx: The ``AgentRunContext`` for this session, or ``None``
                if not yet available (ephemeral state).
            session_pool: Optional ``SessionPool`` for fallback notification
                delivery.  Only used in the normal path where ``run_ctx``
                is not ``None``.

        Returns:
            A new ``SessionTaskState`` instance.
        """
        task_manager = BackgroundTaskManager(max_concurrent_tasks=self._max_concurrent_tasks)

        # Placeholder no-op deliver callback — replaced in normal path
        async def _noop_deliver(*_args: object) -> None:
            pass

        batcher = NotificationBatcher(
            deliver_callback=_noop_deliver,
            debounce_ms=self._notification_debounce_ms,
            deliver_timeout=self._notification_deliver_timeout,
            pending_count_callback=lambda: sum(
                1 for t in task_manager.get_all_tasks() if t.status not in TERMINAL_STATES
            ),
        )

        # In the normal path, wire the real deliver callback with weakref
        if run_ctx is not None:
            batcher.deliver_callback = self._make_deliver_callback(run_ctx, session_pool)

        return SessionTaskState(task_manager=task_manager, batcher=batcher)

    def _make_deliver_callback(
        self,
        run_ctx: AgentRunContext,
        session_pool: SessionPool | None,
    ) -> Callable[[str, list[BackgroundTask], str], Awaitable[None]]:
        """Create a deliver callback for the notification batcher.

        Uses ``weakref.ref(run_ctx)`` to avoid strong references that
        would prevent session cleanup.  The callback:

        1. Calls ``followup()`` FIRST to queue the batched notice for
           the next turn (never ``steer()`` which does mid-turn injection).
        2. THEN pops and sets ``child_done_events`` per-task as a safety
           net (the immediate pop in ``_on_task_completed`` is the
           primary unblock path).

        Args:
            run_ctx: The ``AgentRunContext`` for this session.
            session_pool: Optional ``SessionPool`` for fallback delivery.

        Returns:
            An async callback function for ``NotificationBatcher.deliver_callback``.
        """
        ctx_ref = weakref.ref(run_ctx)

        async def _deliver(
            parent_session_id: str, tasks: list[BackgroundTask], notice: str
        ) -> None:
            rc = ctx_ref()
            if rc is None:
                return  # Dead session

            # 1. Queue notification via followup() (always next-turn, never mid-turn)
            delivered = False
            if rc._run_handle is not None:
                delivered = rc._run_handle.followup(notice) is not None
            elif session_pool is not None:
                result = await session_pool.followup(parent_session_id, notice)
                delivered = result is not None

            if not delivered:
                logger.debug(
                    "deliver_callback could not queue followup for session %s "
                    "(%d tasks) — no active run handle or session pool",
                    parent_session_id,
                    len(tasks),
                )

            # 2. Pop+set child_done_events per-task (safety net for batched path)
            for task in tasks:
                if task.child_session_id is not None:
                    event = rc.child_done_events.pop(task.child_session_id, None)
                    if event is not None:
                        event.set()

        return _deliver

    # ---- Static configuration ----

    def get_instructions(self) -> AgentInstructions[AgentContext] | None:
        """Inject capability description + available agents into system prompt.

        Returns a callable that receives ``RunContext[AgentContext]`` at
        instruction-resolution time (not at init). The callable extracts
        available agents from ``ctx.deps.pool.manifest.agents`` dynamically.

        This is critical: pydantic-ai caches instructions at init time and
        does NOT re-call ``get_instructions()`` per run when ``for_run()`` returns
        self. By returning a callable, the instructions are resolved at
        each model request with fresh pool state.

        Replaces the previous ``prepare_task`` pattern that appended agent info
        to the tool description. System prompt is the architecturally correct
        location for run-level context like available agents.
        """

        def _instructions(ctx: RunContext[AgentContext]) -> str:
            pool = ctx.deps.pool
            if pool is None:
                return "Delegate background tasks to other agents."
            current_agent_name = ctx.deps.node.name if ctx.deps.node else None

            result = "Delegate background tasks to other agents.\n\n# Available Agents:\n"
            for agent_name, agent_config in pool.manifest.agents.items():
                if agent_name == current_agent_name:
                    continue
                node_description = agent_config.description or ""
                result += f"- {agent_name}: {node_description}\n"
            return result

        return _instructions

    def get_toolset(self) -> AgentToolset[AgentContext] | None:
        """Return ``FunctionToolset`` with enabled tools."""
        tools: list[Tool[AgentContext]] = []

        if "task" in self._enabled_tools:
            task_name = (self._task_schema.get("name") if self._task_schema else None) or "task"
            task_description = (
                self._task_schema.get("description") if self._task_schema else None
            ) or "Delegate a background task to another agent."
            tool = Tool(
                self._task,
                name=task_name,
                description=task_description,
                metadata={"category": "switch_mode"},
            )
            tools.append(apply_params_schema(tool, self._task_schema))

        if "background_output" in self._enabled_tools:
            output_name = (
                self._background_output_schema.get("name")
                if self._background_output_schema
                else None
            ) or "background_output"
            output_description = (
                self._background_output_schema.get("description")
                if self._background_output_schema
                else None
            ) or "Get output from a background task."
            tool = Tool(
                self._background_output,
                name=output_name,
                description=output_description,
                metadata={"category": "other"},
            )
            tools.append(apply_params_schema(tool, self._background_output_schema))

        if "background_cancel" in self._enabled_tools:
            cancel_name = (
                self._background_cancel_schema.get("name")
                if self._background_cancel_schema
                else None
            ) or "background_cancel"
            cancel_description = (
                self._background_cancel_schema.get("description")
                if self._background_cancel_schema
                else None
            ) or "Cancel a background task."
            tool = Tool(
                self._background_cancel,
                name=cancel_name,
                description=cancel_description,
                metadata={"category": "other"},
            )
            tools.append(apply_params_schema(tool, self._background_cancel_schema))

        if "steer_task" in self._enabled_tools:
            steer_name = (
                self._steer_task_schema.get("name") if self._steer_task_schema else None
            ) or "steer_task"
            steer_description = (
                self._steer_task_schema.get("description") if self._steer_task_schema else None
            ) or (
                "Send a steering message to a running background task. "
                "The message is injected into the task's active turn (interrupt) "
                "or queued for its next turn (advisory), allowing you to redirect "
                "its research or provide new context without cancelling it."
            )
            tool = Tool(
                self._steer_task,
                name=steer_name,
                description=steer_description,
                metadata={"category": "other"},
            )
            tools.append(apply_params_schema(tool, self._steer_task_schema))

        if not tools:
            return None
        return FunctionToolset(tools)

    def get_ordering(self) -> CapabilityOrdering | None:
        """Declare middleware chain position."""
        return CapabilityOrdering(wrapped_by=[ProcessHistory, NativeTool])

    # ---- Run lifecycle ----

    async def before_run(self, ctx: RunContext[AgentContext]) -> None:
        """Initialize per-run resources.

        Starts the ``NotificationBatcher`` for this session (must be called
        in an async context).  EventBus subscription is per-task, not per-run;
        subscriptions are established in ``_task_async()`` and cleaned up in
        ``_run_and_stream``'s finally block.
        """
        state = self._get_session_state(ctx)
        state.pending_retrievals.clear()
        state.retrieval_retry_count = 0
        await state.batcher.start()

    async def after_run(
        self,
        ctx: RunContext[AgentContext],
        *,
        result: AgentRunResult[AgentContext],
    ) -> AgentRunResult[AgentContext]:
        """Clean up per-run session state after the agent run ends.

        Evicts the ``_session_states`` entry keyed by ``run_id`` and the
        corresponding ``id(ctx.deps)`` from ``_ephemeral_states``, then
        tears down the state's batcher and task manager so their timers
        and handles do not leak across runs.

        Args:
            ctx: The pydantic-ai run context.
            result: The agent run result (passed through unchanged).

        Returns:
            The unchanged ``result``.
        """
        agent_ctx = ctx.deps
        run_ctx = agent_ctx.run_ctx
        if run_ctx is not None:
            state = self._session_states.pop(run_ctx.run_id, None)
            # Drop any ephemeral fallback state keyed by this context object
            # so an id() reuse cannot serve a stale state.
            self._ephemeral_states.pop(id(agent_ctx), None)
        else:
            state = self._ephemeral_states.pop(id(agent_ctx), None)

        if state is not None:
            with contextlib.suppress(Exception):
                await state.batcher.shutdown()
            with contextlib.suppress(Exception):
                await state.task_manager.shutdown()

        return result

    async def after_node_run(
        self,
        ctx: RunContext[AgentContext],
        *,
        node: AgentNode[AgentContext],
        result: NodeResult[AgentContext],
    ) -> NodeResult[AgentContext]:
        """Intercept ``End`` when there are unretrieved background tasks.

        When ``force_retrieval`` is not ``disabled`` and the agent graph reaches
        ``End`` while ``_pending_retrievals`` is non-empty, redirect to a new
        ``ModelRequestNode`` with a prompt instructing the agent to retrieve
        results via ``background_output``.

        A loop breaker caps the number of injection attempts at
        ``_max_retrieval_retries`` to prevent infinite loops when the model
        ignores the directive or the API does not honor ``tool_choice``.
        """
        if self._force_retrieval is ForceRetrievalMode.disabled:
            return result
        if not isinstance(result, End):
            return result

        state = self._get_session_state(ctx)
        if not state.pending_retrievals:
            return result

        # Loop breaker: stop injecting after max retries
        if state.retrieval_retry_count >= self._max_retrieval_retries:
            logger.warning(
                "Force retrieval loop breaker triggered: %d attempts reached, %d pending "
                "task(s) will remain unretrieved: %s",
                state.retrieval_retry_count,
                len(state.pending_retrievals),
                ", ".join(sorted(state.pending_retrievals)),
            )
            return result

        state.retrieval_retry_count += 1

        task_ids = ", ".join(sorted(state.pending_retrievals))
        prompt = (
            f"<system-reminder>\n"
            f"[PENDING BACKGROUND TASKS]\n"
            f"You have {len(state.pending_retrievals)} background task(s) "
            f"that have not been retrieved yet.\n"
            f"Task IDs: {task_ids}\n\n"
            f"Use `background_output` with each task_id to retrieve the results "
            f"before finishing.\n"
            f"</system-reminder>"
        )

        # Deferred import: ModelRequestNode lives in pydantic_ai._agent_graph,
        # a private (underscore-prefixed) module. We guard the import with a
        # try/except so a minor pydantic-ai upgrade that moves or renames the
        # module doesn't silently break — the error is logged and re-raised
        # with a helpful message pointing at the version pin.
        try:
            from pydantic_ai._agent_graph import ModelRequestNode
        except ImportError as e:
            msg = (
                "Failed to import ModelRequestNode from pydantic_ai._agent_graph. "
                "This private API may have moved in a newer pydantic-ai version. "
                "Check the pydantic-ai-slim pin in pyproject.toml (>=2.12.0)."
            )
            raise ImportError(msg) from e

        return ModelRequestNode[AgentContext, Any](
            request=ModelRequest(parts=[UserPromptPart(content=prompt)]),
        )

    def get_model_settings(self) -> AgentModelSettings[AgentContext] | None:
        """Return per-step model settings that force ``tool_choice`` to ``background_output``.

        Only active when ``force_retrieval`` is ``tool_choice`` mode. In
        ``directive`` mode, no ``tool_choice`` forcing is applied — the agent
        is guided by the system-reminder prompt injected in ``after_node_run``.
        """
        if self._force_retrieval is not ForceRetrievalMode.tool_choice:
            return None

        def _settings(ctx: RunContext[AgentContext]) -> ModelSettings:
            if self._get_session_state(ctx).pending_retrievals:
                return ModelSettings(tool_choice=[self._output_tool_name])
            return ModelSettings()

        return _settings

    # ---- Tool methods (moved from BackgroundTaskProvider) ----

    @logfire.instrument("background_task.capability.task")
    async def _task(
        self,
        ctx: RunContext[AgentContext],
        agent: str,
        message: str,
        expected_output: str = "",
        load_skills: list[str] | None = None,
        title: str | None = None,
        async_mode: bool = False,
    ) -> str:
        """Launch a task, either synchronously or in the background.

        In synchronous mode (default), the task runs with streaming progress
        events and returns the result when complete.

        In async mode, the task starts in the background and returns
        immediately with a formatted text containing task_id, session_id,
        and status.

        Args:
            ctx: Run context with AgentContext as deps
            agent: The agent to execute the task
            message: The task instructions for the agent
            expected_output: Description of the expected output
            load_skills: Optional list of skill names to load for the subagent
            title: Optional title for the subtask
            async_mode: When true, run in background and return task_id immediately

        Returns:
            The task result (synchronous) or formatted text with task metadata (background)
        """
        agent_ctx = ctx.deps

        # Validate pool availability
        if agent_ctx.pool is None:
            msg = "No agent pool available"
            raise ToolError(msg)
        mode = agent
        if mode not in agent_ctx.pool.manifest.agents:
            available = ", ".join(agent_ctx.pool.manifest.agents.keys())
            return f"Error: Agent '{mode}' not found. Available: {available}"

        assert agent_ctx.pool.session_pool is not None, "SessionPool required"
        source_type: Literal["agent", "team_parallel", "team_sequential"] = "agent"
        config = agent_ctx.pool.manifest.agents.get(mode)
        if config is not None:
            config_type = str(config.type)  # pyright: ignore[reportAny]
            if config_type == "team":
                source_type = "team_parallel"

        # Handle delegation depth
        current_depth = 0
        if isinstance(agent_ctx.data, dict):
            current_depth = int(agent_ctx.data.get("delegation_depth", 0))

        if current_depth >= MAX_DELEGATION_DEPTH:
            return f"Error: Max delegation depth ({MAX_DELEGATION_DEPTH}) reached."

        # Prepare dependencies with incremented depth
        new_deps: DelegationDeps = {"delegation_depth": current_depth + 1}
        if isinstance(agent_ctx.data, dict):
            # Spread merge: result may contain arbitrary keys from ctx.data
            # beyond the DelegationDeps known fields. TypedDict captures the
            # known structure; the spread result is still a valid DelegationDeps.
            new_deps = {**agent_ctx.data, **new_deps}  # pyright: ignore[reportAssignmentType]

        # Fetch and format skill instructions
        skills_content = ""
        if load_skills:
            skills_content = await self._format_skills_instructions(agent_ctx, load_skills, mode)

        # Format prompt with task and expected_output XML sections
        formatted_prompt = f"<task>\n\n{message}\n</task>\n\n<expected_output>\n\n{expected_output}\n\n</expected_output>"  # noqa: E501
        if skills_content:
            formatted_prompt = f"{skills_content}\n\n{formatted_prompt}"

        # Create child session via AgentContext (uses SessionManager when available)
        parent_session_id = (
            agent_ctx.run_ctx.session_id
            if agent_ctx.run_ctx
            else (
                (agent_ctx.node.session_id or "") if isinstance(agent_ctx.node, BaseAgent) else ""
            )
        )
        try:
            child_session_id = await agent_ctx.create_child_session(
                agent_name=mode,
                agent_type="native",
                parent_session_id=parent_session_id,
                spawn_mechanism="task",
                description=title or "",
                tool_call_id=ctx.tool_call_id,
            )
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to create child session: {type(exc).__name__}: {exc}"
        child_depth = current_depth + 1
        tool_call_id = ctx.tool_call_id

        if async_mode:
            return await self._task_async(
                ctx=agent_ctx,
                mode=mode,
                source_type=source_type,
                formatted_prompt=formatted_prompt,
                new_deps=new_deps,
                child_session_id=child_session_id,
                parent_session_id=parent_session_id,
                child_depth=child_depth,
                tool_call_id=tool_call_id,
                title=title,
                load_skills=load_skills or [],
            )

        # Synchronous mode - stream with SubAgentEvent wrapping
        return await self._task_sync(
            ctx=agent_ctx,
            mode=mode,
            source_type=source_type,
            formatted_prompt=formatted_prompt,
            new_deps=new_deps,
            child_session_id=child_session_id,
            parent_session_id=parent_session_id,
            child_depth=child_depth,
            tool_call_id=tool_call_id,
        )

    @logfire.instrument("background_task.capability.task_sync")
    async def _task_sync(
        self,
        ctx: AgentContext,
        mode: str,
        source_type: Literal["agent", "team_parallel", "team_sequential"],
        formatted_prompt: str,
        new_deps: DelegationDeps,
        child_session_id: str,
        parent_session_id: str,
        child_depth: int,
        tool_call_id: str | None,
    ) -> str:
        """Execute a task synchronously, returning the final result.

        Events are broadcast via the EventBus by TurnRunner; this method
        only drains the stream to extract the final result text.

        Args:
            ctx: Agent context
            node: The agent/team node to execute (unused, kept for compat)
            mode: Name of the agent/team
            source_type: Type of source for event metadata
            formatted_prompt: The formatted prompt with skills and expected_output
            new_deps: Dependencies for the child agent
            child_session_id: ID of the child session
            parent_session_id: ID of the parent session
            child_depth: Nesting depth for the child
            tool_call_id: ID of the tool call that triggered this task

        Returns:
            The final result text from the delegated task
        """
        final_result = ""
        content_parts: list[str] = []

        assert ctx.pool is not None, "Agent pool is required"
        session_pool = ctx.pool.session_pool
        assert session_pool is not None, "SessionPool is required for task delegation"

        try:
            input_provider = ctx.get_input_provider()
        except RuntimeError:
            input_provider = None

        stream = session_pool.run_stream(
            child_session_id,
            formatted_prompt,
            input_provider=input_provider,
            deps=new_deps,
        )

        try:
            async for event in stream:
                # Yield control to allow event queue processing
                await asyncio.sleep(0)

                match event:
                    case ToolCallStartEvent(tool_name="attempt_completion", raw_input=args):
                        final_result = str(args.get("result", ""))
                        break

                    case ToolCallCompleteEvent(
                        tool_name="attempt_completion", tool_result=completion_result
                    ):
                        final_result = str(completion_result) if completion_result else ""
                        break

                    case StreamCompleteEvent(message=final_message):
                        if final_message and final_message.content:
                            final_content = str(final_message.content)
                            if final_content.strip():
                                final_result = final_content
                            elif content_parts:
                                # Preserve accumulated content when the final
                                # message is empty/whitespace-only (e.g. model
                                # stream was killed mid-generation)
                                final_result = "".join(content_parts)
                        elif content_parts:
                            final_result = "".join(content_parts)
                        break

                    case RunErrorEvent(message=error_msg):
                        final_result = f"Error: {error_msg}"
                        break

                    case RunFailedEvent(exception=exc):
                        final_result = f"Task failed: {exc}"
                        break

                    case _:
                        # Accumulate text deltas for fallback content
                        if isinstance(event, PartDeltaEvent) and event.delta:
                            delta = event.delta
                            if (
                                isinstance(delta, (TextPartDelta, ThinkingPartDelta))
                                and delta.content_delta
                            ):
                                content_parts.append(delta.content_delta)
        except Exception as exc:  # noqa: BLE001
            final_result = f"Task failed: {exc}"

        return final_result if final_result else "Error: No result produced"

    @logfire.instrument("background_task.capability.task_async")
    async def _task_async(
        self,
        ctx: AgentContext,
        mode: str,
        source_type: Literal["agent", "team_parallel", "team_sequential"],
        formatted_prompt: str,
        new_deps: DelegationDeps,
        child_session_id: str,
        parent_session_id: str,
        child_depth: int,
        tool_call_id: str | None,
        title: str | None = None,
        load_skills: list[str] | None = None,
    ) -> str:
        """Execute a task asynchronously in the background.

        Starts the task via ``BackgroundTaskManager``, streaming output to
        the internal filesystem. Returns immediately with a formatted text
        containing the task_id, session_id, and status.

        Args:
            ctx: Agent context
            node: The agent/team node to execute
            mode: Name of the agent/team
            source_type: Type of source for event metadata
            formatted_prompt: The formatted prompt with skills and expected_output
            new_deps: Dependencies for the child agent
            child_session_id: ID of the child session
            parent_session_id: ID of the parent session
            child_depth: Nesting depth for the child
            tool_call_id: ID of the tool call that triggered this task
            title: Optional title for the subtask
            load_skills: Optional list of skill names to load for the subagent

        Returns:
            Formatted text with task_id, session_id, description, and status
        """
        description = title or mode
        task_id = _generate_task_id(description[:30])
        output_path = f"/tasks/{task_id}/output.md"

        state = self._get_session_state(ctx)

        # Track for force_retrieval: this task must be retrieved before the run ends
        if self._force_retrieval is not ForceRetrievalMode.disabled:
            state.pending_retrievals.add(task_id)

        # Create the task directory on internal filesystem
        fs = ctx.internal_fs
        try:
            fs.mkdirs(f"/tasks/{task_id}", exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to create task directory: {type(exc).__name__}: {exc}"

        # Register the task model in the manager
        task_model = BackgroundTask(
            id=task_id,
            description=description,
            agent_or_team=mode,
            prompt=formatted_prompt,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            load_skills=load_skills or [],
            output_file=output_path,
        )
        state.task_manager.register_task(task_model)

        # Determine whether SessionPool is available for this background task.
        assert ctx.pool is not None, "Agent pool is required"
        session_pool = ctx.pool.session_pool
        assert session_pool is not None, "SessionPool is required for background tasks"

        # Create the async coroutine that streams to filesystem.
        async def _run_and_stream() -> None:
            """Stream task output to filesystem.

            Events are published to the EventBus by TurnRunner; this coroutine
            only monitors the bus to write incremental output and terminal
            results to the internal filesystem.
            """
            with logfire.span(
                "background_task.capability.run_and_stream",
                task_id=task_id,
                child_session_id=child_session_id,
            ):
                await _run_and_stream_inner()

        async def _run_and_stream_inner() -> None:
            content_parts: list[str] = []

            assert ctx.pool is not None
            session_pool = ctx.pool.session_pool
            assert session_pool is not None

            try:
                # Subscribe to EventBus BEFORE starting the run to avoid
                # missing any events.
                event_queue = await session_pool.event_bus.subscribe(
                    child_session_id,
                    scope="session",
                )

                try:
                    input_provider = ctx.get_input_provider()
                except RuntimeError:
                    input_provider = None

                # Start the run through SessionPool (fire-and-forget).
                # send_message replaced the deprecated receive_request method.
                # It returns str (message_id) on success, None on failure.
                message_id = await session_pool.send_message(
                    child_session_id,
                    formatted_prompt,
                    input_provider=input_provider,
                    deps=new_deps,
                )

                if message_id is None:
                    logger.warning(
                        "SessionPool send_message returned None for child_session_id=%s — run may "
                        "be queued",
                        child_session_id,
                    )
                    fs.pipe(
                        output_path,
                        b"# Task Failed\n\nRun was not started (send_message returned None)",
                    )
                    return

                task_error: str | None = None

                try:
                    while True:
                        try:
                            envelope = await event_queue.get()
                        except asyncio.QueueShutDown:
                            fs.pipe(
                                output_path,
                                b"# Task Failed\n\nEvent stream ended without a terminal event",
                            )
                            break

                        # Unwrap the actual event from EventEnvelope
                        event = envelope.event if isinstance(envelope, EventEnvelope) else envelope

                        # Write to filesystem only
                        match event:
                            case ToolCallStartEvent(
                                tool_name="attempt_completion",
                                raw_input=args,
                            ):
                                result_text = str(args.get("result", ""))
                                if result_text:
                                    fs.pipe(output_path, result_text.encode())
                                break
                            case ToolCallCompleteEvent(
                                tool_name="attempt_completion",
                                tool_result=completion_result,
                            ):
                                result_text = str(completion_result) if completion_result else ""
                                if result_text:
                                    fs.pipe(output_path, result_text.encode())
                                break
                            case StreamCompleteEvent(message=final_message):
                                final_content = ""
                                if final_message and final_message.content:
                                    final_content = str(final_message.content)
                                if final_content and final_content.strip():
                                    fs.pipe(output_path, final_content.encode())
                                elif content_parts:
                                    # Preserve accumulated content from PartDeltaEvent
                                    # when the final message is empty/whitespace-only
                                    # (e.g. model stream was killed mid-generation)
                                    fs.pipe(output_path, "".join(content_parts).encode())
                                break
                            case RunErrorEvent(message=error_msg):
                                fs.pipe(
                                    output_path,
                                    f"# Task Error\n\n{error_msg}".encode(),
                                )
                                task_error = error_msg
                                break
                            case RunFailedEvent(exception=exc):
                                fs.pipe(
                                    output_path,
                                    f"# Task Failed\n\n{exc}".encode(),
                                )
                                task_error = str(exc)
                                break
                            case _:
                                # Accumulate text deltas for incremental filesystem writing
                                if isinstance(event, PartDeltaEvent) and event.delta:
                                    delta = event.delta
                                    if (
                                        isinstance(delta, (TextPartDelta, ThinkingPartDelta))
                                        and delta.content_delta
                                    ):
                                        content_parts.append(delta.content_delta)
                                        fs.pipe(
                                            output_path,
                                            "".join(content_parts).encode(),
                                        )
                finally:
                    try:
                        await session_pool.event_bus.unsubscribe(
                            child_session_id,
                            event_queue,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to unsubscribe from EventBus for session %s",
                            child_session_id,
                            exc_info=True,
                        )

                if task_error is not None:
                    raise RuntimeError(task_error)
            except asyncio.CancelledError:
                # Distinguish timeout from explicit cancellation: the manager
                # marks the task model ``timed_out`` *before* cancelling the
                # coroutine (see ``BackgroundTaskManager._run_with_timeout``),
                # so we can inspect the status here to pick the right message.
                task = state.task_manager.get_task(task_id)
                if task is not None and task.status == "timed_out":
                    timeout_msg = f"Task {task_id} ({mode}) timed out"
                    fs.pipe(output_path, f"# Task Timed Out\n\n{timeout_msg}".encode())
                else:
                    cancel_msg = f"Task {task_id} ({mode}) was cancelled"
                    fs.pipe(output_path, f"# Task Cancelled\n\n{cancel_msg}".encode())
                raise
            except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as e:
                error_msg = f"Task {task_id} ({mode}) failed: {type(e).__name__}: {e}"
                error_content = f"# Task Failed\n\n{error_msg}"
                fs.pipe(output_path, error_content.encode())
                raise
            except Exception:
                error_msg = f"Task {task_id} ({mode}) failed with an error"
                error_content = f"# Task Failed\n\n{error_msg}"
                fs.pipe(output_path, error_content.encode())
                raise

        # Build the completion callback before starting the task so it is
        # installed atomically — a fast task could complete before we reach
        # the handle-assignment line if we set it after start_task().
        def _on_task_completed() -> None:
            """Called by BackgroundTaskManager when the task reaches a terminal state.

            Does TWO things:
            1. ALWAYS pop+set child_done_events IMMEDIATELY (unconditional) —
               this unblocks any parent waiting on ``child_done_events.wait()``
               and prevents permanent hangs.
            2. THEN conditionally submit to the NotificationBatcher for batched
               notification delivery — ONLY if no blocking waiter is registered
               (i.e., ``background_output(block=True)`` is NOT actively waiting).
            """
            run_ctx = ctx.run_ctx

            # 1. ALWAYS pop+set child_done_events immediately (unconditional)
            if run_ctx is not None and child_session_id:
                event = run_ctx.child_done_events.pop(child_session_id, None)
                if event is not None:
                    event.set()

            # 2. Conditionally submit to batcher for notification (skip if blocking waiter)
            if state.task_manager.has_blocking_waiter(task_id):
                return

            task = state.task_manager.get_task(task_id)
            if task is None:
                return

            if not parent_session_id:
                logger.warning(
                    "Background task %s completed but has no parent_session_id — skipping "
                    "notification",
                    task_id,
                )
                return

            # Submit to batcher — it will format, debounce, and deliver via
            # the deliver_callback (which calls followup() + child_done_events pop).
            try:
                state.batcher.submit(task)
            except ValueError:
                logger.warning(
                    "Batcher rejected task %s (missing parent_session_id)",
                    task_id,
                )

        # Start the task via BackgroundTaskManager with the callback
        state.task_manager.start_task(task_id, _run_and_stream(), on_completed=_on_task_completed)

        return (
            f"Background task launched.\n\n"
            f"Task ID: {task_id}\n"
            f"Session ID: {child_session_id}\n"
            f"Description: {description}\n"
            f"Agent: {mode}\n"
            f"Status: running\n\n"
            f"Continue your current workflow. You will receive a <system-reminder> notification "
            f'when the task completes. Use `background_output(task_id="{task_id}")` to retrieve '
            f"the result when ready."
        )

    @logfire.instrument("background_task.capability.background_output")
    async def _background_output(
        self,
        ctx: RunContext[AgentContext],
        task_id: str,
        block: bool = False,
        timeout_seconds: float = 60.0,
    ) -> str:
        """Get output from a background task.

        For running/pending tasks, returns status-only info unless ``block=True``
        is specified, in which case the call waits for the task to reach a
        terminal state.
        You must call this tool at least once if you lanuch a task with `async_mode=true`.
        Never call this tool If you haven't lanuch a sub-task

        Args:
            ctx: Run context with AgentContext as deps
            task_id: The ID of the background task
            block: Whether to wait for the task to complete
            timeout_seconds: Maximum seconds to wait when block=True. Defaults to 60.

        Returns:
            The task output or current status
        """
        agent_ctx = ctx.deps
        state = self._get_session_state(ctx)

        # Mark this task as retrieved (no-op if force_retrieval is disabled)
        state.pending_retrievals.discard(task_id)

        task_model = state.task_manager.get_task(task_id)
        if task_model is None:
            return f"Task {task_id!r} not found"

        # If task is in a terminal state, return result/error/status
        if task_model.status in TERMINAL_STATES:
            return self._format_terminal_task_output(agent_ctx, task_model)

        # Running/pending task: status-only unless blocking
        if not block:
            started_info = (
                f"Started at: {task_model.started_at}."
                if task_model.started_at
                else "Not yet started."
            )
            return (
                f"Task {task_id!r} is {task_model.status}.\n{started_info}\n"
                f"Use block=True to wait for completion."
            )

        # Block until terminal state (or wait timeout).
        # Register as a blocking waiter so the completion callback knows
        # not to inject a duplicate prompt.
        waiter_token = state.task_manager.register_blocking_waiter(task_id)
        try:
            task_model = await state.task_manager.wait_for_task(
                task_id, timeout_seconds=timeout_seconds
            )
        finally:
            if waiter_token is not None:
                state.task_manager.unregister_blocking_waiter(task_id, waiter_token)
        if task_model is None:
            return f"Task {task_id!r} not found"

        # If the wait timed out and the task is still non-terminal, return
        # a timeout/status response — the task continues running and is NOT
        # cancelled.  Only `background_cancel` cancels tasks.
        if task_model.status not in TERMINAL_STATES:
            started_info = (
                f"Started at: {task_model.started_at}."
                if task_model.started_at
                else "Not yet started."
            )
            return (
                f"Task {task_id!r} is still {task_model.status} after waiting {timeout_seconds}s "
                f"(wait timed out; task continues running).\n"
                f"{started_info}\n"
                f"Call background_output again later to check progress, or use "
                f"background_cancel to cancel."
            )

        return self._format_terminal_task_output(agent_ctx, task_model)

    def _format_terminal_task_output(self, ctx: AgentContext, task_model: BackgroundTask) -> str:
        """Format the output string for a task in a terminal state.

        For completed tasks where ``task_model.result`` is None but an
        ``output_file`` exists, reads the content from ``ctx.internal_fs``.

        Args:
            ctx: Agent context (used to access internal_fs)
            task_model: The task model with a terminal status

        Returns:
            Formatted output string in oh-my-openagent style
        """
        duration = _format_duration(task_model.started_at, task_model.completed_at)

        if task_model.status == "completed":
            result_text = task_model.result
            if result_text is None and task_model.output_file:
                try:
                    raw = ctx.internal_fs.cat(task_model.output_file)
                    content = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    result_text = content if content else "No result available."
                except Exception:  # noqa: BLE001
                    result_text = "No result available."
            elif result_text is None:
                result_text = "No result available."

            duration_line = f"\nDuration: {duration}" if duration else ""
            return (
                f"Task Result\n\nTask ID: {task_model.id}\n"
                f"Description: {task_model.description}{duration_line}\n"
                f"Session ID: {task_model.child_session_id}\n\n---\n\n{result_text}"
            )

        if task_model.status in ("error", "timed_out"):
            error_text = task_model.error or "No error details available."
            duration_line = f"\nDuration: {duration}" if duration else ""
            return (
                f"Task Error\n\n"
                f"Task ID: {task_model.id}\n"
                f"Description: {task_model.description}"
                f"{duration_line}\n"
                f"Session ID: {task_model.child_session_id}\n\n"
                f"---\n\nError: {error_text}"
            )

        # cancelled
        return (
            f"Task Cancelled\n\nTask ID: {task_model.id}\n"
            f"Description: {task_model.description}\n"
            f"Session ID: {task_model.child_session_id}"
        )

    @logfire.instrument("background_task.capability.background_cancel")
    async def _background_cancel(
        self,
        ctx: RunContext[AgentContext],
        task_id: str | None = None,
        cancel_all: bool = False,
    ) -> str:
        """Cancel a background task or all background tasks.

        Args:
            ctx: Run context with AgentContext as deps
            task_id: The ID of the task to cancel (not required when cancel_all is true)
            cancel_all: When true, cancel all running background tasks

        Returns:
            A message describing what was cancelled

        Raises:
            ToolError: If neither ``task_id`` nor ``cancel_all=True`` is provided,
                or if both are provided simultaneously.
        """
        if task_id is None and not cancel_all:
            msg = "Either task_id or cancel_all=True must be provided"
            raise ToolError(msg)

        if task_id is not None and cancel_all:
            msg = "Cannot specify both task_id and cancel_all=True"
            raise ToolError(msg)

        state = self._get_session_state(ctx)

        if cancel_all:
            # Cancel all non-terminal tasks and build a markdown table
            cancelled_tasks: list[BackgroundTask] = []
            for task_model in state.task_manager.get_all_tasks():
                if task_model.status not in TERMINAL_STATES:
                    await state.task_manager.cancel_task(task_model.id)
                    cancelled_tasks.append(task_model)

            count = len(cancelled_tasks)
            if count == 0:
                return "No running tasks to cancel."

            lines = [f"Cancelled {count} background task(s):\n"]
            lines.append("| Task ID | Description | Status |")
            lines.append("|---------|-------------|--------|")
            lines.extend(f"| `{t.id}` | {t.description} | cancelled |" for t in cancelled_tasks)
            return "\n".join(lines)

        # task_id is guaranteed non-None by the validation above
        if task_id is None:
            msg = "task_id must be provided when cancel_all is False"
            raise ToolError(msg)

        # Get task description before cancelling (for formatted output)
        task_before = state.task_manager.get_task(task_id)
        description = task_before.description if task_before else task_id

        cancel_result = await state.task_manager.cancel_task(task_id)

        # Check task status after cancellation
        task_after = state.task_manager.get_task(task_id)
        if task_after is not None and task_after.status == "cancelled":
            return (
                f"Task cancelled successfully\n\nTask ID: {task_id}\n"
                f"Description: {description}\nStatus: cancelled"
            )
        if task_after is not None and task_after.status in TERMINAL_STATES:
            return (
                f"Task was already {task_after.status}\n\nTask ID: {task_id}\n"
                f"Description: {description}\nStatus: {task_after.status}"
            )
        # If task not found or not cancelled, return the raw result
        return cancel_result

    @logfire.instrument("background_task.capability.steer_task")
    async def _steer_task(
        self,
        ctx: RunContext[AgentContext],
        task_id: str,
        message: str,
        mode: Literal["interrupt", "advisory"] = "advisory",
    ) -> str:
        """Send a steering message to a running background task.

        Injects a message into the target task's active session, allowing
        the parent agent to redirect the subagent's research or provide
        new context mid-execution without cancelling the task.

        Two modes:

        - ``advisory`` (default): The message is queued for the next turn.
          Non-interrupting — the subagent finishes its current turn before
          seeing the message. Use this when the new context is valuable but
          not urgently redirecting the subagent's current work.
        - ``interrupt``: The message is injected into the active turn
          immediately (mid-turn). Use this sparingly — only when the
          subagent's current direction is wrong and must be corrected
          before it produces more output.

        Args:
            ctx: Run context with AgentContext as deps
            task_id: The ID of the background task to steer
            message: The steering message to inject into the subagent
            mode: ``"advisory"`` (next-turn, default) or ``"interrupt"``
                (mid-turn). Prefer advisory unless the subagent's current
                direction must be immediately corrected.

        Returns:
            Confirmation message indicating whether steering succeeded

        Raises:
            ToolError: If the task is not found, already in a terminal
                state, or the session pool is unavailable.
        """
        agent_ctx = ctx.deps
        state = self._get_session_state(ctx)

        task_model = state.task_manager.get_task(task_id)
        if task_model is None:
            msg = f"Task {task_id!r} not found"
            raise ToolError(msg)

        if task_model.status in TERMINAL_STATES:
            return (
                f"Task {task_id!r} is already {task_model.status} — cannot steer a completed task."
            )

        child_session_id = task_model.child_session_id
        if child_session_id is None:
            msg = f"Task {task_id!r} has no child session — cannot steer"
            raise ToolError(msg)

        session_pool = agent_ctx.pool.session_pool if agent_ctx.pool is not None else None
        if session_pool is None:
            msg = "SessionPool is not available — cannot steer background task"
            raise ToolError(msg)

        if mode == "interrupt":
            wrapped = (
                f"<system-reminder>\n"
                f"[STEERING DIRECTIVE — INTERRUPT]\n"
                f"This is a directive from the parent diagnostic agent, not a new user query.\n"
                f"Adjust your current research direction immediately based on this "
                f"new information.\n"
                f"Do NOT restart from scratch — integrate this into your existing work.\n"
                f"Skip any research branches this directive rules out.\n"
                f"---\n"
                f"{message}\n"
                f"</system-reminder>"
            )
            success = await session_pool.steer(child_session_id, wrapped)
        else:
            wrapped = (
                f"<system-reminder>\n"
                f"[STEERING DIRECTIVE — ADVISORY]\n"
                f"This is a directive from the parent diagnostic agent, not a new user query.\n"
                f"Integrate this new information into your research and adjust your "
                f"plan accordingly.\n"
                f"Do NOT restart from scratch — refine your existing research direction.\n"
                f"Skip any research branches this directive rules out.\n"
                f"---\n"
                f"{message}\n"
                f"</system-reminder>"
            )
            success = await session_pool.followup(child_session_id, wrapped)

        if not success:
            return (
                f"Steering message could not be delivered to task {task_id!r} "
                f"(session {child_session_id} may have ended). "
                f"Mode: {mode}. Message: {message[:200]}"
            )

        return (
            f"Steering message delivered to task {task_id!r} "
            f"(session {child_session_id}).\n"
            f"Mode: {mode} ({'mid-turn interrupt' if mode == 'interrupt' else 'next-turn queue'})\n"
            f"Message: {message[:500]}"
        )

    async def _format_skills_instructions(
        self,
        ctx: AgentContext,
        skill_names: list[str],
        target_agent: str,
    ) -> str:
        """Format skills as XML instructions for subagent context.

        Delegates to ``load_skill_for_node()`` for unified resolution (skill_resolver,
        local fallback, not-found error string).  Both bare skill names and
        ``skill://`` URIs are supported — including reference paths such as
        ``skill://my-skill/references/guide.md`` which load only the
        referenced file instead of the full ``SKILL.md``.

        Uses ``load_skill_for_node`` with the target agent's node name so that
        skill scope visibility is checked against the subagent, not the parent.

        Note: ``arguments`` substitution (``$1``, ``$@``, ``$ARGUMENTS``) is
        intentionally **not** triggered here.  The task ``message`` is already
        injected as a ``<task>`` block in the prompt; passing it again as
        skill arguments would duplicate the content.

        Args:
            ctx: Agent context for skill loading
            skill_names: List of skill names or ``skill://`` URIs to format
            target_agent: Name of the target subagent for scope checking

        Returns:
            XML-formatted skill instructions string
        """
        skill_sections: list[str] = []
        for skill_name in skill_names:
            # Extract a clean display name for the XML attribute.
            # For skill:// URIs this yields "skill-name" or
            # "skill-name/references/file.md"; for bare names it returns
            # the name as-is.
            try:
                resolved = ResolvedSkillURI.parse(skill_name)
                display_name = resolved.skill_name
                if resolved.reference_path:
                    display_name = f"{resolved.skill_name}/{resolved.reference_path}"
            except Exception:  # noqa: BLE001
                display_name = skill_name

            try:
                instructions = await load_skill_for_node(ctx, skill_name, node_name=target_agent)
            except Exception as exc:  # noqa: BLE001
                instructions = (
                    f"Error: Failed to load skill '{skill_name}': {type(exc).__name__}: {exc}"
                )
            skill_sections.append(
                f'<skill-instruction name="{display_name}">\n'
                f"{instructions}\n</skill-instruction>\n",
            )
        return "\n".join(skill_sections)
