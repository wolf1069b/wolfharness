"""TeamCommCapability — capability for dynamic team communication.

This capability provides the protocol instructions and team communication
tools (send_message, task_create, read_blackboard, etc.) to agents that
are members of or leads of a dynamic team.

Universal tools (all members can use):
    - send_message: Send a message to a teammate's inbox.
    - task_create: Create a task on the shared task board (lead-only for
      top-level; any member for subtasks with parent_id).
    - task_list: List tasks on the shared task board.
    - task_update: Update a task's status or owner (lead: any task;
      member: own tasks or unclaimed only).
    - task_get: Get a single task by ID.
    - read_blackboard: Read a key from the shared blackboard.
    - write_blackboard: Write a key to the shared blackboard.
    - list_blackboard: List all keys on the shared blackboard.
    - team_status: Get the current status of the team.

Lead-only tools (only agents with ``team_role == "lead"``):
    - task_create (top-level only): Create a top-level task on the shared task board.
    - team_create: Create a new team with eligible members.
    - team_delete: Delete the current team and close all member sessions.
    - delete_blackboard: Delete a key from the shared blackboard.
    - shutdown_request: Shut down (remove) a specific team member.
    - team_add_member: Add a new member to an existing team.

    Lead-only tools are registered for all agents but filtered out for
    non-lead members by ``prepare_tools()`` before the model receives the
    tool list.  Runtime permission checks in each tool body remain as a
    safety net.

Per-session instantiation:
    The factory creates a shared instance with ``session_metadata=None``
    during ``_compile_agent_capabilities()``. When a session with a
    ``team_id`` in its metadata is created, ``create_session_agent()``
    replaces the shared instance with a per-session instance carrying
    the actual session metadata.

Role-aware tool schema:
    ``prepare_tools()`` modifies tool definitions based on
    ``team_role`` from session metadata:

    - **Non-lead members**: Lead-only tools are removed entirely.
      ``send_message`` has its ``to`` parameter description updated to
      omit the broadcast (``"*"``) mention, and a ``pattern`` constraint
      is added to reject ``"*"`` at the schema level.

    - **Lead agents**: All tool definitions are returned unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import tempfile
from typing import TYPE_CHECKING, Annotated, Any, override
import uuid

from pydantic.fields import Field
from pydantic_ai.messages import ToolReturn
from pydantic_ai.tools import RunContext  # noqa: TC002 - needed at runtime for get_type_hints()

from wolfharness.capabilities.function_toolset import FunctionToolsetCapability
from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import ToolDefinition

    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.file_team_state import FileTeamState
    from wolfharness.lifecycle.types import DeliveryMode
    from wolfharness.tools.base import Tool
    from wolfharness_config.team_mode import TeamModeConfig


logger = get_logger(__name__)

# Strong references to cleanup tasks so asyncio does not garbage-collect them
# while they are awaiting ``RunHandle.complete_event``.
_cleanup_tasks: set[asyncio.Task[Any]] = set()


class TeamCommCapability(FunctionToolsetCapability[Any]):
    """Capability providing team communication protocol instructions and tools.

    Inherits from :class:`FunctionToolsetCapability` and overrides
    ``get_instructions()`` and ``get_tools()`` to respect the
    :class:`TeamModeConfig` enabled flag and session metadata availability.

    Attributes:
        _config: The resolved team mode configuration.
        _agent_name: Name of the agent this capability is attached to.
        _session_metadata: Per-session metadata (team_name, team_role, etc.).
    """

    # Auto-cleanup tuning (override in tests).
    _idle_timeout: float = 300.0
    _poll_interval: float = 30.0

    def __init__(
        self,
        config: TeamModeConfig,
        agent_name: str,
        session_metadata: dict[str, Any] | None = None,
        agent_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Initialize the team communication capability.

        Args:
            config: The resolved team mode configuration (global + agent overlay).
            agent_name: Name of the agent this capability belongs to.
            session_metadata: Optional per-session metadata containing
                ``team_name``, ``team_role``, ``team_member_name``, etc.
                When ``None`` or empty, ``get_instructions()`` returns ``None``.
            agent_descriptions: Optional mapping of agent name to short
                description for eligible agents. Used in ``get_instructions()``
                so the LLM knows what each agent does.
        """
        super().__init__(name="team_comm")
        self._config = config
        self._agent_name = agent_name
        self._session_metadata: dict[str, Any] = session_metadata or {}
        self._agent_descriptions: dict[str, str] = agent_descriptions or {}
        # Per-instance lock serializes _create_member_session calls within
        # the same agent (concurrent PydanticAI tool calls share the same
        # capability instance) without blocking other teams' agents.
        self._create_session_lock = asyncio.Lock()
        # Register universal tools (all members can use)
        if config.enabled:
            self.register_tool(self.send_message)
            self.register_tool(self.task_create)
            self.register_tool(self.task_list)
            self.register_tool(self.task_update)
            self.register_tool(self.task_get)
            self.register_tool(self.read_blackboard)
            self.register_tool(self.write_blackboard)
            self.register_tool(self.list_blackboard)
            self.register_tool(self.team_status)
            # Register lead-only tools — filtered out for non-lead members
            # by prepare_tools() before the model receives the tool list.
            self.register_tool(self.team_create)
            self.register_tool(self.team_delete)
            self.register_tool(self.delete_blackboard)
            self.register_tool(self.shutdown_request)
            self.register_tool(self.team_add_member)
            self.register_tool(self.task_create_batch)

    @property
    def _notice_mode(self) -> DeliveryMode:
        """Resolve delivery mode from config."""
        from wolfharness.lifecycle.types import DeliveryMode

        return (
            DeliveryMode.STEER
            if self._config.notice_delivery_mode == "steer"
            else DeliveryMode.QUEUE
        )

    def _wrap_notice_content(self, body: str) -> str:
        """Return the message body for delivery.

        Team notices are always delivered as user messages.
        """
        return body

    async def _notify_member(
        self,
        agent_ctx: AgentContextDeps,
        team_id: str,
        member_name: str,
        msg_body: str,
    ) -> None:
        """Send a best-effort system notification to a team member.

        Used by task_create and task_update to push notifications when
        tasks are assigned or unblocked, so agents don't have to poll
        task_list to discover their work.

        Silently skips when the member has no session, when notifying
        self, or when session_pool is unavailable.
        """
        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return
        target_sid = team_state.get_member_session_id(team_id, member_name)
        if target_sid is None:
            return
        current_member: str = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )
        if member_name == current_member:
            return
        wrapped = (
            f'<team-message from="system" type="task_notification">\n\n'
            f"{msg_body}\n\n</team-message>"
        )
        try:
            await session_pool.send_message(
                target_sid,
                self._wrap_notice_content(wrapped),
                mode=self._notice_mode,
                source="accepted",
                meta={"from": "system", "team_id": team_id},
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to notify member '%s' for task notification",
                member_name,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_agent_context(self, ctx: RunContext[Any]) -> AgentContextDeps:
        """Extract AgentContextDeps from a pydantic-ai RunContext.

        Delegates to the shared ``resolve_agent_context_from_deps`` utility
        which handles both the production path (``RuntimeAgentContext.data``)
        and the test path (direct ``AgentContextDeps``).

        Args:
            ctx: The RunContext passed to a tool function.

        Returns:
            The AgentContextDeps from ``ctx.deps`` (or ``ctx.deps.data``).

        Raises:
            RuntimeError: If ``ctx.deps`` is None or AgentContextDeps is not found.
        """
        from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

        return resolve_agent_context_from_deps(ctx.deps, capability_name="TeamCommCapability")

    async def _format_member_skills_instructions(
        self,
        ctx: RunContext[Any],
        skill_names: list[str],
        member_agent: str,
    ) -> str:
        """Render skill instructions as XML blocks for a member's system prompt.

        Loads each skill via ``load_skill_for_node`` with the *member's* node
        name so package-scope visibility is checked against the member agent,
        not the lead. Supports bare skill names and ``skill://`` URIs
        (including reference paths). Deduplicates by display name, preserving
        first-occurrence order. Failures degrade to readable error text — they
        never raise (member creation must not abort).

        Skill resolution uses the runtime ``AgentContext`` carried by
        ``ctx.deps`` (the object exposing ``.pool``) — NOT the capability-layer
        ``AgentContextDeps`` (frozen dataclass without ``.pool``). See
        ``resolve_agent_context_from_deps`` for the relationship.

        Args:
            ctx: pydantic-ai run context whose ``deps`` is the runtime
                ``AgentContext`` (mocked in unit tests).
            skill_names: Bare skill names or ``skill://`` URIs to inject.
            member_agent: Registry name of the member agent; its node scope
                governs skill visibility.

        Returns:
            Newline-joined ``<skill-instruction name="...">...</skill-instruction>``
            blocks, or ``""`` when no skills are requested.
        """
        from wolfharness.skills.uri_resolver import ResolvedSkillURI
        from wolfharness_toolsets.builtin.skills import load_skill_for_node

        if not skill_names:
            return ""

        sections: list[str] = []
        # Dedupe preserving first-occurrence order (spec: duplicate skill
        # names SHALL be injected exactly once).
        unique_names = list(dict.fromkeys(skill_names))
        skill_ctx: Any = ctx.deps  # RuntimeAgentContext with .pool; mocked in unit tests
        for skill_name in unique_names:
            # Clean display name for the XML attribute.
            try:
                resolved = ResolvedSkillURI.parse(skill_name)
                display_name = resolved.skill_name
                if resolved.reference_path:
                    display_name = f"{resolved.skill_name}/{resolved.reference_path}"
            except Exception:  # noqa: BLE001
                display_name = skill_name

            try:
                instructions = await load_skill_for_node(
                    skill_ctx,
                    skill_name,
                    node_name=member_agent,
                    include_assembly=False,
                )
            except Exception as exc:  # noqa: BLE001
                instructions = (
                    f"Error: Failed to load skill '{skill_name}': {type(exc).__name__}: {exc}"
                )
            sections.append(
                f'<skill-instruction name="{display_name}">\n{instructions}\n</skill-instruction>'
            )
        return "\n".join(sections)

    @staticmethod
    def _coerce_skill_names(value: Any) -> list[str]:
        """Coerce a ``skills`` value to a list of names, dropping bad types.

        Accepts lists of strings (empty included). Any other shape (``None``,
        a bare string, a list containing non-strings) is treated as empty so
        a malformed tool argument can never abort member creation.
        """
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def _get_team_state(self, agent_ctx: AgentContextDeps) -> FileTeamState | None:
        """Create a FileTeamState for the current team, or None if not in a team.

        Args:
            agent_ctx: The per-turn agent context.

        Returns:
            A FileTeamState rooted at the configured base_dir, or None
            if no ``team_id`` is present in session metadata.
        """
        from wolfharness.capabilities.file_team_state import FileTeamState

        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        if team_id is None:
            return None
        # Prefer base_dir from session metadata (set by team_create) so
        # that team_status and other tools always find the state even if
        # team_mode_config is None in the per-turn AgentContextDeps.
        base_dir: str = agent_ctx.session.metadata.get(
            "team_base_dir",
            "",
        )
        if not base_dir:
            base_dir = (
                agent_ctx.team_mode_config.effective_base_dir
                if agent_ctx.team_mode_config is not None
                else tempfile.gettempdir()
            )
        return FileTeamState(base_dir)

    async def _create_member_session(
        self,
        agent_ctx: AgentContextDeps,
        agent_name: str,
        parent_session_id: str,
        description: str,
        *,
        tool_call_id: str | None = None,
        **metadata: Any,
    ) -> str:
        """Create a child session for a team member via SessionPool.

        Uses ``SessionPool.create_child_session()`` (generates ``ses_``
        prefixed sortable ID) and eagerly registers the agent via
        ``get_or_create_session_agent()``.  Emits ``SpawnSessionStart``
        so protocol servers (OpenCode, ACP) discover the child session.

        Args:
            agent_ctx: Per-turn agent context.
            agent_name: Name of the agent for the child session.
            parent_session_id: Parent session ID.
            description: Human-readable description.
            tool_call_id: ID of the tool call that triggered member creation,
                passed to ``SpawnSessionStart`` for protocol correlation.
            **metadata: Additional metadata (team_id, team_role, etc.).

        Returns:
            The child session ID.

        Raises:
            RuntimeError: If SessionPool is not available.
        """
        from wolfharness.agents.events.events import SpawnSessionStart

        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            msg = "SessionPool not available for member session creation"
            raise RuntimeError(msg)

        # Serialize session creation so generate_session_id() calls are
        # sequential within this agent.  Concurrent tool invocations share
        # the same millisecond, producing equal time.created values.
        async with self._create_session_lock:
            state = await session_pool.create_child_session(
                parent_session_id=parent_session_id,
                agent_name=agent_name,
                agent_type="native",
                **metadata,
            )
        child_sid = state.session_id

        # Eagerly register agent so receive_request / run_stream can
        # find it without a separate get_or_create_session_agent call.
        await session_pool.sessions.get_or_create_session_agent(
            child_sid,
            agent_name,
        )

        # Resolve display_name: prefer team_member_name (the member's display
        # name within the team), fall back to manifest config display_name.
        team_member_name: str | None = metadata.get("team_member_name")
        child_display_name: str | None = None
        if team_member_name is not None and team_member_name != agent_name:
            child_display_name = team_member_name
        else:
            child_config = session_pool.pool.manifest.agents.get(agent_name)
            if child_config is not None and child_config.display_name is not None:
                child_display_name = (
                    child_config.display_name if child_config.display_name != agent_name else None
                )

        # Emit SpawnSessionStart so protocol servers discover the child.
        event_bus = session_pool.event_bus
        if event_bus is not None:
            # Build metadata with team context for protocol display enrichment.
            spawn_metadata: dict[str, Any] = {"prompt": ""}
            team_id_meta: str | None = metadata.get("team_id")
            team_role_meta: str | None = metadata.get("team_role")
            team_member_name_meta: str | None = metadata.get("team_member_name")
            team_name_meta: str | None = metadata.get("team_name")
            if team_id_meta is not None:
                spawn_metadata["team_id"] = team_id_meta
            if team_name_meta is not None:
                spawn_metadata["team_name"] = team_name_meta
            if team_role_meta is not None:
                spawn_metadata["team_role"] = team_role_meta
            if team_member_name_meta is not None:
                spawn_metadata["team_member_name"] = team_member_name_meta

            spawn_event = SpawnSessionStart(
                child_session_id=child_sid,
                parent_session_id=parent_session_id,
                tool_call_id=tool_call_id or "",
                spawn_mechanism="task",
                source_name=agent_name,
                display_name=child_display_name,
                source_type="agent",
                depth=1,
                description=description,
                metadata=spawn_metadata,
            )
            await event_bus.publish(parent_session_id, spawn_event)

        return child_sid

    def _get_team_id(self, agent_ctx: AgentContextDeps) -> str | None:
        """Return the team_id from session metadata, or None."""
        team_id: str | None = agent_ctx.session.metadata.get("team_id")
        return team_id

    @staticmethod
    def _build_member_work_summary(
        team_state: FileTeamState,
        team_id: str,
        members: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        """Build a one-line work-status summary for each member.

        For each member, inspects tasks in the team state and produces:
        - In-progress tasks: ``"Currently working on: <subject>"``
        - Only completed tasks: ``"Just completed: <subject>"``
        - No tasks: ``"No active work"``

        Args:
            team_state: File team state for reading tasks.
            team_id: Team whose tasks to inspect.
            members: Members dict from team state.

        Returns:
            Mapping of member name to work-status description string.
        """
        all_tasks = team_state.list_tasks(team_id)
        summaries: dict[str, str] = {}
        for m_name in members:
            in_progress = [
                t
                for t in all_tasks
                if t.get("owner") == m_name and t.get("status") == "in_progress"
            ]
            completed = [
                t for t in all_tasks if t.get("owner") == m_name and t.get("status") == "completed"
            ]
            if in_progress:
                subjects = ", ".join(t.get("subject", "?") for t in in_progress)
                summaries[m_name] = f"Currently working on: {subjects}"
            elif completed:
                subjects = ", ".join(t.get("subject", "?") for t in completed)
                summaries[m_name] = f"Just completed: {subjects}"
            else:
                summaries[m_name] = "No active work"
        return summaries

    @staticmethod
    def _snapshot_task_mtimes(
        team_state: FileTeamState,
        team_id: str,
        task_ids: list[str],
    ) -> dict[str, float]:
        """Snapshot mtimes for the given task IDs.

        Returns a mapping of task_id to file mtime (0.0 if file missing).
        Used by watch loops to detect when specific tasks change.
        """
        tasks_dir = team_state._tasks_dir(team_id)
        mtimes: dict[str, float] = {}
        for tid in task_ids:
            path = tasks_dir / f"{tid}.json"
            mtimes[tid] = path.stat().st_mtime if path.exists() else 0.0
        return mtimes

    # ------------------------------------------------------------------
    # Universal tools
    # ------------------------------------------------------------------

    async def send_message(  # noqa: PLR0911, PLR0915
        self,
        ctx: RunContext[Any],
        to: Annotated[
            str,
            Field(
                description='Recipient member name. "*" broadcasts to all '
                "members (lead-only — returns error for non-lead agents)"
            ),
        ],
        body: Annotated[str, Field(description="Message body text")],
        message_type: Annotated[str, Field(description="Optional message type tag")] = "",
        persist_to_blackboard: Annotated[
            str | None,
            Field(
                description="If set, also writes the message body to this "
                "blackboard key (overwrite mode). Use when the message "
                "contains findings that should persist beyond the notification"
            ),
        ] = None,
    ) -> ToolReturn:
        """Send a message to a teammate's inbox.

        Use this for communication and coordination.  For assigning work
        to a team member, use ``task_create`` instead.

        Returns:
            Success or error message string.
        """
        # Message size enforcement.
        body_bytes = len(body.encode())
        if body_bytes > self._config.message_max_bytes:
            return ToolReturn(
                return_value=(
                    f"Message exceeds max size "
                    f"({body_bytes} > {self._config.message_max_bytes} bytes)"
                )
            )

        # Broadcast: lead-only.
        if to == "*":
            agent_ctx = self._resolve_agent_context(ctx)
            role: str = agent_ctx.session.metadata.get("team_role", "")
            if role != "lead":
                return ToolReturn(return_value="Broadcast is lead-only")

            team_state = self._get_team_state(agent_ctx)
            if team_state is None:
                return ToolReturn(return_value="Not in a team session")

            team_id: str = agent_ctx.session.metadata["team_id"]
            session_pool = agent_ctx.host.session_pool
            if session_pool is None:
                return ToolReturn(return_value="SessionPool not available")

            from wolfharness.capabilities.file_team_state import FileTeamState

            state_path = team_state._state_path(team_id)
            if not state_path.exists():
                return ToolReturn(return_value="Team state not found")
            state: dict[str, Any] = FileTeamState._read_json(state_path)
            members: dict[str, dict[str, str]] = state.get("members", {})

            from wolfharness.lifecycle.types import DeliveryMode

            mode = self._notice_mode
            delivered = 0
            lead_sid = agent_ctx.session.session_id
            msg_body = (
                f'<team-message from="{self._agent_name}" type="broadcast">'
                f"\n\n{body}\n\n</team-message>"
            )
            for member_name in members:
                target_sid = team_state.get_member_session_id(team_id, member_name)
                if target_sid is None or target_sid == lead_sid:
                    continue  # Skip self (lead broadcasting to itself).
                result = await session_pool.send_message(
                    target_sid,
                    self._wrap_notice_content(msg_body),
                    mode=mode,
                    source="accepted",
                    meta={"from": self._agent_name, "team_id": team_id},
                )
                # Distinguish None-as-queued from None-as-failure:
                # - STEER: None = failure.
                # - QUEUE: None = queued (success) OR failure. Check
                #   session existence to disambiguate.
                if result is not None:
                    delivered += 1
                elif mode is DeliveryMode.QUEUE:
                    target_session = session_pool.sessions.get_session(target_sid)
                    if target_session is not None and not target_session.closing:
                        delivered += 1
                team_state.write_message(
                    team_id,
                    member_name,
                    {"from": self._agent_name, "body": body},
                )
            return ToolReturn(return_value=f"Broadcast sent to {delivered} members")

        agent_ctx = self._resolve_agent_context(ctx)
        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        team_id = agent_ctx.session.metadata["team_id"]
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return ToolReturn(return_value="SessionPool not available")

        # Check member exists BEFORE bounds check to avoid creating
        # phantom entries in team state for non-existent members.
        target_sid = team_state.get_member_session_id(team_id, to)
        if target_sid is None:
            return ToolReturn(return_value=f"Member '{to}' not found or no session registered")

        # Bounds: max_member_turns and inbox_max_bytes checks.
        from wolfharness.capabilities.file_team_state import FileTeamState

        state_path = team_state._state_path(team_id)
        if state_path.exists():
            current_state: dict[str, Any] = FileTeamState._read_json(state_path)
            members_state: dict[str, dict[str, Any]] = current_state.get("members", {})
            member_info: dict[str, Any] = members_state.get(to, {})
            turn_count: int = member_info.get("turn_count", 0)
            if turn_count >= self._config.bounds.max_member_turns:
                return ToolReturn(
                    return_value=(
                        f"Member '{to}' has exceeded max turns "
                        f"({turn_count} >= {self._config.bounds.max_member_turns})"
                    )
                )

            existing_messages = team_state.read_messages(team_id, to)
            inbox_size = sum(len(json.dumps(m).encode()) for m in existing_messages)
            body_bytes_len = len(body.encode())
            if inbox_size + body_bytes_len > self._config.inbox_max_bytes:
                return ToolReturn(
                    return_value=(
                        f"Inbox exceeds max size ({inbox_size + body_bytes_len} > "
                        f"{self._config.inbox_max_bytes} bytes)"
                    )
                )

            member_info["turn_count"] = turn_count + 1
            members_state[to] = member_info
            current_state["members"] = members_state
            FileTeamState._atomic_write(state_path, current_state)

        from wolfharness.lifecycle.types import DeliveryMode

        mode = self._notice_mode
        msg_body = (
            f'<team-message from="{self._agent_name}" type="private">\n\n{body}\n\n</team-message>'
        )
        result = await session_pool.send_message(
            target_sid,
            self._wrap_notice_content(msg_body),
            mode=mode,
            source="accepted",
            meta={"from": self._agent_name, "team_id": team_id},
        )
        # Distinguish None-as-queued from None-as-failure:
        # - STEER mode: None always means failure (session not found/closing).
        # - QUEUE mode: None means queued (success) OR failure. Check
        #   session existence to disambiguate.
        if result is None:
            if mode is DeliveryMode.STEER:
                return ToolReturn(return_value=f"Failed to deliver message to '{to}'")
            # QUEUE mode — verify session still exists.
            target_session = session_pool.sessions.get_session(target_sid)
            if target_session is None or target_session.closing or target_session.is_closing:
                return ToolReturn(return_value=f"Failed to deliver message to '{to}'")

        # Persist to inbox for audit trail.
        team_state.write_message(
            team_id,
            to,
            {"from": self._agent_name, "body": body},
        )

        # Persist to blackboard if requested.
        result_msg = f"Message sent to {to}"
        if persist_to_blackboard is not None:
            try:
                team_state.write_blackboard(
                    team_id,
                    persist_to_blackboard,
                    {"text": body},
                    written_by=self._agent_name,
                    mode="overwrite",
                )
                result_msg += f"\nPersisted to blackboard key '{persist_to_blackboard}'"
            except Exception as exc:  # noqa: BLE001
                result_msg += f"\nBlackboard write failed for key '{persist_to_blackboard}': {exc}"

        return ToolReturn(return_value=result_msg)

    async def task_create(
        self,
        ctx: RunContext[Any],
        subject: Annotated[str, Field(description="Short task title")],
        owner: Annotated[
            str,
            Field(
                description="Team member name responsible for this task. "
                'Use empty string "" for unassigned tasks that can be claimed later. '
                "As lead, assign to the responsible agent — not yourself."
            ),
        ],
        description: Annotated[str, Field(description="Optional longer description")] = "",
        blocked_by: Annotated[
            list[str] | None,
            Field(description="Optional list of task_ids this task depends on"),
        ] = None,
        parent_id: Annotated[
            str | None,
            Field(
                description="Optional parent task ID to create a subtask. "
                "When set, any team member can create subtasks. "
                "When None (top-level), only lead can create"
            ),
        ] = None,
    ) -> ToolReturn:
        """Assign work to a team member by creating a task on the shared task board.

        Top-level tasks (``parent_id=None``) are lead-only.  Subtasks
        (``parent_id`` set) can be created by any team member.

        Returns:
            Success message with task_id, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if parent_id is None and role != "lead":
            return ToolReturn(return_value="Only lead can use task_create")
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        if parent_id is not None:
            parent = team_state.get_task(team_id, parent_id)
            if parent is None:
                return ToolReturn(return_value=f"Parent task not found: {parent_id}")

        task_dict: dict[str, Any] = {
            "subject": subject,
            "description": description,
            "blocked_by": blocked_by or [],
        }
        if parent_id is not None:
            task_dict["parent_id"] = parent_id
        if owner:
            task_dict["owner"] = owner

        try:
            task_id = team_state.create_task(team_id, task_dict)
        except ValueError as exc:
            return ToolReturn(return_value=str(exc))

        current_member: str = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )

        # Warn lead when assigning a task to itself.
        warnings: list[str] = []
        if owner and owner == current_member and role == "lead":
            warnings.append(
                f"Warning: task assigned to yourself ('{owner}'). "
                "As lead, you should assign work to other team members, "
                "not yourself. Use send_message only for coordination."
            )

        # Notify the assigned member about the new task.
        if owner != current_member:
            blocked_str = ""
            if blocked_by:
                blocked_str = f" (blocked by: {', '.join(blocked_by)})"
            notif_body = (
                f"New task assigned to you:\n"
                f"- [{task_id}] {subject}{blocked_str}\n"
                f"Use task_list to see details and task_get to read "
                f"full description."
            )
            await self._notify_member(agent_ctx, team_id, owner, notif_body)

        result = f"Task created: {task_id}"
        if warnings:
            result += "\n" + "\n".join(warnings)
        return ToolReturn(return_value=result)

    async def task_create_batch(
        self,
        ctx: RunContext[Any],
        tasks: Annotated[
            list[dict[str, Any]],
            Field(
                description="List of task definitions. Each dict supports: "
                '"subject" (required), "description", "owner", '
                '"blocked_by" (list of #N index refs or symbolic id refs), '
                '"parent_id" (also supports #N/id refs), "id" (optional '
                'symbolic name for cross-references), "progress_total" '
                '(optional int)")'
            ),
        ],
    ) -> ToolReturn:
        """Create multiple tasks atomically on the shared task board (lead-only).

        All tasks are created or none are.  Supports ``#N`` positional
        references and symbolic ``id`` references for ``blocked_by`` and
        ``parent_id`` fields.

        Returns:
            Success message with list of created task IDs, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use task_create_batch")
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        try:
            task_ids = team_state.create_tasks_batch(team_id, tasks)
        except ValueError as exc:
            return ToolReturn(return_value=str(exc))

        # Build mapping of #N / symbolic id -> actual task ID for the return.
        id_mapping: list[str] = []
        for i, tid in enumerate(task_ids):
            sym_id: str | None = tasks[i].get("id") if i < len(tasks) else None
            if sym_id:
                id_mapping.append(f"#{i} / '{sym_id}' -> {tid}")
            else:
                id_mapping.append(f"#{i} -> {tid}")

        mapping_str = "\n".join(id_mapping)
        return ToolReturn(
            return_value=(
                f"Created {len(task_ids)} tasks:\n{mapping_str}\nTask IDs: {', '.join(task_ids)}"
            )
        )

    async def task_list(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        parent_id: Annotated[
            str | None,
            Field(
                description="Filter to show only subtasks of this parent. "
                "None = show top-level tasks only (default)"
            ),
        ] = None,
        include_children: Annotated[
            bool,
            Field(
                description="If True, recursively include subtasks nested under each top-level task"
            ),
        ] = False,
        mine_only: Annotated[
            bool,
            Field(description="If True, show only tasks owned by you"),
        ] = False,
    ) -> ToolReturn:
        """List tasks on the shared task board.

        By default, shows only top-level tasks (no ``parent_id``).
        When ``parent_id`` is specified, shows only direct children of
        that task (as a flat list).  When ``include_children=True``,
        subtasks are nested inside parent tasks in the XML output.
        When ``mine_only=True``, filters to tasks owned by the calling member.

        Returns:
            XML task list with owner summary, or error string.
        """
        from wolfharness.capabilities.file_team_state import (
            TaskRecord,
            format_owner_summary,
        )

        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        all_tasks = team_state.list_tasks(team_id)
        if not all_tasks:
            return ToolReturn(return_value="<task_list>(empty)</task_list>")

        # Filter to mine_only if requested.
        if mine_only:
            current_member: str = agent_ctx.session.metadata.get(
                "team_member_name",
                self._agent_name,
            )
            all_tasks = [t for t in all_tasks if t.get("owner") == current_member]
            if not all_tasks:
                return ToolReturn(
                    return_value=(f"<task_list>(empty)</task_list>\n0 tasks for {current_member}")
                )

        # Build TaskRecord list for owner summary.
        task_records = [TaskRecord.from_dict(t) for t in all_tasks]
        owner_summary = format_owner_summary(task_records)

        task_by_id: dict[str, dict[str, Any]] = {t.get("task_id", ""): t for t in all_tasks}

        if parent_id is not None:
            # Show only direct children of the specified parent (flat list).
            children = [t for t in all_tasks if t.get("parent_id") == parent_id]
            if not children:
                return ToolReturn(return_value="<task_list>(empty)</task_list>")
            lines = ["<task_list>"]
            lines.append(f"<!-- {owner_summary} -->")
            lines.extend(self._format_task_xml(t, indent=2) for t in children)
            lines.append("</task_list>")
            return ToolReturn(return_value="\n".join(lines))

        # Show top-level tasks only (no parent_id).
        top_level = [t for t in all_tasks if t.get("parent_id") is None]
        if not top_level:
            return ToolReturn(return_value="<task_list>(empty)</task_list>")

        lines = ["<task_list>"]
        lines.append(f"<!-- {owner_summary} -->")
        for t in top_level:
            lines.append(
                self._format_task_xml(
                    t,
                    indent=2,
                    include_children=include_children,
                    task_by_id=task_by_id,
                )
            )
        lines.append("</task_list>")
        return ToolReturn(return_value="\n".join(lines))

    @staticmethod
    def _format_task_xml(
        t: dict[str, Any],
        *,
        indent: int = 2,
        include_children: bool = False,
        task_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Format a task dict as an XML element, optionally nesting subtasks.

        The ``owner`` attribute is always present (``owner=""`` for
        unassigned tasks).  When both ``progress_current`` and
        ``progress_total`` are set, a ``progress="{current}/{total}"``
        attribute is included.

        Args:
            t: Task dict.
            indent: Number of spaces for indentation.
            include_children: If True, nest subtask XML inside this task.
            task_by_id: Lookup map for resolving children by task_id.

        Returns:
            XML string for the task.
        """
        tid = t.get("task_id", "?")
        status = t.get("status", "?")
        owner = t.get("owner", "")
        subject = t.get("subject", "")
        description = t.get("description", "")
        last_note = t.get("last_note", "")
        blocked = t.get("is_unblocked", True)
        blocked_attr = "" if blocked else ' blocked="true"'
        # Always-present owner attribute (even for unassigned tasks).
        owner_attr = f' owner="{owner}"'
        # Progress attribute when both values are set.
        progress_current: int | None = t.get("progress_current")
        progress_total: int | None = t.get("progress_total")
        progress_attr = ""
        if progress_current is not None and progress_total is not None:
            progress_attr = f' progress="{progress_current}/{progress_total}"'
        pad = " " * indent

        parts: list[str] = []
        parts.append(
            f'{pad}<task id="{tid}" status="{status}"{owner_attr}{blocked_attr}{progress_attr}>'
        )
        content_line = f"{subject}: {description}" if description else subject
        parts.append(f"{pad}  {content_line}")
        if last_note:
            parts.append(f"{pad}  note: {last_note}")

        if include_children and task_by_id is not None:
            children_ids: list[str] = t.get("children", [])
            for child_id in children_ids:
                child = task_by_id.get(child_id)
                if child is not None:
                    child_xml = TeamCommCapability._format_task_xml(
                        child,
                        indent=indent + 4,
                        include_children=True,
                        task_by_id=task_by_id,
                    )
                    # Wrap as <subtask> instead of <task>
                    child_xml = child_xml.replace("<task ", "<subtask ", 1)
                    child_xml = child_xml.replace("</task>", "</subtask>", 1)
                    parts.append(child_xml)

        parts.append(f"{pad}</task>")
        return "\n".join(parts)

    async def task_update(  # noqa: PLR0911, PLR0915
        self,
        ctx: RunContext[Any],
        task_id: Annotated[str, Field(description="ID of the task to update")],
        status: Annotated[
            str,
            Field(description='New status (e.g. "in_progress", "completed"). Empty = no change'),
        ] = "",
        owner: Annotated[str, Field(description="New owner name. Empty = no change")] = "",
        technical_note: Annotated[
            str,
            Field(
                description="Optional technical note for the task audit trail. "
                "NOT for communication — use send_message for that. "
                "Appended to the task's update log"
            ),
        ] = "",
        handoff_to: Annotated[
            str | None,
            Field(
                description="When setting status to 'completed', optionally "
                "hand off to another member. Sends them a notification with "
                "the task context and any blackboard keys you've written to"
            ),
        ] = None,
        handoff_context_keys: Annotated[
            list[str] | None,
            Field(
                description="Blackboard keys to include in the handoff "
                "notification so the receiver knows where to find context"
            ),
        ] = None,
        progress_current: Annotated[
            int | None,
            Field(description="Current progress value. Must be <= progress_total"),
        ] = None,
        progress_total: Annotated[
            int | None,
            Field(description="Total progress value (denominator)"),
        ] = None,
    ) -> ToolReturn:
        """Update a task's status or owner on the shared task board.

        Lead can update any task.  Members can update only tasks they
        own or tasks with no owner (to claim them).

        Returns:
            Updated task as XML, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        updates: dict[str, Any] = {}
        if status:
            updates["status"] = status
        if owner:
            updates["owner"] = owner
        if technical_note:
            updates["last_note"] = technical_note
            updates["updated_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            updates["updated_by"] = agent_ctx.session.metadata.get(
                "team_member_name",
                self._agent_name,
            )

        # --- Progress tracking validation (tasks 42-45) ---
        if progress_current is not None and progress_current < 0:
            return ToolReturn(
                return_value=f"progress_current must be non-negative, got {progress_current}"
            )
        if progress_total is not None and progress_total < 0:
            return ToolReturn(
                return_value=f"progress_total must be non-negative, got {progress_total}"
            )
        if (
            progress_current is not None
            and progress_total is not None
            and progress_current > progress_total
        ):
            return ToolReturn(
                return_value=(
                    f"progress_current ({progress_current}) must be <= "
                    f"progress_total ({progress_total})"
                )
            )

        # Auto-complete: when status="completed" and progress_total is
        # already set on the task but progress_current is not explicitly
        # provided in this call, auto-set progress_current = progress_total.
        existing_task = team_state.get_task(team_id, task_id)
        if existing_task is None:
            return ToolReturn(return_value=f"Task not found: {task_id}")
        existing_progress_total: int | None = existing_task.get("progress_total")
        if (
            status == "completed"
            and progress_current is None
            and existing_progress_total is not None
        ):
            progress_current = existing_progress_total

        if not updates and progress_current is None and progress_total is None:
            return ToolReturn(return_value="No updates specified")

        current_member: str = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )

        # Permission check: lead bypasses, members need ownership or unclaimed.
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            current_owner: str = existing_task.get("owner", "")
            if current_owner and current_owner != current_member:
                return ToolReturn(
                    return_value=(
                        f"Task {task_id} is owned by '{current_owner}'. "
                        f"Use send_message(to='{current_owner}', ...) to "
                        f"coordinate, or ask the lead to reassign."
                    )
                )

        try:
            updated = team_state.update_task(
                team_id,
                task_id,
                updates,
                progress_current=progress_current,
                progress_total=progress_total,
            )
        except (FileNotFoundError, OSError):
            return ToolReturn(return_value=f"Task not found: {task_id}")
        tid = updated.get("task_id", "?")
        status = updated.get("status", "?")
        owner = updated.get("owner", "")
        subject = updated.get("subject", "")
        description = updated.get("description", "")
        last_note = updated.get("last_note", "")

        # --- Handoff guard: only triggers when status == "completed" ---
        handoff_messages: list[str] = []
        if handoff_to is not None and status != "completed":
            handoff_messages.append(
                f"Warning: handoff_to='{handoff_to}' ignored — "
                "handoff only applies when status='completed'"
            )

        # --- Push notifications for task changes ---
        # 1. Notify new owner when a task is assigned (skip if already completed).
        if "owner" in updates and owner and owner != current_member and status != "completed":
            notif_body = (
                f"Task assigned to you:\n"
                f"- [{tid}] {subject}\n"
                f"Use task_list to see details and task_get to read "
                f"full description."
            )
            await self._notify_member(agent_ctx, team_id, owner, notif_body)
        # 2. Notify downstream task owners when a dependency completes.
        if "status" in updates and updates.get("status") == "completed":
            all_tasks = team_state.list_tasks(team_id)
            for t in all_tasks:
                if tid in t.get("blocked_by", []) and t.get("is_unblocked"):
                    downstream_owner: str = t.get("owner", "")
                    if not downstream_owner:
                        continue
                    # Skip self-notification: if the dependent task is
                    # owned by the same member who completed the dependency.
                    if downstream_owner == current_member:
                        continue
                    downstream_subject = t.get("subject", "?")
                    downstream_id = t.get("task_id", "?")
                    dep_notif_body = (
                        f'<team-message type="dependency_resolved">\n'
                        f"Completed task '{subject}' (id={tid}) was blocking "
                        f"your task '{downstream_subject}' (id={downstream_id}).\n"
                        f"The dependency is now resolved — your task is unblocked.\n"
                        f"Use task_list to see details.\n"
                        f"</team-message>"
                    )
                    await self._notify_member(
                        agent_ctx,
                        team_id,
                        downstream_owner,
                        dep_notif_body,
                    )

        # --- Handoff notification (tasks 20-26) ---
        if handoff_to is not None and status == "completed":
            # Look up the member in team state.
            from wolfharness.capabilities.file_team_state import FileTeamState

            handoff_state_path = team_state._state_path(team_id)
            if handoff_state_path.exists():
                handoff_state: dict[str, Any] = FileTeamState._read_json(
                    handoff_state_path,
                )
                handoff_members: dict[str, dict[str, Any]] = handoff_state.get(
                    "members",
                    {},
                )
                if handoff_to in handoff_members:
                    keys_list = ""
                    if handoff_context_keys:
                        keys_list = ", ".join(handoff_context_keys)
                    tech_note_line = ""
                    if technical_note:
                        tech_note_line = f"Technical note: {technical_note}\n"
                    handoff_body = (
                        f'<team-message from="{current_member}" type="handoff">\n'
                        f'Task "{subject}" (id={tid}) has been completed and '
                        f"handed off to you.\n"
                        f"Context is available in blackboard keys: {keys_list}\n"
                        f"{tech_note_line}"
                        f"Please review and continue the work.\n"
                        f"</team-message>"
                    )
                    try:
                        await self._notify_member(
                            agent_ctx,
                            team_id,
                            handoff_to,
                            handoff_body,
                        )
                        handoff_messages.append(f"handoff notification sent to {handoff_to}")
                    except Exception as exc:  # noqa: BLE001
                        handoff_messages.append(f"handoff notification delivery failed: {exc}")
                else:
                    handoff_messages.append(f"handoff failed: member '{handoff_to}' not found")
            else:
                handoff_messages.append("handoff failed: team state not found")

        owner_attr = f' owner="{owner}"' if owner else ""
        content = f"{subject}: {description}" if description else subject
        note_attr = f"\n{last_note}" if last_note else ""
        result_parts = [
            f'<task id="{tid}" status="{status}"{owner_attr}>\n{content}{note_attr}\n</task>'
        ]
        if handoff_messages:
            result_parts.extend(handoff_messages)
        return ToolReturn(return_value="\n".join(result_parts))

    async def task_get(
        self,
        ctx: RunContext[Any],
        task_id: Annotated[str, Field(description="ID of the task to retrieve")],
        include_children: Annotated[
            bool,
            Field(description="If True, include subtasks nested in the output"),
        ] = False,
    ) -> ToolReturn:
        """Get a single task by ID.

        Returns:
            Task details as XML, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        task = team_state.get_task(team_id, task_id)
        if task is None:
            return ToolReturn(return_value=f"Task not found: {task_id}")

        task_by_id: dict[str, dict[str, Any]] | None = None
        if include_children:
            all_tasks = team_state.list_tasks(team_id)
            task_by_id = {t.get("task_id", ""): t for t in all_tasks}
            # Ensure the requested task is from the enriched list (has children).
            enriched = task_by_id.get(task_id)
            if enriched is not None:
                task = enriched

        xml = self._format_task_xml(
            task,
            indent=0,
            include_children=include_children,
            task_by_id=task_by_id,
        )
        return ToolReturn(return_value=xml)

    async def read_blackboard(
        self,
        ctx: RunContext[Any],
        key: Annotated[str, Field(description="Blackboard key to read")],
        limit: Annotated[
            int,
            Field(
                description="Maximum number of lines to return (default 200). "
                "Use a smaller value to reduce context window usage"
            ),
        ] = 200,
        offset: Annotated[
            int,
            Field(
                description="Starting line number, 0-indexed (default 0). "
                "Ignored when context is provided"
            ),
        ] = 0,
        context: Annotated[
            int | None,
            Field(
                description="If provided, center the output around this line "
                "number (0-indexed). Returns limit/2 lines before and after "
                "the specified line. Overrides offset"
            ),
        ] = None,
    ) -> ToolReturn:
        """Read a key from the shared blackboard with line-based pagination.

        Returns:
            JSON value + metadata + paginated lines, or "Key not found" /
            error string.  When the result is truncated, a trailing
            ``<!--- total=N offset=M limit=K has_more=true --->`` hint
            is appended so the caller knows how to fetch the next page.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        result = team_state.read_blackboard(team_id, key)
        if result is None:
            return ToolReturn(return_value="Key not found")

        value_text: str = result.get("value", {}).get("text", "")
        version = result.get("version", 0)
        written_by = result.get("written_by", "unknown")
        written_at = result.get("written_at", "")

        lines = value_text.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return ToolReturn(
                return_value=[
                    (
                        f'<blackboard version="{version}" written_by="{written_by}" '
                        f'written_at="{written_at}" total_lines="0">'
                    ),
                    "</blackboard>",
                ],
            )

        # Determine the effective offset and limit.
        if context is not None:
            # Center around the context line: limit/2 before, rest after.
            half = max(limit // 2, 1)
            eff_offset = max(context - half, 0)
            eff_limit = limit
        else:
            eff_offset = max(offset, 0)
            eff_limit = max(limit, 1)

        # Clamp offset to valid range.
        eff_offset = min(eff_offset, total_lines)

        page_lines = lines[eff_offset : eff_offset + eff_limit]

        has_more = eff_offset + eff_limit < total_lines

        header = (
            f'<blackboard version="{version}" written_by="{written_by}" '
            f'written_at="{written_at}" total_lines="{total_lines}" '
            f'offset="{eff_offset}" limit="{eff_limit}">'
        )

        parts: list[str] = [header, *page_lines]
        if has_more:
            parts.append(
                f"<!--- total={total_lines} offset={eff_offset} "
                f"limit={eff_limit} has_more=true --->"
            )
        parts.append("</blackboard>")

        return ToolReturn(return_value=parts)

    async def write_blackboard(
        self,
        ctx: RunContext[Any],
        key: Annotated[str, Field(description="Blackboard key to write")],
        value: Annotated[
            str,
            Field(
                description="Value to store. Can be any text format: "
                "inline JSON, Markdown, or plain text"
            ),
        ],
        expected_version: Annotated[
            int | None,
            Field(
                description="Expected current version for optimistic locking. "
                "If None, no version check is performed"
            ),
        ] = None,
        mode: Annotated[
            str,
            Field(
                description='Write mode: "overwrite" (default) replaces '
                'the value entirely; "append" concatenates to the '
                "existing value. Use append for accumulating findings "
                "or logs across multiple writes"
            ),
        ] = "overwrite",
    ) -> ToolReturn:
        """Write a key to the shared blackboard with optimistic locking.

        Returns:
            "Written, version=N" on success, or "Conflict: current version is N".
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        result = team_state.write_blackboard(
            team_id,
            key,
            {"text": value},
            expected_version=expected_version,
            written_by=self._agent_name,
            mode=mode,
        )

        # Bounds: max_size_mb check on the resulting blackboard file.
        if result.startswith("Written"):
            key_path = team_state._validate_key(key, team_state._blackboard_dir(team_id))
            file_size = key_path.stat().st_size
            max_size = self._config.blackboard.max_size_mb * 1024 * 1024
            if file_size > max_size:
                # Clean up the oversized file so subsequent reads return
                # "Key not found" instead of stale oversized data.
                key_path.unlink(missing_ok=True)
                return ToolReturn(
                    return_value=(
                        f"Blackboard write exceeds max size "
                        f"({file_size / 1024 / 1024:.1f}MB > "
                        f"{self._config.blackboard.max_size_mb}MB)"
                    )
                )

        return ToolReturn(return_value=result)

    async def list_blackboard(
        self,
        ctx: RunContext[Any],
        watch: Annotated[
            bool,
            Field(
                description="If True, block until the blackboard keys change "
                "(new key added or removed) or timeout expires, then return "
                "the current state. If False, return immediately"
            ),
        ] = False,
        timeout: Annotated[
            int,
            Field(
                description="Maximum seconds to wait when watch=True. "
                "If <= 0, uses the configured max watch timeout (default 120s). "
                "Always capped by max_watch_timeout config. "
                "Returns current state if timeout expires without changes"
            ),
        ] = 300,
        watch_task_ids: Annotated[
            list[str] | None,
            Field(
                description="When watch=True, only watch for changes to these "
                "specific task IDs. The watch ends as soon as any listed task "
                "file is modified. If empty or None, watches for any "
                "blackboard key change instead"
            ),
        ] = None,
    ) -> ToolReturn:
        """List all keys on the shared blackboard.

        Returns:
            JSON array of key names, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        keys = team_state.list_blackboard(team_id)

        if watch:
            import time

            effective_timeout = (
                min(timeout, self._config.max_watch_timeout)
                if timeout > 0
                else self._config.max_watch_timeout
            )
            deadline = time.monotonic() + effective_timeout

            if watch_task_ids:
                initial_mtimes = self._snapshot_task_mtimes(team_state, team_id, watch_task_ids)
                while time.monotonic() < deadline:
                    await asyncio.sleep(1)
                    current_mtimes = self._snapshot_task_mtimes(team_state, team_id, watch_task_ids)
                    if current_mtimes != initial_mtimes:
                        keys = sorted(team_state.list_blackboard(team_id))
                        break
                else:
                    return ToolReturn(
                        return_value=(
                            "<blackboard_keys> (watch timeout, no task changes)\n"
                            + "\n".join(sorted(keys))
                            + "\n</blackboard_keys>"
                        )
                    )
            else:
                initial = set(keys)
                while time.monotonic() < deadline:
                    await asyncio.sleep(1)
                    current = set(team_state.list_blackboard(team_id))
                    if current != initial:
                        keys = sorted(current)
                        break
                else:
                    return ToolReturn(
                        return_value=(
                            "<blackboard_keys> (watch timeout, no changes)\n"
                            + "\n".join(sorted(keys))
                            + "\n</blackboard_keys>"
                        )
                    )

        if not keys:
            return ToolReturn(return_value="<blackboard_keys>(empty)</blackboard_keys>")
        return ToolReturn(
            return_value="<blackboard_keys>\n" + "\n".join(keys) + "\n</blackboard_keys>"
        )

    async def team_status(  # noqa: PLR0915
        self,
        ctx: RunContext[Any],
        watch: Annotated[
            bool,
            Field(
                description="If True, block until team state changes "
                "(member status, task updates, member joins/leaves) or "
                "timeout expires, then return current status. If False, "
                "return immediately"
            ),
        ] = False,
        timeout: Annotated[
            int,
            Field(
                description="Maximum seconds to wait when watch=True. "
                "If <= 0, uses the configured max watch timeout (default 120s). "
                "Always capped by max_watch_timeout config. "
                "Returns current status if timeout expires without changes"
            ),
        ] = 300,
        watch_task_ids: Annotated[
            list[str] | None,
            Field(
                description="When watch=True, only watch for changes to these "
                "specific task IDs. The watch ends as soon as any listed task "
                "file is modified. If empty or None, watches for any team "
                "state change instead"
            ),
        ] = None,
    ) -> ToolReturn:
        """Get the current status of the team.

        Returns:
            Formatted status string with team name, members, and status.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        from wolfharness.capabilities.file_team_state import FileTeamState

        state_path = team_state._state_path(team_id)
        if not state_path.exists():
            return ToolReturn(return_value="Team state not found")

        state: dict[str, Any] = FileTeamState._read_json(state_path)
        team_name: str = state.get("team_name", "unknown")
        status: str = state.get("status", "unknown")
        members: dict[str, dict[str, Any]] = state.get("members", {})

        # Query tasks for member association.
        all_tasks = team_state.list_tasks(team_id)
        max_turns = self._config.bounds.max_member_turns

        # Access SessionPool for runtime member state.
        session_pool = agent_ctx.host.session_pool

        member_lines: list[str] = []
        for m_name, info in members.items():
            sid: str = info.get("session_id", "")
            agent_name: str = info.get("agent", m_name)
            turn_count: int = info.get("turn_count", 0)

            # Determine runtime status from SessionPool.
            if session_pool is not None and sid:
                member_session = session_pool.sessions.get_session(sid)
                if member_session is None:
                    runtime_status = "offline"
                elif member_session.closing or member_session.is_closing:
                    runtime_status = "closing"
                elif member_session.current_run_id is not None:
                    runtime_status = "busy"
                else:
                    runtime_status = "idle"
            else:
                runtime_status = "unregistered" if not sid else "unknown"

            # Count inbox messages.
            inbox_count = len(team_state.read_messages(team_id, m_name))

            # Find tasks owned by this member.
            member_tasks = [
                t for t in all_tasks if t.get("owner") == m_name and t.get("status") != "completed"
            ]
            task_summary = (
                f"tasks={len(member_tasks)}"
                if not member_tasks
                else f"tasks={len(member_tasks)} ("
                + ", ".join(
                    f"{t.get('status', '?')}: {t.get('subject', '?')}" for t in member_tasks
                )
                + ")"
            )

            member_lines.append(
                f"  - `{m_name}` (agent=`{agent_name}`, status=`{runtime_status}`, "
                f"turns={turn_count}/{max_turns}, inbox={inbox_count}, {task_summary})"
            )

        lines = [
            f"Team: {team_name}",
            f"Status: {status}",
            f"Team ID: {team_id}",
            f"Members ({len(members)}):",
            *member_lines,
        ]
        result = "\n".join(lines)

        if watch:
            import time

            effective_timeout = (
                min(timeout, self._config.max_watch_timeout)
                if timeout > 0
                else self._config.max_watch_timeout
            )
            deadline = time.monotonic() + effective_timeout

            if watch_task_ids:
                initial_mtimes = self._snapshot_task_mtimes(team_state, team_id, watch_task_ids)
                while time.monotonic() < deadline:
                    await asyncio.sleep(1)
                    current_mtimes = self._snapshot_task_mtimes(team_state, team_id, watch_task_ids)
                    if current_mtimes != initial_mtimes:
                        state = FileTeamState._read_json(state_path)
                        break
                else:
                    result += "\n(watch timeout, no task changes detected)"
            else:
                initial_snapshot = state_path.stat().st_mtime
                while time.monotonic() < deadline:
                    await asyncio.sleep(1)
                    if state_path.exists():
                        current_mtime = state_path.stat().st_mtime
                        if current_mtime != initial_snapshot:
                            state = FileTeamState._read_json(state_path)
                            break
                else:
                    result += "\n(watch timeout, no changes detected)"

        return ToolReturn(return_value=result)

    # ------------------------------------------------------------------
    # Lead-only tools
    # ------------------------------------------------------------------

    async def team_create(  # noqa: PLR0911, PLR0915
        self,
        ctx: RunContext[Any],
        name: Annotated[str, Field(description="Human-readable team name")],
        members: Annotated[
            list[dict[str, Any]],
            Field(
                description='List of member dicts, each with "agent" '
                '(registered agent name) and "name" (display name) keys. '
                'Optional keys: "instructions" (per-member instructions), '
                '"skills" (list of skill names or skill:// URIs injected as '
                "instruction text). Example: "
                '[{"agent": "historian", "name": "researcher", '
                '"skills": ["lodestone"]}, '
                '{"agent": "logician", "name": "analyst"}]'
            ),
        ],
        prompt: Annotated[
            str,
            Field(
                description="Optional task instructions sent to each member "
                "after team creation. If empty, only the protocol template "
                "(role description and tool guide) is sent"
            ),
        ] = "",
    ) -> ToolReturn:
        """Create a new team with eligible members (lead-only).

        Returns:
            Success message with team_id, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use team_create")

        # Config defaults: when LLM passes empty members, use defaults config.
        if not members and self._config.defaults is not None:
            members = [
                {
                    "name": m.name,
                    "agent": m.agent,
                    "instructions": m.instructions,
                    "skills": m.skills,
                }
                for m in self._config.defaults.members
            ]

        # Eligibility checks.
        for member in members:
            agent_name: str = member.get("agent", "")
            if not agent_ctx.agent_registry.exists(agent_name):
                return ToolReturn(return_value=f"Agent '{agent_name}' not found in registry")
            if agent_name not in self._config.member_eligible:
                return ToolReturn(
                    return_value=(f"Agent '{agent_name}' is not eligible for team membership")
                )

        # Bounds: max_members check.
        if len(members) > self._config.bounds.max_members:
            return ToolReturn(
                return_value=(
                    f"Team exceeds max_members ({len(members)} > {self._config.bounds.max_members})"
                )
            )

        # Generate team_id and create state.
        team_id = f"team_{uuid.uuid4().hex[:8]}"
        lead_session_id: str = agent_ctx.session.session_id

        from wolfharness.capabilities.file_team_state import FileTeamState

        base_dir = (
            agent_ctx.team_mode_config.effective_base_dir
            if agent_ctx.team_mode_config is not None
            else tempfile.gettempdir()
        )
        team_state = FileTeamState(base_dir)
        team_state.init(
            team_id,
            name,
            [{"name": m["name"], "agent": m["agent"]} for m in members],
        )

        # Register the lead as a member so other members can send_message
        # to the lead by name.  The lead's member name comes from session
        # metadata (set by the factory), falling back to the agent name.
        lead_member_name = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )
        team_state.register_member(team_id, lead_member_name, lead_session_id)

        # Record started_at timestamp for wall-clock enforcement.
        state = team_state._read_json(team_state._state_path(team_id))
        state["started_at"] = datetime.datetime.now(datetime.UTC).isoformat()
        team_state._atomic_write(team_state._state_path(team_id), state)

        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return ToolReturn(return_value="SessionPool not available")

        from wolfharness.lifecycle.types import DeliveryMode

        created_sessions: list[str] = []
        try:
            for member in members:
                member_instructions: str = member.get("instructions", "")
                # Inject requested skills as instruction text into the
                # member's system prompt (pure prompt guidance — no tool/MCP
                # assembly). Visibility is checked against the member agent's
                # node scope, and loading failures degrade to error text.
                member_skills = self._coerce_skill_names(member.get("skills"))
                if member_skills:
                    skills_content = await self._format_member_skills_instructions(
                        ctx,
                        member_skills,
                        member["agent"],
                    )
                    if skills_content:
                        member_instructions = f"{skills_content}\n\n{member_instructions}"
                member_session_id = await self._create_member_session(
                    agent_ctx,
                    member["agent"],
                    parent_session_id=lead_session_id,
                    description=f"Team member: {member['name']}",
                    tool_call_id=ctx.tool_call_id,
                    team_id=team_id,
                    team_name=name,
                    team_role="member",
                    team_member_name=member["name"],
                    team_member_instructions=member_instructions,
                )
                created_sessions.append(member_session_id)
                team_state.register_member(
                    team_id,
                    member["name"],
                    member_session_id,
                    agent=member["agent"],
                )
                # Propagate member display name to the agent instance so
                # protocol frontends (ACP, OpenCode) show the correct name.
                member_agent = session_pool.sessions._session_agents.get(
                    member_session_id,
                )
                if member_agent is not None:
                    member_agent._display_name = member["name"]
                # Build initial prompt with member roster so the new
                # member knows who their teammates are.
                roster_lines: list[str] = []
                for m in members:
                    role_label = "lead" if m["name"] == lead_member_name else "member"
                    roster_lines.append(
                        f"  - `{m['name']}` (agent=`{m['agent']}`, role=`{role_label}`)"
                    )
                roster = "\n".join(roster_lines)
                base_prompt = self._config.protocol_template.format(
                    team_name=name,
                    role="member",
                    member_name=member["name"],
                )
                full_prompt = f"{base_prompt}\n\n## Team Members\n{roster}"
                if prompt:
                    full_prompt += (
                        f"\n\n## Task\n{prompt}"
                        "\n\nRemember to report progress regularly using "
                        '`task_update(technical_note="...")`.'
                    )
                await session_pool.send_message(
                    member_session_id,
                    full_prompt,
                    mode=DeliveryMode.QUEUE,
                    source="accepted",
                    meta={"from": self._agent_name, "team_id": team_id},
                )
        except Exception as exc:  # noqa: BLE001
            for sid in created_sessions:
                with contextlib.suppress(Exception):
                    await session_pool.close_session(sid)
            with contextlib.suppress(Exception):
                team_state.cleanup(team_id)
            return ToolReturn(return_value=f"Failed to create team: {exc}")

        # Write team_id back to session metadata so subsequent tool calls
        # can access the team state without requiring a new session.
        agent_ctx.session.metadata["team_id"] = team_id
        agent_ctx.session.metadata["team_name"] = name
        agent_ctx.session.metadata["team_base_dir"] = base_dir
        agent_ctx.session.metadata["team_role"] = "lead"
        # Store member session IDs so the auto-cleanup callback (and
        # team_delete) can close them without re-reading team state.
        agent_ctx.session.metadata["team_member_sessions"] = list(created_sessions)

        # Schedule auto-cleanup: when the lead's RunHandle terminates
        # (complete_event fires), close all member sessions to prevent
        # leaks.  This covers the scenario where the lead's run finishes
        # but ``close_session(lead)`` is not called (e.g. protocol server
        # keeps the lead session alive for follow-ups).
        self._schedule_member_cleanup(agent_ctx, lead_session_id, list(created_sessions))

        team_dir = team_state._team_dir(team_id)
        logger.info(
            "Team created — state at %s",
            str(team_dir),
            team_id=team_id,
            team_name=name,
            member_count=len(members),
        )

        return ToolReturn(
            return_value=(f"Team '{name}' created with {len(members)} members. team_id={team_id}")
        )

    def _schedule_member_cleanup(
        self,
        agent_ctx: AgentContextDeps,
        lead_session_id: str,
        member_session_ids: list[str],
    ) -> None:
        """Schedule a background task to close member sessions when the lead goes idle.

        Polls ``session.last_active_at`` every 30 seconds.  When the lead
        has been inactive for longer than ``idle_timeout`` (default 300s),
        closes every member session whose ID is still recorded in
        ``session.metadata["team_member_sessions"]`` (``team_delete``
        clears this list to signal manual cleanup was already performed).

        This approach correctly handles protocol-server sessions (e.g.
        OpenCode) where the lead's RunLoop stays alive between turns —
        the cleanup fires based on wall-clock inactivity, not on
        ``complete_event`` which may never fire.

        Args:
            agent_ctx: The lead agent's per-turn context.
            lead_session_id: The lead session ID.
            member_session_ids: Member session IDs to close on idle.
        """
        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return

        import time

        # Configurable via TeamModeConfig (YAML) or class attributes (tests).
        idle_timeout = self._config.idle_timeout
        poll_interval = self._config.poll_interval

        async def _cleanup_when_idle() -> None:
            """Poll lead activity; close members after idle timeout.

            Uses ``last_active_at`` as the primary idle signal but **also
            checks ``current_run_id``** on both the lead and every member
            before closing.  This prevents closing sessions that are still
            actively processing (e.g. a long model call or tool execution
            that does not update ``last_active_at``).
            """
            try:
                while True:
                    await asyncio.sleep(poll_interval)
                    # Check if team_delete already cleaned up.
                    current_session = session_pool.sessions.get_session(
                        lead_session_id,
                    )
                    if current_session is None:
                        return  # Lead session closed — cascade handled it.
                    remaining = current_session.metadata.get(
                        "team_member_sessions",
                    )
                    if not remaining:
                        return  # team_delete already closed members.

                    # Check idle time via last_active_at.
                    now = time.monotonic()
                    idle_seconds = now - current_session.last_active_at
                    if idle_seconds < idle_timeout:
                        continue  # Lead still active, keep waiting.

                    # Idle threshold reached — but before closing, verify
                    # that neither the lead nor any member has an active
                    # run.  ``last_active_at`` is only updated by
                    # ``send_message()``, so a long-running turn (model
                    # calls, tool execution) can make the lead appear idle
                    # even though it is actively processing.
                    if current_session.current_run_id is not None:
                        logger.debug(
                            "Lead idle for %.0fs but has active run, deferring member cleanup",
                            idle_seconds,
                            lead_session_id=lead_session_id,
                            run_id=current_session.current_run_id,
                        )
                        continue

                    any_member_running = False
                    for msid in member_session_ids:
                        member_session = session_pool.sessions.get_session(msid)
                        if member_session is not None and member_session.current_run_id is not None:
                            any_member_running = True
                            break

                    if any_member_running:
                        logger.debug(
                            "Lead idle for %.0fs but members still running, deferring cleanup",
                            idle_seconds,
                            lead_session_id=lead_session_id,
                        )
                        continue

                    # Lead is idle AND no member has an active run —
                    # safe to close.
                    logger.info(
                        "Lead idle for %.0fs, auto-closing member sessions",
                        idle_seconds,
                        lead_session_id=lead_session_id,
                    )
                    for msid in member_session_ids:
                        try:
                            await session_pool.close_session(msid)
                        except Exception:
                            logger.exception(
                                "Failed to auto-close member session",
                                member_session_id=msid,
                                lead_session_id=lead_session_id,
                            )
                    # Clear list so cascade close is a no-op.
                    current_session.metadata["team_member_sessions"] = []
                    return  # Cleanup done, exit loop.
            except asyncio.CancelledError:
                pass  # Pool shutdown — exit gracefully.

        task = asyncio.create_task(_cleanup_when_idle())
        _cleanup_tasks.add(task)

        def _on_done(t: asyncio.Task[Any]) -> None:
            _cleanup_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error(
                    "Member session cleanup task failed: %s",
                    t.exception(),
                    lead_session_id=lead_session_id,
                )

        task.add_done_callback(_on_done)

    async def team_delete(self, ctx: RunContext[Any]) -> ToolReturn:
        """Delete the current team and close all member sessions (lead-only).

        Returns:
            ``"Team deleted"`` on success, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use team_delete")

        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        from wolfharness.capabilities.file_team_state import FileTeamState

        state_path = team_state._state_path(team_id)
        if not state_path.exists():
            return ToolReturn(return_value="Team state not found")
        state: dict[str, Any] = FileTeamState._read_json(state_path)
        members: dict[str, dict[str, str]] = state.get("members", {})

        session_pool = agent_ctx.host.session_pool
        lead_session_id = agent_ctx.session.session_id
        if session_pool is not None:
            for member_info in members.values():
                sid: str = member_info.get("session_id", "")
                if sid and sid != lead_session_id:
                    await session_pool.close_session(sid)

        # Clear member session IDs from metadata so the auto-cleanup
        # callback (scheduled in team_create) knows manual cleanup was
        # already performed and skips double-closing.
        agent_ctx.session.metadata["team_member_sessions"] = []

        team_state.cleanup(team_id)
        return ToolReturn(return_value="Team deleted")

    async def delete_blackboard(
        self,
        ctx: RunContext[Any],
        key: Annotated[str, Field(description="Blackboard key to delete")],
    ) -> ToolReturn:
        """Delete a key from the shared blackboard (lead-only).

        Returns:
            ``"Blackboard key '{key}' deleted"`` on success, or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use delete_blackboard")

        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        existing = team_state.read_blackboard(team_id, key)
        if existing is None:
            available = team_state.list_blackboard(team_id)
            keys_str = ", ".join(available) if available else "(empty)"
            return ToolReturn(return_value=f"Key '{key}' not found. Available keys: {keys_str}")

        team_state.delete_blackboard(team_id, key)
        return ToolReturn(return_value=f"Blackboard key '{key}' deleted")

    async def team_add_member(  # noqa: PLR0911, PLR0915
        self,
        ctx: RunContext[Any],
        name: Annotated[
            str,
            Field(
                description="Display name for the new member. If empty, "
                "falls back to the agent's display_name. Prefer semantic "
                "name reflecting the member's purpose."
            ),
        ],
        agent: Annotated[str, Field(description="Registered agent name to use as the member")],
        prompt: Annotated[
            str,
            Field(
                description="Optional initial prompt to send the member. If "
                "empty, the protocol template is used"
            ),
        ] = "",
        lifecycle: Annotated[
            str,
            Field(
                description='"persistent" (default) or "ephemeral". Ephemeral '
                "members are auto-closed when their run completes"
            ),
        ] = "persistent",
        notify: Annotated[
            str,
            Field(
                description="Optional notice describing why the new member was "
                "added or what they can help with, included in the "
                "auto-broadcast to existing members"
            ),
        ] = "",
        instructions: Annotated[
            str,
            Field(
                description="Optional per-member instructions injected into "
                "the member's system prompt"
            ),
        ] = "",
        skills: Annotated[
            list[str] | None,
            Field(
                description="Optional skill names or skill:// URIs (including "
                "reference paths) injected as instruction text into the "
                "member's system prompt"
            ),
        ] = None,
    ) -> ToolReturn:
        """Add a new member to an existing team (lead-only).

        Returns:
            Success message or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use team_add_member")

        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        # Check agent exists in registry.
        if not agent_ctx.agent_registry.exists(agent):
            return ToolReturn(return_value=f"Agent '{agent}' not found in registry")

        # Check agent is eligible.
        if agent not in self._config.member_eligible:
            return ToolReturn(return_value=f"Agent '{agent}' is not eligible")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        from wolfharness.capabilities.file_team_state import FileTeamState

        # Check name not already in team state members.
        state_path = team_state._state_path(team_id)
        if not state_path.exists():
            return ToolReturn(return_value="Team state not found")
        state: dict[str, Any] = FileTeamState._read_json(state_path)
        members: dict[str, dict[str, Any]] = state.get("members", {})
        if name in members:
            return ToolReturn(return_value=f"Member '{name}' already exists")

        # Bounds: max_members check (exclude lead from count).
        lead_member_name = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )
        non_lead_count = sum(1 for mname in members if mname != lead_member_name)
        if non_lead_count >= self._config.bounds.max_members:
            return ToolReturn(
                return_value=(
                    f"Team exceeds max_members "
                    f"({non_lead_count + 1} > {self._config.bounds.max_members})"
                )
            )

        session_pool = agent_ctx.host.session_pool
        if session_pool is None:
            return ToolReturn(return_value="SessionPool not available")

        lead_session_id: str = agent_ctx.session.session_id

        # Create child session for the new member.
        try:
            # Inject requested skills as instruction text into the member's
            # system prompt (pure prompt guidance — no tool/MCP assembly).
            member_instructions: str = instructions
            coerced_skills = self._coerce_skill_names(skills)
            if coerced_skills:
                skills_content = await self._format_member_skills_instructions(
                    ctx,
                    coerced_skills,
                    agent,
                )
                if skills_content:
                    member_instructions = f"{skills_content}\n\n{member_instructions}"
            member_session_id = await self._create_member_session(
                agent_ctx,
                agent,
                parent_session_id=lead_session_id,
                description=f"Team member: {name}",
                tool_call_id=ctx.tool_call_id,
                team_id=team_id,
                team_name=agent_ctx.session.metadata.get("team_name"),
                team_role="member",
                team_member_name=name,
                team_member_instructions=member_instructions,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolReturn(return_value=f"Failed to create member session: {exc}")

        # Propagate member display name to the agent instance so protocol
        # frontends (ACP, OpenCode) show the correct name.
        member_agent = session_pool.sessions._session_agents.get(member_session_id)
        if member_agent is not None:
            member_agent._display_name = name

        # Register member in team state.
        team_state.register_member(
            team_id,
            name,
            member_session_id,
            agent=agent,
        )

        # Send initial prompt to member (with existing member roster).
        from wolfharness.lifecycle.types import DeliveryMode

        team_name: str = state.get("team_name", "unknown")
        base_prompt = prompt or self._config.protocol_template.format(
            team_name=team_name,
            role="member",
            member_name=name,
        )
        # Append current member roster so the new member knows their teammates.
        existing_members: dict[str, dict[str, Any]] = state.get("members", {})
        work_summaries = self._build_member_work_summary(
            team_state,
            team_id,
            existing_members,
        )
        roster_lines = []
        for m_name, m_info in existing_members.items():
            m_agent = m_info.get("agent", m_name)
            role_label = "lead" if m_name == lead_member_name else "member"
            work = work_summaries.get(m_name, "No active work")
            roster_lines.append(f"  - `{m_name}` (agent=`{m_agent}`, role=`{role_label}`) — {work}")
        roster = "\n".join(roster_lines)
        initial_prompt = f"{base_prompt}\n\n## Team Members\n{roster}"
        if prompt:
            initial_prompt += (
                f"\n\n## Task\n{prompt}"
                "\n\nRemember to report progress regularly using "
                '`task_update(technical_note="...")`.'
            )
        await session_pool.send_message(
            member_session_id,
            initial_prompt,
            mode=DeliveryMode.QUEUE,
            source="accepted",
            meta={"from": self._agent_name, "team_id": team_id},
        )

        # Ephemeral lifecycle: schedule auto-close when run completes.
        if lifecycle == "ephemeral":
            base_dir = (
                agent_ctx.team_mode_config.effective_base_dir
                if agent_ctx.team_mode_config is not None
                else tempfile.gettempdir()
            )
            self._schedule_ephemeral_cleanup(
                session_pool,
                member_session_id,
                team_id,
                name,
                base_dir,
            )

        # Notify existing members about the new member.
        # Re-read state to get the updated members dict.
        updated_state: dict[str, Any] = FileTeamState._read_json(
            team_state._state_path(team_id),
        )
        updated_members: dict[str, dict[str, Any]] = updated_state.get(
            "members",
            {},
        )

        # Auto-broadcast to existing members (excluding lead and new member).
        if self._config.broadcast_on_create:
            roster_lines = []
            for m_name, m_info in updated_members.items():
                m_agent = m_info.get("agent", m_name)
                role_tag = " (lead)" if m_name == lead_member_name else ""
                roster_lines.append(f"  - `{m_name}` (`{m_agent}`){role_tag}")
            roster = "\n".join(roster_lines)
            notice_line = f"\n\nnote: {notify}" if notify else ""
            notice_text = (
                f"New member `{name}` (`{agent}`) joined the team."
                f"{notice_line}\n\ncurrent members:\n{roster}"
            )
            broadcast_msg = (
                f'<team-message from="{self._agent_name}" type="broadcast">'
                f"\n\n{notice_text}\n\n</team-message>"
            )
            for existing_name, existing_info in updated_members.items():
                if existing_name in (lead_member_name, name):
                    continue
                existing_sid: str = existing_info.get("session_id", "")
                if not existing_sid:
                    continue
                await session_pool.send_message(
                    existing_sid,
                    self._wrap_notice_content(broadcast_msg),
                    mode=self._notice_mode,
                    source="accepted",
                    meta={"from": self._agent_name, "team_id": team_id},
                )

        # Write to blackboard (non-fatal — audit trail only).
        import re

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        try:
            team_state.write_blackboard(
                team_id,
                f"member_update/{safe_name}",
                {"action": "added", "agent": agent, "lifecycle": lifecycle, "name": name},
                written_by=self._agent_name,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to write member_update to blackboard for '%s'",
                name,
            )

        # Append member_session_id to session metadata.
        team_member_sessions: list[str] = agent_ctx.session.metadata.get(
            "team_member_sessions",
            [],
        )
        team_member_sessions.append(member_session_id)
        agent_ctx.session.metadata["team_member_sessions"] = team_member_sessions

        return ToolReturn(return_value=f"Member '{name}' added to team (lifecycle={lifecycle})")

    async def shutdown_request(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        member_name: Annotated[str, Field(description="Name of the member to shut down")],
    ) -> ToolReturn:
        """Shut down (remove) a team member and release its resources (lead-only).

        Returns:
            Success message or error string.
        """
        agent_ctx = self._resolve_agent_context(ctx)
        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role != "lead":
            return ToolReturn(return_value="Only lead can use shutdown_request")

        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return ToolReturn(return_value="Not in a team session")

        # Cannot remove yourself.
        lead_member_name = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )
        if member_name == lead_member_name:
            return ToolReturn(return_value="Cannot shut down yourself")

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return ToolReturn(return_value="Not in a team session")

        from wolfharness.capabilities.file_team_state import FileTeamState

        member_sid = team_state.get_member_session_id(team_id, member_name)
        if member_sid is None:
            return ToolReturn(return_value=f"Member '{member_name}' not found")

        # Check for unfinished tasks before closing the session.
        unfinished = self._get_unfinished_tasks(team_state, team_id, member_name)

        session_pool = agent_ctx.host.session_pool
        if session_pool is not None:
            await session_pool.close_session(member_sid)

        # Remove from team state: read, delete member, write back.
        state_path = team_state._state_path(team_id)
        if state_path.exists():
            state: dict[str, Any] = FileTeamState._read_json(state_path)
            members_dict: dict[str, dict[str, Any]] = state.get("members", {})
            members_dict.pop(member_name, None)
            state["members"] = members_dict
            FileTeamState._atomic_write(state_path, state)

        # Remove from session metadata team_member_sessions.
        team_member_sessions: list[str] = agent_ctx.session.metadata.get(
            "team_member_sessions",
            [],
        )
        if member_sid in team_member_sessions:
            team_member_sessions.remove(member_sid)
            agent_ctx.session.metadata["team_member_sessions"] = team_member_sessions

        # Write to blackboard (non-fatal — audit trail only).
        import re

        safe_member_name = re.sub(r"[^a-zA-Z0-9_]", "_", member_name)
        try:
            team_state.write_blackboard(
                team_id,
                f"member_update/{safe_member_name}",
                {"action": "removed", "name": member_name},
                written_by=self._agent_name,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to write member_update to blackboard for '%s'",
                member_name,
            )

        if unfinished:
            task_list = ", ".join(
                f"'{t.get('subject', '?')}' (id={t.get('task_id', '?')})" for t in unfinished
            )
            return ToolReturn(
                return_value=(
                    f"Shutdown completed for {member_name}. "
                    f"Warning: member had {len(unfinished)} unfinished "
                    f"task(s): {task_list}. "
                    f"Please update task status or reassign to another member."
                ),
            )
        return ToolReturn(return_value=f"Shutdown completed for {member_name}")

    @staticmethod
    def _schedule_ephemeral_cleanup(
        session_pool: Any,
        member_session_id: str,
        team_id: str,
        member_name: str,
        base_dir: str,
    ) -> None:
        """Wait for ephemeral member run to complete, then remove from team state.

        Uses ``complete_event.wait()`` on the RunHandle for event-driven
        completion detection instead of polling ``current_run_id`` every
        5 seconds.  The state file is updated BEFORE closing the session
        so that ``team_status(watch=True)`` detects the change immediately
        even if ``close_session`` fails.

        Args:
            session_pool: The SessionPool managing the member session.
            member_session_id: Session ID of the ephemeral member.
            team_id: Team ID for state cleanup.
            member_name: Member name for state cleanup.
            base_dir: Base directory for FileTeamState.
        """
        from wolfharness.capabilities.file_team_state import FileTeamState

        async def _wait_and_cleanup() -> None:
            try:
                # Event-driven: wait for the member's run to complete.
                while True:
                    session = session_pool.sessions.get_session(member_session_id)
                    if session is None:
                        return  # Already closed
                    run_id = session.current_run_id
                    if run_id is None:
                        break  # Session idle — run already completed
                    run_handle = session_pool.get_run(run_id)
                    if run_handle is None:
                        # Run handle already cleaned up — session should
                        # be idle or starting a new chained run.
                        break
                    await run_handle.complete_event.wait()
                    # Check if a new chained run started (prompt_queue).
                    session = session_pool.sessions.get_session(member_session_id)
                    if session is None:
                        return
                    if session.current_run_id is not None and session.current_run_id != run_id:
                        continue  # New chained run — wait for it too
                    break

                # Run completed — update state file FIRST so that
                # team_status(watch=True) detects the mtime change
                # immediately, even if close_session fails below.
                team_state = FileTeamState(base_dir)
                state_path = team_state._state_path(team_id)
                if state_path.exists():
                    state = team_state._read_json(state_path)
                    members = state.get("members", {})
                    members.pop(member_name, None)
                    state["members"] = members
                    team_state._atomic_write(state_path, state)

                # Then close the session — suppress errors so a failing
                # close does not prevent the state file update above.
                with contextlib.suppress(Exception):
                    await session_pool.close_session(member_session_id)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception(
                    "Ephemeral cleanup failed for member %s (session %s)",
                    member_name,
                    member_session_id,
                )

        task = asyncio.create_task(_wait_and_cleanup())
        _cleanup_tasks.add(task)

        def _on_done(t: asyncio.Task[Any]) -> None:
            _cleanup_tasks.discard(t)

        task.add_done_callback(_on_done)

    # ------------------------------------------------------------------
    # AbstractCapability overrides
    # ------------------------------------------------------------------

    # Tool names that only lead agents may use.  Non-lead members never
    # see these tools — ``prepare_tools`` filters them out before the
    # model receives the tool list, so the LLM cannot attempt to call
    # them.  The runtime permission checks in each tool body remain as a
    # safety net.
    _LEAD_ONLY_TOOLS: frozenset[str] = frozenset(
        {
            "team_create",
            "team_delete",
            "delete_blackboard",
            "shutdown_request",
            "team_add_member",
            "task_create_batch",
        },
    )

    @override
    async def prepare_tools(
        self,
        ctx: RunContext[Any],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter and modify tool definitions based on the agent's team role.

        For non-lead members:
            - Lead-only tools (``team_create``, ``team_delete``,
              ``delete_blackboard``, ``shutdown_request``,
              ``team_add_member``) are removed
              entirely so the LLM never sees them.
            - ``send_message`` has its ``to`` parameter description
              updated to remove the broadcast (``"*"``) mention, and a
              ``pattern`` constraint is added to reject ``"*"`` at the
              schema level.

        For lead agents, all tool definitions are returned unchanged.

        Args:
            ctx: The PydanticAI run context (unused — role is read from
                ``self._session_metadata``).
            tool_defs: The full list of tool definitions for this step.

        Returns:
            Filtered/modified tool definitions.
        """
        # No session metadata = compile-time shared instance; no role
        # filtering to apply.
        if not self._session_metadata:
            return tool_defs

        role: str = self._session_metadata.get("team_role", "")
        if role == "lead":
            return tool_defs

        result: list[ToolDefinition] = []
        for td in tool_defs:
            if td.name in self._LEAD_ONLY_TOOLS:
                continue
            if td.name == "send_message":
                self._strip_broadcast_from_send_message(td)
            result.append(td)
        return result

    @override
    async def after_run(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Check for unfinished tasks after a member agent's run completes.

        If this is a team member (not lead) with ``in_progress`` tasks,
        routes a reminder message to the member's own session via the same
        delivery mechanism as ``send_message``.  Skipped when the session
        is being closed (shutdown path).  Limited to one reminder per
        session to avoid infinite loops.
        """
        try:
            agent_ctx = self._resolve_agent_context(ctx)
        except RuntimeError:
            return result

        team_id = self._get_team_id(agent_ctx)
        if team_id is None:
            return result

        role: str = agent_ctx.session.metadata.get("team_role", "")
        if role == "lead":
            return result

        # Skip if session is being closed (shutdown handles notification
        # via the shutdown_request tool return value instead).
        if agent_ctx.session.closing or agent_ctx.session.is_closing:
            return result

        team_state = self._get_team_state(agent_ctx)
        if team_state is None:
            return result

        member_name: str = agent_ctx.session.metadata.get(
            "team_member_name",
            self._agent_name,
        )
        unfinished = self._get_unfinished_tasks(team_state, team_id, member_name)
        if not unfinished:
            return result

        # Avoid infinite loops: max 1 reminder per session.
        reminder_count: int = agent_ctx.session.metadata.get("_task_reminder_count", 0)
        if reminder_count >= 1:
            return result

        task_lines = "\n".join(
            f"  - '{t.get('subject', '?')}' (id={t.get('task_id', '?')})" for t in unfinished
        )
        reminder_body = (
            f"You have {len(unfinished)} unfinished task(s) still "
            f"in_progress:\n{task_lines}\n\n"
            f"Please complete your work and update the task status using "
            f"task_update(task_id=..., status='completed') or "
            f"task_update(task_id=..., status='failed') if you encountered "
            f"issues."
        )
        msg_body = (
            f'<team-message from="system" type="task_reminder">\n\n'
            f"{reminder_body}\n\n</team-message>"
        )

        session_pool = agent_ctx.host.session_pool
        if session_pool is not None:
            from wolfharness.lifecycle.types import DeliveryMode

            # Use QUEUE mode: the run is ending, so STEER would be lost.
            await session_pool.send_message(
                agent_ctx.session.session_id,
                self._wrap_notice_content(msg_body),
                mode=DeliveryMode.QUEUE,
                source="accepted",
                meta={"from": "system", "team_id": team_id},
            )
            agent_ctx.session.metadata["_task_reminder_count"] = reminder_count + 1

        return result

    @staticmethod
    def _get_unfinished_tasks(
        team_state: FileTeamState,
        team_id: str,
        member_name: str,
    ) -> list[dict[str, Any]]:
        """Return tasks owned by *member_name* that are still in_progress."""
        all_tasks = team_state.list_tasks(team_id)
        return [
            t
            for t in all_tasks
            if t.get("owner") == member_name and t.get("status") == "in_progress"
        ]

    @staticmethod
    def _strip_broadcast_from_send_message(tool_def: ToolDefinition) -> None:
        """Remove broadcast (``to="*"``) from the send_message tool schema.

        Mutates ``tool_def`` in place:
            - Updates the ``to`` parameter description to omit the
              broadcast mention.
            - Adds a ``pattern`` constraint that rejects ``"*"``.

        Args:
            tool_def: The ``send_message`` ToolDefinition to modify.
        """
        schema = tool_def.parameters_json_schema
        props = schema.get("properties", {})
        to_prop = props.get("to")
        if to_prop is not None and isinstance(to_prop, dict):
            to_prop["description"] = "Recipient member name."
            to_prop["pattern"] = r"^[^*]+$"

    @override
    def get_instructions(self) -> str | None:
        """Render the team protocol template using session metadata.

        Returns ``None`` when:
            - ``config.enabled`` is ``False``, OR
            - ``session_metadata`` is empty/``None``

        When both conditions are met, renders ``config.protocol_template``
        via ``str.format()`` with ``team_name``, ``role``, and ``member_name``
        extracted from session metadata (with sensible defaults).
        """
        if not self._config.enabled or not self._session_metadata:
            return None
        role: str = self._session_metadata.get("team_role", "unknown")
        base = self._config.protocol_template.format(
            team_name=self._session_metadata.get("team_name", "unknown"),
            role=role,
            member_name=self._session_metadata.get(
                "team_member_name",
                self._agent_name,
            ),
        )

        # Role-specific capabilities section.
        if role == "lead":
            base += (
                "\n\n## Your Capabilities (Lead)\n\n"
                "- You can broadcast to all members via `send_message` with "
                '`to="*"`.\n'
                "- You can create and delete teams, delete blackboard keys, "
                "and shut down members.\n"
            )
        else:
            base += (
                "\n\n## Your Capabilities (Member)\n\n"
                "- Use `send_message` to send messages to individual members "
                "by name.\n"
                '- Broadcast (`to="*"`) is not available to you — send '
                "individual messages to each member instead.\n"
            )

        # Per-member instructions injection (## Your Assignment).
        instructions_text: str = self._session_metadata.get(
            "team_member_instructions",
            "",
        )
        if instructions_text:
            base += f"\n\n## Your Assignment\n\n{instructions_text}"

        # Append eligible agent names + descriptions so the LLM knows
        # which agents can be used as team members in team_create.
        eligible = self._config.member_eligible
        if eligible:
            base += (
                "\n\n## Eligible Agents\n\n"
                "The following agents can be used as team members in `team_create`:\n"
            )
            for name in eligible:
                desc = self._agent_descriptions.get(name)
                if desc:
                    base += f"- `{name}`: {desc}\n"
                else:
                    base += f"- `{name}`\n"
        return base

    @override
    async def get_tools(self) -> Sequence[Tool[Any]]:
        """Return the list of team communication tools.

        Returns an empty list when ``config.enabled`` is ``False``.
        """
        if not self._config.enabled:
            return []
        return self._tools
