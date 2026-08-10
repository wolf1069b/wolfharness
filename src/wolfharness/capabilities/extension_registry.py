"""ExtensionRegistry — unified capability registry with 4-level scope storage.

Replaces fragmented infrastructure (SkillURIResolver._providers,
AggregatedResourceSource) with a single registry
that supports pool, agent, session, and turn-level capability scoping.

Scope hierarchy (outer → inner):
    POOL → AGENT → SESSION → TURN

Pool-level capabilities are visible to all agents and sessions.
Agent-level capabilities are visible to a specific named agent across
all sessions (agent configs are compiled once and reused). Session-level
capabilities are visible only within their session (e.g. MCP connections).
Turn-level capabilities are visible only for the duration of one turn
and are guarded by an ``asyncio.Lock`` for concurrent access.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
import warnings

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai.capabilities import AbstractCapability

    from wolfharness.capabilities.change_event import ChangeEvent
    from wolfharness.capabilities.resource_protocols import (
        ChangeObservable,
        CommandResource,
        ResourceAccess,
        ResourceTemplateAccess,
        SkillResource,
        ToolAccess,
    )
    from wolfharness.skills.skill import Skill


logger = get_logger(__name__)

DEFAULT_MAX_COMPOSITION_DEPTH = 3
"""Default maximum composition depth (root-inclusive)."""


class ScopeLevel(Enum):
    """Capability scope level.

    Attributes:
        POOL: Visible to all agents, sessions, and turns.
        AGENT: Visible to a specific named agent across all sessions.
        SESSION: Visible only within a specific session.
        TURN: Visible only for the duration of one turn.
    """

    POOL = auto()
    AGENT = auto()
    SESSION = auto()
    TURN = auto()


@dataclass(frozen=True, slots=True)
class Scope:
    """Immutable scope identifying where a capability is visible.

    Attributes:
        level: The scope level (POOL/AGENT/SESSION/TURN).
        agent_name: Agent name (required for AGENT scope).
        session_id: Session identifier (required for SESSION/TURN).
        turn_id: Turn identifier (required for TURN).
    """

    level: ScopeLevel
    agent_name: str = ""
    session_id: str = ""
    turn_id: str = ""


class CircularCompositionError(Exception):
    """Raised when a capability composition cycle is detected.

    Args:
        message: Description of the cycle.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ExtensionRegistry:
    """Registry for capabilities with 4-level scope storage.

    Capabilities are registered at a specific scope level and are
    visible to all inner scopes. The scope hierarchy is::

        POOL > AGENT > SESSION > TURN

    - **POOL**: visible to all agents and sessions (e.g. skills, subagent).
    - **AGENT**: visible to a specific named agent across all sessions
      (e.g. config-derived caps compiled by ``AgentFactory.compile()``).
    - **SESSION**: visible within a specific session (e.g. MCP connections).
    - **TURN**: visible for one turn (e.g. ModalityFilter with resolved
      model caps).

    A SESSION scope query returns ``POOL + AGENT + SESSION`` capabilities,
    providing a complete view for a specific agent in a specific session.

    The registry provides typed query methods that filter by Resource
    Protocol type, URI resolution by scheme, and change stream merging.

    Turn-level registration is guarded by an ``asyncio.Lock`` to
    prevent concurrent modification. Pool, agent, and session-level
    dicts do not require locking (mutated only at startup/shutdown).

    Composition cycle detection and depth limiting are performed at
    ``add_child()`` registration time.
    """

    def __init__(
        self,
        max_composition_depth: int = DEFAULT_MAX_COMPOSITION_DEPTH,
    ) -> None:
        """Initialize the registry with empty scope storage.

        Args:
            max_composition_depth: Maximum composition depth (root-inclusive).
                When depth exceeds this limit, a warning is logged but
                registration is NOT blocked. Default: 3.
        """
        self._max_composition_depth = max_composition_depth

        # 4-level scope storage: POOL > AGENT > SESSION > TURN
        self._pool: list[AbstractCapability[Any]] = []
        self._agent: dict[str, list[AbstractCapability[Any]]] = {}
        # agent: agent_name → caps (pool lifetime, reused across sessions)
        self._session: dict[str, list[AbstractCapability[Any]]] = {}
        # session: session_id → caps (session lifetime)
        self._turn: dict[str, dict[str, dict[str, list[AbstractCapability[Any]]]]] = {}
        # turn: session_id → agent_name → turn_id → caps (turn lifetime)

        # Lock for turn-level mutations
        self._turn_lock = asyncio.Lock()

        # Child tracking for cycle detection and depth limiting
        # Maps id(capability) → list of child capabilities
        self._children: dict[int, list[AbstractCapability[Any]]] = {}

        # Parent tracking for ancestor chain walking
        # Maps id(capability) → parent capability (or None for roots)
        self._parents: dict[int, AbstractCapability[Any] | None] = {}

        # Background tasks for change stream merging
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        capability: AbstractCapability[Any],
        scope: Scope,
    ) -> None:
        """Register a capability at the given scope.

        For turn-level scope, this method acquires the turn lock.
        For other scopes, registration is synchronous (no lock).

        Args:
            capability: The capability to register.
            scope: The scope at which to register it.
        """
        match scope.level:
            case ScopeLevel.POOL:
                self._pool.append(capability)
            case ScopeLevel.AGENT:
                self._agent.setdefault(scope.agent_name, []).append(capability)
            case ScopeLevel.SESSION:
                self._session.setdefault(scope.session_id, []).append(capability)
            case ScopeLevel.TURN:
                # Turn-level registration requires the lock
                # But register() is sync — we need to use the async variant
                # for turn-level. Fall through to a sync approach that's safe
                # because the lock is only needed for concurrent async access.
                self._turn.setdefault(scope.session_id, {}).setdefault(
                    scope.agent_name, {}
                ).setdefault(scope.turn_id, []).append(capability)

    async def register_async(
        self,
        capability: AbstractCapability[Any],
        scope: Scope,
    ) -> None:
        """Register a capability at the given scope (async variant).

        For turn-level scope, this method acquires the turn lock to
        prevent concurrent modification. For other scopes, behaves
        identically to ``register()``.

        Args:
            capability: The capability to register.
            scope: The scope at which to register it.
        """
        if scope.level == ScopeLevel.TURN:
            async with self._turn_lock:
                self._turn.setdefault(scope.session_id, {}).setdefault(
                    scope.agent_name, {}
                ).setdefault(scope.turn_id, []).append(capability)
        else:
            self.register(capability, scope)

    def unregister(  # noqa: PLR0911
        self,
        capability: AbstractCapability[Any],
        scope: Scope,
    ) -> bool:
        """Unregister a capability from the given scope.

        For turn-level scope, use ``unregister_async()`` instead to
        ensure proper locking.

        Args:
            capability: The capability to unregister.
            scope: The scope from which to unregister it.

        Returns:
            True if the capability was found and removed, False otherwise.
        """
        match scope.level:
            case ScopeLevel.POOL:
                if capability in self._pool:
                    self._pool.remove(capability)
                    return True
                return False
            case ScopeLevel.AGENT:
                caps = self._agent.get(scope.agent_name, [])
                if capability in caps:
                    caps.remove(capability)
                    return True
                return False
            case ScopeLevel.SESSION:
                caps = self._session.get(scope.session_id, [])
                if capability in caps:
                    caps.remove(capability)
                    return True
                return False
            case ScopeLevel.TURN:
                session_map = self._turn.get(scope.session_id, {})
                agent_map = session_map.get(scope.agent_name, {})
                caps = agent_map.get(scope.turn_id, [])
                if capability in caps:
                    caps.remove(capability)
                    return True
                return False
        return False

    async def unregister_async(
        self,
        capability: AbstractCapability[Any],
        scope: Scope,
    ) -> bool:
        """Unregister a capability from the given scope (async variant).

        For turn-level scope, acquires the turn lock.

        Args:
            capability: The capability to unregister.
            scope: The scope from which to unregister it.

        Returns:
            True if the capability was found and removed, False otherwise.
        """
        if scope.level == ScopeLevel.TURN:
            async with self._turn_lock:
                session_map = self._turn.get(scope.session_id, {})
                agent_map = session_map.get(scope.agent_name, {})
                caps = agent_map.get(scope.turn_id, [])
                if capability in caps:
                    caps.remove(capability)
                    return True
                return False
        return self.unregister(capability, scope)

    def clear_turn(self, session_id: str, agent_name: str, turn_id: str) -> None:
        """Clear all turn-level capabilities for a specific turn.

        Args:
            session_id: Session identifier.
            agent_name: Agent name.
            turn_id: Turn identifier.
        """
        session_map = self._turn.get(session_id, {})
        agent_map = session_map.get(agent_name, {})
        agent_map.pop(turn_id, None)
        if not agent_map:
            session_map.pop(agent_name, None)
        if not session_map:
            self._turn.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """Clear all SESSION and TURN level entries for the given session.

        AGENT and POOL level caps are NOT affected (they outlive sessions).

        Args:
            session_id: Session identifier to clean up.
        """
        self._session.pop(session_id, None)
        self._turn.pop(session_id, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_visible_capabilities(
        self,
        scope: Scope,
    ) -> list[AbstractCapability[Any]]:
        """Get all capabilities visible at the given scope.

        Walks the scope hierarchy from pool → agent → session → turn,
        collecting all capabilities at each level.

        With the hierarchy ``POOL > AGENT > SESSION > TURN``, a SESSION
        scope query returns ``POOL + AGENT + SESSION`` capabilities —
        the complete view for a specific agent in a specific session.

        Args:
            scope: The scope to query.

        Returns:
            List of visible capabilities (pool + matching agent +
            matching session + matching turn).
        """
        result: list[AbstractCapability[Any]] = list(self._pool)

        if scope.level.value >= ScopeLevel.AGENT.value:
            result.extend(self._agent.get(scope.agent_name, []))

        if scope.level.value >= ScopeLevel.SESSION.value:
            result.extend(self._session.get(scope.session_id, []))

        if scope.level.value >= ScopeLevel.TURN.value:
            session_map = self._turn.get(scope.session_id, {})
            agent_map = session_map.get(scope.agent_name, {})
            result.extend(agent_map.get(scope.turn_id, []))

        return result

    def get_skill_resources(
        self,
        scope: Scope,
    ) -> list[SkillResource]:
        """Get visible capabilities implementing ``SkillResource``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``SkillResource``.
        """
        from wolfharness.capabilities.resource_protocols import SkillResource

        return [
            cap for cap in self.get_visible_capabilities(scope) if isinstance(cap, SkillResource)
        ]

    def get_resource_access(
        self,
        scope: Scope,
    ) -> list[ResourceAccess]:
        """Get visible capabilities implementing ``ResourceAccess``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``ResourceAccess``.
        """
        from wolfharness.capabilities.resource_protocols import ResourceAccess

        return [
            cap for cap in self.get_visible_capabilities(scope) if isinstance(cap, ResourceAccess)
        ]

    def get_tool_access(
        self,
        scope: Scope,
    ) -> list[ToolAccess]:
        """Get visible capabilities implementing ``ToolAccess``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``ToolAccess``.
        """
        from wolfharness.capabilities.resource_protocols import ToolAccess

        return [cap for cap in self.get_visible_capabilities(scope) if isinstance(cap, ToolAccess)]

    def get_resource_template_access(
        self,
        scope: Scope,
    ) -> list[ResourceTemplateAccess]:
        """Get visible capabilities implementing ``ResourceTemplateAccess``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``ResourceTemplateAccess``.
        """
        from wolfharness.capabilities.resource_protocols import ResourceTemplateAccess

        return [
            cap
            for cap in self.get_visible_capabilities(scope)
            if isinstance(cap, ResourceTemplateAccess)
        ]

    def get_mcp_resources(
        self,
        scope: Scope,
    ) -> list[ResourceAccess]:
        """Get visible capabilities implementing ``McpResource``.

        .. deprecated::
            Use ``get_resource_access()`` and/or ``get_tool_access()`` instead.
            This method emits a ``DeprecationWarning`` and delegates to
            ``get_resource_access()``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``ResourceAccess``.
        """
        warnings.warn(
            "get_mcp_resources() is deprecated. Use get_resource_access() "
            "and/or get_tool_access() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_resource_access(scope)

    def get_command_resources(
        self,
        scope: Scope,
    ) -> list[CommandResource]:
        """Get visible capabilities implementing ``CommandResource``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``CommandResource``.
        """
        from wolfharness.capabilities.resource_protocols import CommandResource

        return [
            cap for cap in self.get_visible_capabilities(scope) if isinstance(cap, CommandResource)
        ]

    def get_observable_capabilities(
        self,
        scope: Scope,
    ) -> list[ChangeObservable]:
        """Get visible capabilities implementing ``ChangeObservable``.

        Args:
            scope: The scope to query.

        Returns:
            List of capabilities implementing ``ChangeObservable``.
        """
        from wolfharness.capabilities.resource_protocols import ChangeObservable

        return [
            cap for cap in self.get_visible_capabilities(scope) if isinstance(cap, ChangeObservable)
        ]

    # ------------------------------------------------------------------
    # URI Resolution
    # ------------------------------------------------------------------

    async def resolve_uri(
        self,
        uri: str,
        scope: Scope,
    ) -> Skill | str | bytes | None:
        """Resolve a URI by routing based on scheme.

        ``skill://`` URIs return a ``Skill`` object (with metadata and path).
        ``mcp://`` URIs and unknown schemes return ``str | bytes | None``.

        Uses ``skill_exists()`` / ``resource_exists()`` for a
        cheap-check-first pattern before reading content.

        Args:
            uri: The URI to resolve.
            scope: The scope to query for capabilities.

        Returns:
            The resolved content as string or bytes, or ``None`` if
            the URI cannot be resolved.
        """
        if uri.startswith("skill://"):
            # Flat URI (D9): parse via ResolvedSkillURI for consistency.
            from wolfharness.skills.skill import Skill
            from wolfharness.skills.uri_resolver import ResolvedSkillURI, _name_alternatives

            resolved = ResolvedSkillURI.parse(uri)
            skill_name = resolved.skill_name

            for cap in self.get_skill_resources(scope):
                try:
                    # Try to find the SkillEntry for metadata and real path.
                    entries = await cap.list_skills()
                    entry = next((e for e in entries if e.name == skill_name), None)
                    if entry is None:
                        # Fuzzy match: try underscore↔hyphen alternatives.
                        for alt_name in _name_alternatives(skill_name):
                            entry = next((e for e in entries if e.name == alt_name), None)
                            if entry is not None:
                                break
                    if entry is None:
                        continue
                    content = await cap.read_skill(entry.name)
                    if content is None:
                        continue
                    return Skill(
                        name=entry.name,
                        description=entry.description or f"Skill {entry.name}",
                        skill_path=entry.skill_path or PurePosixPath(uri),
                        instructions=content,
                    )
                except Exception:
                    logger.warning(
                        "Failed to resolve skill URI %r via %s",
                        uri,
                        type(cap).__name__,
                        exc_info=True,
                    )
            return None

        if uri.startswith("mcp://"):
            from wolfharness.capabilities.resource_protocols import TextResourceContent

            for resource_cap in self.get_resource_access(scope):
                try:
                    contents = await resource_cap.read_resource(uri)
                    if contents is None:
                        continue
                    for c in contents:
                        if isinstance(c, TextResourceContent):
                            return c.text
                    # Only BlobResourceContent found — cannot return as string
                except Exception:
                    logger.warning(
                        "Failed to resolve MCP URI %r via %s",
                        uri,
                        type(resource_cap).__name__,
                        exc_info=True,
                    )
            return None

        # Unknown scheme
        return None

    # ------------------------------------------------------------------
    # Change Stream Merging
    # ------------------------------------------------------------------

    def merge_change_streams(
        self,
        scope: Scope,
    ) -> AsyncIterator[ChangeEvent] | None:
        """Merge ``on_change()`` streams from all visible observables.

        Uses a sentinel-based merge pattern: each source stream is
        consumed in a separate task. When a source completes (returns
        ``None`` or raises), a sentinel is pushed. The merge completes
        when all sources have sent sentinels.

        Exceptions in individual streams are logged via
        ``logger.warning()`` and not propagated.

        Args:
            scope: The scope to query for observables.

        Returns:
            An ``AsyncIterator[ChangeEvent]`` if any observables are
            visible, ``None`` if no observables are found.
        """
        observables = self.get_observable_capabilities(scope)
        if not observables:
            return None

        # Collect active streams
        streams: list[AsyncIterator[ChangeEvent]] = []
        for cap in observables:
            stream = cap.on_change()
            if stream is not None:
                streams.append(stream)

        if not streams:
            return None

        return self._merge_streams(streams)

    async def _merge_streams(
        self,
        streams: list[AsyncIterator[ChangeEvent]],
    ) -> AsyncIterator[ChangeEvent]:
        """Merge multiple change event streams into one.

        Args:
            streams: List of async iterators to merge.

        Yields:
            ``ChangeEvent`` instances from all streams.
        """
        queue: asyncio.Queue[ChangeEvent | None] = asyncio.Queue()
        tasks: list[asyncio.Task[None]] = []
        num_streams = len(streams)

        async def _consume(stream: AsyncIterator[ChangeEvent]) -> None:
            try:
                async for event in stream:
                    await queue.put(event)
            except Exception:
                logger.warning(
                    "Change stream consumer encountered an error",
                    exc_info=True,
                )
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(_consume(s)) for s in streams]
        for task in tasks:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        finished_count = 0
        try:
            while finished_count < num_streams:
                event = await queue.get()
                if event is None:
                    finished_count += 1
                    continue
                yield event
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Composition: Cycle Detection & Depth Limit
    # ------------------------------------------------------------------

    def add_child(
        self,
        parent: AbstractCapability[Any],
        child: AbstractCapability[Any],
    ) -> None:
        """Add a child capability to a parent, with cycle detection.

        Checks the ancestor chain of ``parent`` for ``child``. If
        ``child`` already appears in its own ancestor chain, a
        ``CircularCompositionError`` is raised.

        Also checks composition depth. If depth exceeds
        ``max_composition_depth``, a warning is logged but registration
        is NOT blocked.

        Args:
            parent: The parent capability.
            child: The child capability to add.

        Raises:
            CircularCompositionError: If adding ``child`` would create
                a cycle in the composition graph.
        """
        # Cycle detection: walk ancestor chain of parent
        current: AbstractCapability[Any] | None = parent
        visited: set[int] = set()
        depth = 1  # root-inclusive

        while current is not None:
            current_id = id(current)
            if current_id in visited:
                # Cycle in existing chain (shouldn't happen, but guard)
                parent_name = type(parent).__name__
                msg = f"Circular composition detected: cycle in ancestor chain of {parent_name}"
                raise CircularCompositionError(msg)
            visited.add(current_id)

            if current_id == id(child):
                msg = (
                    f"Circular composition detected: "
                    f"{type(child).__name__} already appears in the "
                    f"ancestor chain of {type(parent).__name__}"
                )
                raise CircularCompositionError(msg)

            depth += 1
            current = self._parents.get(current_id)

        # Depth limit check (warning, not block)
        if depth > self._max_composition_depth:
            logger.warning(
                "Composition depth %d exceeds limit %d for %s → %s",
                depth,
                self._max_composition_depth,
                type(parent).__name__,
                type(child).__name__,
            )

        # Record the relationship
        self._children.setdefault(id(parent), []).append(child)
        self._parents[id(child)] = parent

    def get_children(
        self,
        capability: AbstractCapability[Any],
    ) -> list[AbstractCapability[Any]]:
        """Get the children of a capability.

        Args:
            capability: The capability to query.

        Returns:
            List of child capabilities, or empty list if none.
        """
        return list(self._children.get(id(capability), []))

    def get_depth(
        self,
        capability: AbstractCapability[Any],
    ) -> int:
        """Get the composition depth of a capability.

        Root capabilities have depth 1 (root-inclusive).

        Args:
            capability: The capability to query.

        Returns:
            The depth (1 for roots, 2 for children, etc.).
        """
        depth = 1
        current: AbstractCapability[Any] | None = self._parents.get(id(capability))
        visited: set[int] = {id(capability)}
        while current is not None:
            current_id = id(current)
            if current_id in visited:
                break
            visited.add(current_id)
            depth += 1
            current = self._parents.get(current_id)
        return depth
