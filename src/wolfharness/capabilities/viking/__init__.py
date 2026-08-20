"""VikingCapability — Viking knowledge graph integration for AgentPool.

Provides 15 tools for interacting with a Viking knowledge graph server,
organized into three categories:

- **Retrieve** (7 tools): search, find, recall, grep, glob, ls, read
- **Write** (6 tools): remember, write, edit, mkdir, add_resource, forget
- **Graph** (2 tools): link, set_tags

The capability also implements the ``SkillResource`` protocol, enabling
remote skill discovery and reading from the Viking server.

Configuration is via ``VikingCapabilityConfig`` in
``wolfharness_config.capabilities``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import logfire
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from wolfharness.capabilities.viking.identity import VikingIdentity, _try_decode_api_key
from wolfharness.capabilities.viking.ingest import (
    _MEMORY_INTENT_TEMPLATE,
    _extract_conversation_pairs,
    _ingest_conversation,
    _sanitize_message,
    format_memory_diff_summary,
    read_memory_diff,
)
from wolfharness.log import get_logger

logger = get_logger(__name__)

# Maximum consecutive failed remember drains before pending reasons are
# dropped. Bounds the marker accumulation on a persistently failing server.
_REMEMBER_MAX_RETRIES = 3

# Time budget for the background memory-diff notification task.
_REMEMBER_NOTIFY_TIMEOUT = 30.0


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from types import TracebackType
    from typing import Self

    from pydantic_ai import RunContext
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.models import ModelRequestContext

    from wolfharness.capabilities.change_event import ChangeEvent
    from wolfharness.capabilities.resource_protocols import (
        BlobResourceContent,
        ResourceEntry,
        SkillEntry,
        TextResourceContent,
    )
    from wolfharness_config.model_capabilities import ModelCapabilities


@dataclass
class VikingCapability(AbstractCapability[Any]):
    """Capability for interacting with a Viking knowledge graph.

    Provides 15 tools across retrieve, write, and graph categories.
    The SDK client is lazily initialized in ``__aenter__`` and shared
    across per-run copies via ``for_run()``.

    Attributes:
        mode: Tool exposure mode — ``"retrieve"`` (7 tools), ``"write"``
            (6 tools), ``"graph"`` (2 tools), or ``"all"`` (15 tools).
        url: Viking server URL. If ``None``, SDK resolves from
            ``OPENVIKING_URL`` env var or ``~/.openviking/ovcli.conf``.
        api_key: Viking API key. If ``None``, SDK resolves from env vars.
        account: Viking account ID. If ``None``, SDK resolves from env vars.
        user: Viking user ID. If ``None``, SDK resolves from env vars.
        timeout: Request timeout in seconds. If ``None``, SDK uses 60s.
        skills_uri: Override for skills URI.
        resources_uri: Override for resources URI.
        multimodal_bridge: Enable multimodal bridge (not yet implemented).
        uploads_uri: Override for uploads URI.
        public_download_base_url: Base URL for public download links.
    """

    mode: Literal["retrieve", "write", "graph", "all"] = "all"
    url: str | None = None
    api_key: str | None = None
    account: str | None = None
    user: str | None = None
    timeout: float | None = None
    skills_uri: str | None = None
    resources_uri: str | None = None
    sessions_uri: str | None = None
    """Override for sessions URI. Default: ``viking://user/{user}/sessions/``.
    When set, ``list_resources()`` includes files from this URI tree in
    addition to ``resources_uri``."""
    multimodal_bridge: bool = False
    """Enable multimodal bridge — auto-upload binary content to Viking
    before sending to the model."""
    support_vision: bool | None = None
    """Result of ``viking_read`` for image URIs.

    Tri-state control over how image resources are returned to the model:

    - ``True`` — return image bytes (``BinaryImage``) regardless of model.
    - ``False`` — return a text URI description, never image bytes.
    - ``None`` (default) — auto-detect from ``model_capabilities.image_input``;
      treated as text-only when capabilities are unknown (not injected or
      field is ``None``).

    Note: unlike ``ModalityFilterCapability._is_modality_supported``, which
    treats ``capabilities=None`` as pass-through, this capability treats an
    unset/model ``None`` capability as text-only (safe degradation) — it is
    the *producer* of image content and must not emit ``BinaryImage`` it
    cannot guarantee the model accepts.
    """
    uploads_uri: str | None = None
    public_download_base_url: str | None = None
    enable_link: bool = False
    """Enable the ``viking_link`` tool. Requires backend support for
    the graph link API. Disabled by default since not all Viking
    deployments support linking."""
    enable_memory: bool = False
    """Enable ``viking_remember`` and ``viking_recall`` tools. Requires
    backend support for session-based memory. Disabled by default
    since not all Viking deployments support memory sessions."""
    resource_file_extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".html",
    )
    """File extensions to include in ``list_resources()``. Files with
    extensions not in this set are skipped. Set to an empty tuple to
    include all files regardless of extension."""
    resource_read_level: Literal["abstract", "overview", "read"] = "overview"
    """Default content level for ``read_resource()`` (ResourceAccess Protocol).
    ``"abstract"`` (L0, ~100 tokens), ``"overview"`` (L1, ~2k tokens, default),
    or ``"read"`` (L2, full content). When ``read_resource()`` is called (e.g.
    via @ mention in OpenCode), this controls how much content is returned.
    Falls back to L2 if the requested level is unavailable."""
    model_capabilities: ModelCapabilities | None = None
    """Resolved model capabilities for multimodal bridge. Set by the
    agent factory after capability construction."""
    auto_resolve_identity: bool = True
    """When ``True`` (default), resolve ``account_id`` and ``user_id``
    automatically from the API key or ``/health`` endpoint after client
    initialization. Set to ``False`` to disable dynamic resolution."""
    memories_uri: str | None = None
    """Override for memories URI. Default: ``viking://user/{user_id}/memories/``."""
    actor_peer_id: str | None = None
    """Explicit actor peer ID for multi-agent isolation. When ``None``
    (default), the Viking server uses ``user_id`` for isolation. When set,
    passed to the SDK client for all requests."""
    auto_recall_enabled: bool = False
    """When ``True``, perform semantic recall before each model request.
    Results are injected as an ``<openviking-recall>`` XML block."""
    auto_recall_method: Literal["search", "find"] = "search"
    """Recall method: ``"search"`` (session-aware) or ``"find"`` (faster)."""
    auto_recall_max_tokens: int = 2000
    """Maximum token budget for injected recall block content."""
    auto_recall_limit: int = 10
    """Maximum number of results per recall request."""
    auto_recall_min_score: float = 0.3
    """Minimum composite score for including a recall hit."""
    auto_recall_lexical_boost: float = 0.1
    """Score boost per overlapping word between query and content."""
    auto_recall_category_boost: float = 0.05
    """Score boost for hits with ``context_type="memory"``."""
    auto_recall_context_types: list[str] = field(default_factory=lambda: ["memory", "resource"])
    """Context types to include in recall results."""
    enable_forget: bool = False
    """Enable the ``viking_forget`` tool. This is a destructive operation
    that removes documents from the Viking knowledge graph. Disabled by
    default — independent from ``enable_memory``."""
    uri_guard_enabled: bool = False
    """When ``True``, block file-access tools (``read``, ``bash``, ``grep``,
    ``glob``) from accessing ``viking://`` URIs in their arguments."""
    uri_guard_protected_tools: list[str] = field(
        default_factory=lambda: ["read", "bash", "grep", "glob"]
    )
    """Tool names protected by the URI guard. When ``uri_guard_enabled`` is
    ``True``, these tools are blocked from accessing ``viking://`` URIs."""
    allowed_uri_prefixes: list[str] = field(default_factory=list)
    """URI prefix allowlist for the ``viking://resources/`` namespace only.

    When non-empty, knowledge-base access (all ``viking_*`` tools + the
    @-mention flow) rejects ``viking://resources/...`` URIs outside the
    listed prefixes. All other namespaces — ``viking://user/...``
    (the agent's own memories, sessions, skills, and other users'
    namespaces), ``viking://skills/``, etc. — are always allowed and
    governed by their own feature flags. Skills discovery
    (``list_skills``/``read_skill``/``skill_exists``) only lists the
    skills URI, so it is unaffected by this allowlist.
    Empty list (default) means unrestricted — backward compatible."""
    profile_enabled: bool = False
    """Enable first-turn profile injection from Viking memories. When True,
    the capability queries Viking for memory search results on the first
    turn and injects them as an ``<openviking-profile>`` XML block."""
    profile_max_tokens: int = 1000
    """Maximum token budget for the injected profile block. Content is
    truncated if it exceeds this budget (chars-to-tokens 4:1 heuristic)."""
    profile_limit: int = 5
    """Maximum number of memory hits to retrieve for the profile block."""
    profile_first_turn_only: bool = True
    """When True (default), profile injection runs only on the first turn
    of a session. When False, injection runs on every ``before_model_request``
    call (not recommended — expensive and static)."""
    enabled_tools: list[str] | None = None
    """If set, only these tools are exposed (whitelist). Mutually exclusive
    with ``disabled_tools``."""
    disabled_tools: list[str] | None = None
    """Tools to exclude from the exposed set (blacklist). Mutually exclusive
    with ``enabled_tools``. For example, disable a slow semantic-search
    backend while keeping deterministic tools:
    ``["viking_search", "viking_find"]``."""
    compaction_enabled: bool = False
    """When True, archive old conversation messages to Viking before
    context overflow. Disabled by default."""
    compaction_threshold: int = 100_000
    """Estimated token count above which compaction is triggered."""
    compaction_keep_recent_turns: int = 5
    """Number of recent turns to keep when compacting."""
    compaction_expand_tool: bool = True
    """When True (and compaction_enabled), expose ``viking_expand`` tool."""
    auto_ingest_enabled: bool = False
    """Enable automatic conversation ingestion. When ``True``, the
    capability ingests the previous turn's conversation at the start
    of the next ``before_model_request`` call."""
    auto_ingest_mode: Literal["async", "sync"] = "async"
    """Ingestion mode — ``"async"`` (fire-and-forget) or ``"sync"``."""
    auto_ingest_sanitize: bool = True
    """Strip injected Viking XML blocks from messages before ingestion."""
    auto_ingest_source_type: str = "wolfharness"
    """Source type metadata for ingested sessions."""
    auto_ingest_keep_recent_turns: int = 0
    """Number of recent turns to retain after commit. 0 = no retention."""
    remember_notify: bool = True
    """When ``True`` (default), notify the session of an ingested memory's
    captured URIs (added/updated/deleted) via the session steer channel
    once the background extraction completes. Set to ``False`` to silence
    the notification — memories are still captured either way."""
    _client: Any = field(default=None, repr=False)
    _owns_client: bool = field(default=True, repr=False)
    _identity: VikingIdentity | None = field(default=None, repr=False)
    """Cached resolved identity. Set by ``_resolve_identity()`` after
    client initialization. Shared across per-run copies via ``for_run()``."""
    _profile_injected: bool = field(default=False, repr=False)
    """Flag tracking whether profile injection has already run for this
    session. Reset to ``False`` in ``for_run()`` so each run starts fresh."""
    _last_ingested_idx: int = field(default=0, repr=False)
    """Cursor tracking the last message index ingested to Viking.
    Reset to 0 on each per-run copy via ``for_run()``."""
    _remember_pending: list[str] = field(default_factory=list, repr=False)
    """Reasons queued by ``viking_remember`` calls for deferred capture.
    Drained at the next ``before_model_request`` boundary (or flushed in
    ``after_run``). Cleared only on a successful commit — a failed drain
    retains them so the retry keeps its ``<memory-intent>`` markers."""
    _remember_drain_failures: int = field(default=0, repr=False)
    """Consecutive failed remember drains in this run. When it reaches
    ``_REMEMBER_MAX_RETRIES``, pending reasons are dropped with a warning
    to avoid unbounded marker accumulation on a failing server."""
    _pending_tasks: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    """References to fire-and-forget ingestion tasks. Prevents GC and
    enables ``after_run()`` to await them before client teardown."""

    @property
    def has_wrap_node_run(self) -> bool:
        """Return ``False`` — Viking does not wrap node execution."""
        return False

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: Any,
        tool_def: Any,
        args: Any,
        handler: Any,
    ) -> Any:
        """Intercept tool execution to block viking:// URIs in protected tools.

        When ``uri_guard_enabled`` is ``True`` and the tool name is in
        ``uri_guard_protected_tools`` and any argument contains
        ``viking://``, return an error string without calling ``handler``.
        Otherwise, pass through to ``handler(args)``.

        Args:
            ctx: The pydantic-ai run context.
            call: The ``ToolCallPart`` containing the tool name.
            tool_def: The ``ToolDefinition`` for the tool being called.
            args: The validated tool arguments.
            handler: The next executor callable.

        Returns:
            The tool result from ``handler(args)``, or an error string when
            the URI guard blocks the call.
        """
        if self.uri_guard_enabled:
            tool_name: str = call.tool_name
            if tool_name in self.uri_guard_protected_tools:
                args_str = str(args)
                if "viking://" in args_str:
                    return (
                        f"viking:// URIs cannot be accessed with '{tool_name}'. "
                        "Use viking_read or viking_search to access Viking resources."
                    )
        return await handler(args)

    async def _resolve_identity(self) -> VikingIdentity:
        """Resolve Viking identity using a three-tier fallback chain.

        Resolution order:
        1. Explicit config fields (``self.user`` and ``self.account`` both set)
        2. API key decode (``_try_decode_api_key()``)
        3. ``/health`` endpoint query
        4. Fallback to ``VikingIdentity("default", "default", "user")``

        Returns:
            The resolved ``VikingIdentity``, cached in ``self._identity``.
        """
        if self._identity is not None:
            return self._identity

        # Tier 1: explicit config fields
        if self.account is not None and self.user is not None:
            self._identity = VikingIdentity(
                account_id=self.account,
                user_id=self.user,
                role="user",
            )
            return self._identity

        # Tier 2: API key decode
        if self.api_key is not None:
            decoded = _try_decode_api_key(self.api_key)
            if decoded is not None:
                account_id, user_id = decoded
                self._identity = VikingIdentity(
                    account_id=account_id,
                    user_id=user_id,
                    role="user",
                )
                return self._identity

        # Tier 3: /health endpoint
        if self._client is not None:
            try:
                resp = await self._client._request("GET", "/health")
                # _request returns httpx.Response; parse JSON body
                if hasattr(resp, "json"):
                    resp = resp.json()
                if isinstance(resp, dict):
                    account_id = str(resp.get("account_id") or "")
                    user_id = str(resp.get("user_id") or "")
                    role = str(resp.get("role") or "user")
                    if account_id and user_id:
                        self._identity = VikingIdentity(
                            account_id=account_id,
                            user_id=user_id,
                            role=role,
                        )
                        return self._identity
            except Exception:
                logger.debug("Failed to resolve identity from /health", exc_info=True)

        # Tier 4: fallback
        logger.warning("All identity resolution tiers failed — falling back to default")
        self._identity = VikingIdentity(
            account_id="default",
            user_id="default",
            role="user",
        )
        return self._identity

    async def _ensure_client(self) -> Any:
        """Return the SDK client, lazily initializing if needed.

        Follows the same pattern as ``McpServerCap._ensure_client()``:
        if the client is already set (e.g. from ``__aenter__`` or a
        ``for_run`` copy), return it directly. Otherwise, lazily import
        and initialize the SDK client. After initialization, triggers
        identity resolution when ``auto_resolve_identity`` is enabled.

        Returns:
            The ``AsyncHTTPClient`` instance.
        """
        if self._client is not None:
            return self._client

        from openviking_sdk import AsyncHTTPClient

        kwargs: dict[str, Any] = {
            "url": self.url,
            "api_key": self.api_key,
            "account": self.account,
            "user": self.user,
            "timeout": self.timeout if self.timeout is not None else 60.0,
        }
        if self.actor_peer_id is not None:
            kwargs["actor_peer_id"] = self.actor_peer_id

        self._client = AsyncHTTPClient(**kwargs)
        await self._client.initialize()
        if self.auto_resolve_identity:
            self._identity = await self._resolve_identity()
        return self._client

    def _resolve_skills_uri(self) -> str:
        """Return the skills URI, using override or default convention.

        Uses the resolved identity's ``user_id`` when available,
        falling back to ``self.user`` then ``"default"``.

        Returns:
            The skills URI string (e.g. ``viking://user/alice/skills/``).
        """
        if self.skills_uri is not None:
            return self.skills_uri
        user_id = self._identity.user_id if self._identity is not None else (self.user or "default")
        return f"viking://user/{user_id}/skills/"

    def _check_uri_allowed(self, uri: str, *, tool_name: str = "") -> str | None:
        """Return an error message if ``uri`` is outside the allowed prefixes.

        When ``allowed_uri_prefixes`` is empty (unrestricted), always returns
        ``None``. The allowlist applies **only** to the
        ``viking://resources/`` namespace: URIs in any other namespace
        (``viking://user/...`` and everything else) are always allowed.
        Within the resources namespace, ``uri`` is allowed only when it
        starts with one of the listed prefixes.

        Args:
            uri: The ``viking://`` URI to validate.
            tool_name: Optional tool name for the error message.

        Returns:
            ``None`` if allowed, otherwise a string suitable as a tool
            return value.
        """
        if not self.allowed_uri_prefixes:
            return None
        if not uri or not uri.startswith("viking://resources/"):
            return None
        if any(uri.startswith(prefix) for prefix in self.allowed_uri_prefixes):
            return None
        name = tool_name or "viking"
        return f"{name}: URI {uri!r} is outside the allowed prefixes ({self.allowed_uri_prefixes})."

    def _allowed_prefix_for(self, uri: str) -> str | None:
        """Return the allowed prefix matched by ``uri``, or ``None``.

        Args:
            uri: The ``viking://`` URI to match against the allowlist.

        Returns:
            The first allowed prefix that ``uri`` starts with, or ``None``
            when the allowlist is empty or no prefix matches.
        """
        if not self.allowed_uri_prefixes:
            return uri
        for prefix in self.allowed_uri_prefixes:
            if uri.startswith(prefix):
                return prefix
        return None

    async def __aenter__(self) -> Self:
        """Initialize the Viking SDK client.

        Lazily imports ``AsyncHTTPClient`` from ``openviking_sdk`` and
        creates a client with the configured fields. If the client is
        already set (e.g. a ``for_run`` copy sharing the parent's client),
        this is a no-op. After initialization, triggers identity
        resolution when ``auto_resolve_identity`` is enabled.

        Returns:
            ``self`` with the client initialized.
        """
        if self._client is not None:
            return self

        from openviking_sdk import AsyncHTTPClient

        kwargs: dict[str, Any] = {
            "url": self.url,
            "api_key": self.api_key,
            "account": self.account,
            "user": self.user,
            "timeout": self.timeout if self.timeout is not None else 60.0,
        }
        if self.actor_peer_id is not None:
            kwargs["actor_peer_id"] = self.actor_peer_id

        self._client = AsyncHTTPClient(**kwargs)
        await self._client.initialize()
        if self.auto_resolve_identity:
            self._identity = await self._resolve_identity()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the SDK client if this instance owns it.

        Sets ``_client`` to ``None`` regardless of ownership to prevent
        use-after-close errors on shared clients.
        """
        if self._owns_client and self._client is not None:
            await self._client.close()
        self._client = None

    async def for_run(self, ctx: RunContext[Any]) -> VikingCapability:
        """Create a per-run copy that shares the parent's client.

        Uses ``dataclasses.replace()`` to copy all fields, then resets
        per-run state to defaults. The returned copy shares the parent's
        ``_client`` and ``_identity`` (inherited) but has
        ``_owns_client=False`` so it will not close the shared client
        on ``__aexit__``.

        Args:
            ctx: The pydantic-ai run context (unused but required by
                the ``AbstractCapability`` interface).

        Returns:
            A new ``VikingCapability`` sharing the same client and identity.
        """
        return dataclasses.replace(
            self,
            _owns_client=False,
            _identity=self._identity,
            _profile_injected=False,
            _last_ingested_idx=0,
            _remember_pending=[],
            _remember_drain_failures=0,
            _pending_tasks=set(),
        )

    def get_instructions(self) -> str | None:
        """Return the Viking workflow instructions.

        When ``allowed_uri_prefixes`` is configured, appends a dynamic
        block listing the allowed prefixes so the model can pass a
        ``target_uri`` and skip discovery probing.

        Returns:
            The instruction string from ``instructions.py``.
        """
        from wolfharness.capabilities.viking.instructions import (
            _VIKING_INSTRUCTIONS,
            format_allowed_prefixes_block,
        )

        if not self.allowed_uri_prefixes:
            return _VIKING_INSTRUCTIONS
        return _VIKING_INSTRUCTIONS + format_allowed_prefixes_block(self.allowed_uri_prefixes)

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build a ``FunctionToolset`` from tools filtered by ``self.mode``.

        Returns ``None`` if no tools are available for the current mode.

        Returns:
            A ``FunctionToolset`` with the mode-appropriate tools, or
            ``None`` if the tool list is empty.
        """
        from wolfharness.capabilities.viking.tools import build_tools

        tool_fns = build_tools(self)
        if not tool_fns:
            return None
        return FunctionToolset(tool_fns, id="viking")

    async def get_tools(self) -> Sequence[Any]:
        """Return tools as ``Tool`` objects for listing endpoints.

        This is required by ``_get_all_tools()`` in ``base_agent.py``,
        which uses the ``_ToolProviding`` Protocol (``get_tools()``).
        Without this, Viking tools won't appear in the OpenCode
        ``/experimental/tool`` endpoint.

        Returns:
            A list of ``FunctionTool`` objects wrapping the tool closures.
        """
        from wolfharness.capabilities.viking.tools import build_tools
        from wolfharness.tools.base import FunctionTool

        tool_fns = build_tools(self)
        return [FunctionTool.from_callable(fn) for fn in tool_fns]

    def on_change(self) -> AsyncIterator[ChangeEvent] | None:
        """Return ``None`` — Viking tools never change at runtime."""
        return None

    # ---- SkillResource Protocol ----

    async def list_skills(self) -> list[SkillEntry]:
        """List available skills from the Viking server.

        Calls ``client.ls(skills_uri)`` and filters for ``.md`` files.

        Returns:
            A list of ``SkillEntry`` descriptors with ``source="remote"``.
            Returns an empty list on error.
        """
        try:
            client = await self._ensure_client()
            uri = self._resolve_skills_uri()
            if self._check_uri_allowed(uri, tool_name="list_skills") is not None:
                return []
            entries = await client.ls(uri)
            if not isinstance(entries, list):
                return []

            from wolfharness.capabilities.resource_protocols import SkillEntry

            skills: list[SkillEntry] = []
            for entry in entries:
                name: str
                if isinstance(entry, dict):
                    name = str(entry.get("name") or entry.get("uri") or "")
                else:
                    name = str(entry)
                if name.endswith(".md"):
                    skill_name = name[:-3]  # strip .md
                    skills.append(
                        SkillEntry(
                            name=skill_name,
                            uri=f"{uri}{name}",
                            source="remote",
                            skill_path=None,
                        )
                    )
            return skills
        except Exception:
            logger.warning("list_skills failed", exc_info=True)
            return []

    async def read_skill(self, name: str) -> str | None:
        """Read a skill's content from the Viking server.

        Args:
            name: Skill name (without ``.md`` extension).

        Returns:
            Skill content as a string, or ``None`` if not found or on error.
        """
        try:
            client = await self._ensure_client()
            uri = self._resolve_skills_uri()
            if self._check_uri_allowed(uri, tool_name="read_skill") is not None:
                return None
            content = await client.read(f"{uri}{name}.md")
            return str(content) if content else None
        except Exception:
            logger.warning("read_skill failed", name=name, exc_info=True)
            return None

    async def skill_exists(self, name: str) -> bool:
        """Check if a skill exists on the Viking server.

        Args:
            name: Skill name (without ``.md`` extension).

        Returns:
            ``True`` if the skill exists, ``False`` otherwise or on error.
        """
        try:
            client = await self._ensure_client()
            uri = self._resolve_skills_uri()
            if self._check_uri_allowed(uri, tool_name="skill_exists") is not None:
                return False
            entries = await client.ls(uri)
            if not isinstance(entries, list):
                return False
            target = f"{name}.md"
            for entry in entries:
                entry_name: str
                if isinstance(entry, dict):
                    entry_name = str(entry.get("name") or entry.get("uri") or "")
                else:
                    entry_name = str(entry)
                if entry_name == target:
                    return True
            return False
        except Exception:
            logger.warning("skill_exists failed", name=name, exc_info=True)
            return False

    # ---- ResourceAccess Protocol (Phase 5) ----

    def _resolve_resources_uri(self) -> str:
        """Return the resources URI, using override or default convention.

        Returns:
            The resources URI string (e.g. ``viking://resources/``).
        """
        if self.resources_uri is not None:
            return self.resources_uri
        return "viking://resources/"

    def _resolve_sessions_uri(self) -> str:
        """Return the sessions URI, using override or default convention.

        Uses the resolved identity's ``user_id`` when available,
        falling back to ``self.user`` then ``"default"``.

        Returns:
            The sessions URI string (e.g.
            ``viking://user/{user_id}/sessions/``).
        """
        if self.sessions_uri is not None:
            return self.sessions_uri
        user_id = self._identity.user_id if self._identity is not None else (self.user or "default")
        return f"viking://user/{user_id}/sessions/"

    def _resolve_memories_uri(self) -> str:
        """Return the memories URI, using override or default convention.

        Uses the resolved identity's ``user_id`` when available,
        falling back to ``self.user`` then ``"default"``.

        Returns:
            The memories URI string (e.g.
            ``viking://user/{user_id}/memories/``).
        """
        if self.memories_uri is not None:
            return self.memories_uri
        user_id = self._identity.user_id if self._identity is not None else (self.user or "default")
        return f"viking://user/{user_id}/memories/"

    async def _handle_compaction(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Archive old conversation messages when token count exceeds threshold.

        Estimates total tokens from ``request_context.messages``. If the
        count exceeds ``compaction_threshold``, splits messages into
        archivable (old) and keep (recent) lists, writes the archivable
        portion to ``viking://user/{user_id}/memories/compacted/{uuid}.md``,
        and replaces them with a summary + URI reference.

        After removing N messages from the front of the list, decrements
        ``_last_ingested_idx`` by N (clamped to 0) to prevent a stale
        ingestion cursor.

        On any error, logs a warning and returns the original
        ``request_context`` unchanged.

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The (possibly modified) model request context.
        """
        if not self.compaction_enabled:
            return request_context

        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        from wolfharness.capabilities.viking.compaction import (
            _estimate_tokens,
            _replace_old_messages,
            _serialize_messages,
            _split_archivable,
            _summarize_messages,
        )

        # Estimate total tokens across all messages.
        total_tokens = 0
        for msg in request_context.messages:
            if isinstance(msg, ModelRequest | ModelResponse):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            total_tokens += _estimate_tokens(content)
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, str):
                                    total_tokens += _estimate_tokens(item)
                                elif isinstance(item, TextPart):
                                    total_tokens += _estimate_tokens(item.content)
                    elif isinstance(part, TextPart):
                        total_tokens += _estimate_tokens(part.content)

        if total_tokens <= self.compaction_threshold:
            return request_context

        try:
            client = await self._ensure_client()

            archivable, keep = _split_archivable(
                request_context.messages,
                self.compaction_keep_recent_turns,
            )

            if not archivable:
                return request_context

            # Serialize and summarize.
            serialized = _serialize_messages(archivable)
            summary = _summarize_messages(archivable)

            # Write archive to Viking.
            user_id = (
                self._identity.user_id if self._identity is not None else (self.user or "default")
            )
            archive_uri = f"viking://user/{user_id}/memories/compacted/{uuid.uuid4().hex[:12]}.md"
            if self._check_uri_allowed(archive_uri, tool_name="compaction") is not None:
                logger.debug(
                    "compaction: archive URI %s outside allowed prefixes — skipping",
                    archive_uri,
                )
                return request_context
            await client.write(archive_uri, serialized, mode="create")

            # Replace old messages with summary + URI reference.
            new_context = _replace_old_messages(
                request_context,
                archivable,
                keep,
                archive_uri,
                summary,
            )

            # Adjust ingestion cursor: N messages removed from front.
            n_removed = len(archivable)
            self._last_ingested_idx = max(0, self._last_ingested_idx - n_removed)

            logger.info(
                "Compaction archived %d messages to %s (tokens: %d -> est. %d)",
                n_removed,
                archive_uri,
                total_tokens,
                total_tokens // 2,  # rough estimate after compaction
            )
            return new_context
        except Exception:
            logger.warning("Compaction failed — returning original context", exc_info=True)
            return request_context

    async def _list_resource_entries_from_uri(self, client: Any, uri: str) -> list[ResourceEntry]:
        """Recursively list files under a single Viking URI.

        Performs a per-directory recursive ``client.ls()`` to work around
        Viking's incomplete root-level recursive traversal, then builds
        ``ResourceEntry`` objects for each file (filtering by configured
        extensions and inferring MIME types).

        Args:
            client: The Viking SDK client.
            uri: The base URI to list (e.g. ``viking://resources/``).

        Returns:
            A list of ``ResourceEntry`` descriptors for text files.
        """
        from wolfharness.capabilities.resource_protocols import ResourceEntry

        top_entries = await client.ls(uri)
        if not isinstance(top_entries, list):
            return []

        sub_uris = [
            str(entry.get("uri"))
            for entry in top_entries
            if isinstance(entry, dict) and entry.get("isDir") and entry.get("uri")
        ]
        # Per-directory recursive ls in parallel; a slow directory must not
        # serialize the rest.
        sub_results = await asyncio.gather(
            *[client.ls(sub_uri, recursive=True, node_limit=5000) for sub_uri in sub_uris],
            return_exceptions=True,
        )

        all_entries: list[dict[str, Any]] = [
            entry for entry in top_entries if isinstance(entry, dict) and not entry.get("isDir")
        ]
        for result in sub_results:
            if isinstance(result, list):
                all_entries.extend(e for e in result if isinstance(e, dict))

        resources: list[ResourceEntry] = []
        for entry in all_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("isDir"):
                continue
            resource_uri = str(entry.get("uri") or "")
            if not resource_uri:
                continue
            name = str(entry.get("name") or resource_uri.rsplit("/", 1)[-1] or resource_uri)
            # Filter by configured extensions; skip files not in the set
            lowered = name.lower()
            if self.resource_file_extensions and not lowered.endswith(
                self.resource_file_extensions
            ):
                continue
            # Infer MIME type from extension
            mime_type = ""
            if lowered.endswith(".md"):
                mime_type = "text/markdown"
            elif lowered.endswith(".txt"):
                mime_type = "text/plain"
            elif lowered.endswith(".json"):
                mime_type = "application/json"
            elif lowered.endswith((".yaml", ".yml")):
                mime_type = "text/yaml"
            elif lowered.endswith(".html"):
                mime_type = "text/html"
            resources.append(
                ResourceEntry(
                    uri=resource_uri,
                    name=name,
                    description=resource_uri.removeprefix(uri),
                    mime_type=mime_type,
                )
            )
        return resources

    async def list_resources(self) -> Sequence[ResourceEntry]:
        """List Viking resources from multiple URI trees.

        Lists files from both ``resources_uri`` (shared resources) and
        ``sessions_uri`` (user session content), merges and deduplicates
        by URI, then enriches descriptions with L0 abstracts.

        When ``allowed_uri_prefixes`` is non-empty, only ``viking://resources/``
        trees are restricted — a resources tree outside the allowlist is
        narrowed to its allowed sub-prefixes. All other trees (own sessions,
        override locations outside the resources namespace) are listed as-is.

        Files are what users @ mention — directories can't be read as
        content.

        Returns:
            A sequence of ``ResourceEntry`` descriptors for text files.
            Returns an empty list on error.
        """
        try:
            client = await self._ensure_client()
            candidates = [self._resolve_resources_uri(), self._resolve_sessions_uri()]
            uris: list[str] = []
            for u in candidates:
                if not self.allowed_uri_prefixes or not u.startswith("viking://resources/"):
                    uris.append(u)
                    continue
                if self._allowed_prefix_for(u) is not None:
                    uris.append(u)
                    continue
                # Base tree outside the allowlist — list each allowed prefix
                # that lives under this tree instead.
                for prefix in self.allowed_uri_prefixes:
                    if prefix.startswith(u) and prefix not in uris:
                        uris.append(prefix)

            if not uris:
                return []

            # List from each URI tree in parallel
            results = await asyncio.gather(
                *[self._list_resource_entries_from_uri(client, u) for u in uris],
                return_exceptions=True,
            )

            # Merge results, deduplicate by URI
            seen_uris: set[str] = set()
            resources: list[ResourceEntry] = []

            for result in results:
                if not isinstance(result, list):
                    continue
                for entry in result:
                    if entry.uri not in seen_uris:
                        seen_uris.add(entry.uri)
                        resources.append(entry)

            return resources
        except Exception:
            logger.warning("list_resources failed", exc_info=True)
            return []

    async def read_resource(
        self, uri: str
    ) -> list[TextResourceContent | BlobResourceContent] | None:
        """Read a Viking resource by URI.

        Uses the configured ``resource_read_level`` to determine content
        depth (L0 abstract, L1 overview, or L2 full content). Falls back
        to L2 (``client.read``) if the requested level is unavailable.

        Args:
            uri: The Viking URI of the resource to read.

        Returns:
            A list containing a ``TextResourceContent`` with the resource
            content, or ``None`` if not found or on error.
        """
        if self._check_uri_allowed(uri, tool_name="read_resource") is not None:
            return None
        try:
            client = await self._ensure_client()

            # Image resources are served as decoded blob bytes (with their
            # MIME type) so vision-capable models can consume them directly —
            # mirrors ``viking_read``. SVG stays on the text path: most vision
            # APIs reject the vector format.
            from pathlib import PurePosixPath

            from wolfharness.capabilities.viking.constants import (
                IMAGE_BLOB_MAX_BYTES,
                IMAGE_EXTENSIONS,
                IMAGE_MIME_TYPES,
            )
            from wolfharness.capabilities.resource_protocols import (
                TextResourceContent,
            )

            suffix = PurePosixPath(uri).suffix.lower()
            if (
                suffix in IMAGE_EXTENSIONS
                and suffix != ".svg"
                and uri.startswith("viking://")
                and self._should_return_image_bytes()
            ):
                data = await client.download_bytes(uri)
                if len(data) > IMAGE_BLOB_MAX_BYTES:
                    from wolfharness.capabilities.viking.tools import _image_uri_hint

                    return [
                        TextResourceContent(
                            uri=uri,
                            mime_type="text/plain",
                            text=_image_uri_hint(uri),
                        )
                    ]
                import base64

                mime_type = IMAGE_MIME_TYPES.get(suffix, "application/octet-stream")
                from wolfharness.capabilities.resource_protocols import (
                    BlobResourceContent,
                )

                return [
                    BlobResourceContent(
                        uri=uri,
                        mime_type=mime_type,
                        blob=base64.b64encode(data).decode("ascii"),
                    )
                ]

            # Use configured read level (L0/L1/L2), fallback to L2 if unavailable
            content: str | None = None
            if self.resource_read_level == "abstract":
                try:
                    content = await client.abstract(uri)
                except Exception:
                    content = await client.read(uri)  # fallback to L2
            elif self.resource_read_level == "overview":
                try:
                    content = await client.overview(uri)
                except Exception:
                    content = await client.read(uri)  # fallback to L2
            else:
                content = await client.read(uri)

            if not content:
                return None

            return [
                TextResourceContent(
                    uri=uri,
                    mime_type="text/markdown" if uri.endswith(".md") else None,
                    text=str(content),
                )
            ]
        except Exception:
            logger.warning("read_resource failed", uri=uri, exc_info=True)
            return None

    async def resource_exists(self, uri: str) -> bool:
        """Check if a Viking resource exists.

        Args:
            uri: The Viking URI of the resource to check.

        Returns:
            ``True`` if the resource exists, ``False`` otherwise or on error.
        """
        if self._check_uri_allowed(uri, tool_name="resource_exists") is not None:
            return False
        try:
            client = await self._ensure_client()
            parent = uri.rsplit("/", 1)[0] + "/"
            name = uri.rsplit("/", 1)[1]
            entries = await client.ls(parent)
            if not isinstance(entries, list):
                return False
            for entry in entries:
                entry_name = str(entry.get("name") or "") if isinstance(entry, dict) else str(entry)
                if entry_name == name:
                    return True
            return False
        except Exception:
            logger.warning("resource_exists failed", uri=uri, exc_info=True)
            return False

    # ---- Profile Injection (Feature 5) ----

    _FIRST_TURN_MAX_MESSAGES = 2

    async def _handle_profile_inject(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Inject a profile block from Viking memories on the first turn.

        When ``profile_enabled`` is ``True`` and this is the first
        ``before_model_request`` call for the session (``_profile_injected``
        is ``False`` and message count is <= 2), the method:

        1. Derives a context hint from ``ctx.deps``.
        2. Queries Viking memories via ``client.find()``.
        3. Formats results as an ``<openviking-profile>`` XML block.
        4. Injects it as a ``SystemPromptPart`` before the latest user
           message using ``dataclasses.replace()``.
        5. Sets ``_profile_injected = True``.

        On any error, logs a warning, sets ``_profile_injected = True``
        (to avoid retrying), and returns the original ``request_context``.

        When ``profile_first_turn_only`` is ``False``, the first-turn
        message count check is skipped — injection runs whenever
        ``_profile_injected`` is ``False``.

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The (possibly modified) model request context.
        """
        if not self.profile_enabled or self._profile_injected:
            return request_context

        if (
            self.profile_first_turn_only
            and len(request_context.messages) > self._FIRST_TURN_MAX_MESSAGES
        ):
            self._profile_injected = True
            return request_context

        self._profile_injected = True

        if (
            self._check_uri_allowed(self._resolve_memories_uri(), tool_name="profile_inject")
            is not None
        ):
            logger.debug("profile_inject: memories_uri outside allowed prefixes — skipping")
            return request_context

        try:
            client = await self._ensure_client()
            from wolfharness.capabilities.viking.profile import (
                _derive_context_hint,
                _format_profile_block,
            )

            hint = _derive_context_hint(ctx)
            memories_uri = self._resolve_memories_uri()
            results = await client.find(
                query=hint,
                target_uri=memories_uri,
                limit=self.profile_limit,
                context_type="memory",
            )
            profile_block = _format_profile_block(
                results,
                max_tokens=self.profile_max_tokens,
            )
            if not profile_block.strip():
                return request_context

            from dataclasses import replace

            from pydantic_ai.messages import (
                ModelRequest,
                SystemPromptPart,
                UserPromptPart,
            )

            messages = list(request_context.messages)
            insert_idx: int | None = None
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, ModelRequest) and any(
                    isinstance(p, UserPromptPart) for p in msg.parts
                ):
                    insert_idx = i
                    break

            if insert_idx is None:
                return request_context

            system_msg = ModelRequest(parts=[SystemPromptPart(content=profile_block)])
            new_messages = [*messages[:insert_idx], system_msg, *messages[insert_idx:]]
            return replace(request_context, messages=new_messages)
        except Exception:
            logger.warning("Profile injection failed", exc_info=True)
            return request_context

    # ---- Multimodal Bridge (Phase 6) ----

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Run the pre-model handler chain.

        Executes all enabled handlers in order (D7):

        0. Remember drain (capture deferred ``viking_remember`` intents)
        1. Auto-ingest (process previous turn, fire-and-forget)
        2. Profile injection (add static profile, first turn only)
        3. Auto-recall (add dynamic recalled memories)
        4. Compaction archive (reduce context if too large)
        5. Multimodal bridge (handle binary content)

        Each handler checks its own enabled flag and returns early if
        disabled (D14). Handlers receive the output of the previous
        handler (chained ``request_context`` modification).

        When ``self._client`` is ``None``, returns the original
        ``request_context`` unchanged — all handlers require a client.

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The (possibly modified) model request context.
        """
        if self._client is None:
            return request_context

        with logfire.span("viking.before_model_request"):
            request_context = await self._handle_remember_drain(ctx, request_context)
            request_context = await self._handle_auto_ingest(ctx, request_context)
            request_context = await self._handle_profile_inject(ctx, request_context)
            request_context = await self._handle_auto_recall(ctx, request_context)
            request_context = await self._handle_compaction(ctx, request_context)
            return await self._handle_multimodal_bridge(ctx, request_context)

    async def _handle_remember_drain(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Capture conversations queued by ``viking_remember`` calls.

        Runs as step 0 of ``before_model_request`` — before auto-ingest —
        so it drains first and advances the shared cursor, making the
        auto-ingest range disjoint. Independent of ``auto_ingest_enabled``.

        Extracts the real conversation pairs since ``_last_ingested_idx``,
        appends one ``<memory-intent>`` marker per pending reason, and
        sanitizes unconditionally (deferred capture runs after recall /
        profile injection, so injected blocks must always be stripped to
        avoid re-ingesting them). The commit is awaited synchronously —
        only an HTTP round-trip; Phase 2 extraction is not awaited. On
        success the cursor advances and pending reasons clear; on failure
        both are left intact for the next boundary's retry (bounded by
        ``_REMEMBER_MAX_RETRIES``). A background task then polls the
        extraction task and steers the resulting memory diff into the
        session when ``remember_notify`` is enabled.

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The unchanged ``request_context`` (ingestion does not modify
            the request context).
        """
        if not self._remember_pending:
            return request_context

        messages = request_context.messages
        current_count = len(messages)
        if current_count <= self._last_ingested_idx:
            return request_context

        reasons = list(self._remember_pending)
        pairs = _extract_conversation_pairs(messages, self._last_ingested_idx)
        if not pairs:
            self._remember_pending = reasons
            return request_context

        # Sanitize unconditionally (D5), then append intent markers.
        pairs = [{"role": p["role"], "content": _sanitize_message(p["content"])} for p in pairs]
        for reason in reasons:
            if reason:
                pairs.append({
                    "role": "user",
                    "content": _MEMORY_INTENT_TEMPLATE.format(reason=reason),
                })

        try:
            client = await self._ensure_client()
        except Exception:
            logger.warning("remember: failed to ensure client", exc_info=True)
            self._mark_remember_failure(reasons)
            return request_context

        session_id = f"remember-{uuid.uuid4().hex[:12]}"
        try:
            commit_result = await _ingest_conversation(
                client,
                pairs,
                session_id=session_id,
                source_type="remember",
            )
        except Exception:
            logger.warning("remember: ingestion failed", exc_info=True)
            self._mark_remember_failure(reasons)
            return request_context

        # Success — advance cursor and clear reasons atomically.
        self._last_ingested_idx = current_count
        self._remember_pending = []
        self._remember_drain_failures = 0
        self._spawn_memory_notify(ctx, client, commit_result)
        return request_context

    def _mark_remember_failure(self, reasons: list[str]) -> None:
        """Record a failed remember drain, capping retries.

        ``_remember_pending`` was snapshotted but never cleared at drain
        start, so on failure it already holds the original reasons
        untouched — the retry at the next boundary re-extracts the same
        range with the same ``<memory-intent>`` markers. Only the retry
        cap (``_REMEMBER_MAX_RETRIES`` consecutive failures) drops them,
        with a warning, to bound accumulation on a failing server.

        Args:
            reasons: The snapshot of reasons from the failed drain (used
                for the drop warning).
        """
        self._remember_drain_failures += 1
        if self._remember_drain_failures >= _REMEMBER_MAX_RETRIES:
            logger.warning(
                "remember: %d consecutive drain failures — dropping %d pending reason(s)",
                self._remember_drain_failures,
                len(reasons),
            )
            self._remember_pending = []

    def _spawn_memory_notify(self, ctx: RunContext[Any], client: Any, commit_result: Any) -> None:
        """Spawn the background memory-diff notification task.

        Best-effort: skips silently when notifications are disabled, the
        commit result carries no archive/task ids, or no steer channel is
        available. The task self-bounds via ``_REMEMBER_NOTIFY_TIMEOUT``
        and is intentionally not tracked in ``_pending_tasks`` (that flush
        exists for commit tasks; steering is a post-commit notification).

        Args:
            ctx: The pydantic-ai run context (used to capture the steer
                channel and run session id at drain time).
            client: The SDK client used for the commit.
            commit_result: The ``commit_session`` response.
        """
        if not self.remember_notify or not isinstance(commit_result, dict):
            return
        if not commit_result.get("archive_uri") or not commit_result.get("task_id"):
            return
        session_pool, run_session_id = self._capture_steer_channel(ctx)
        if session_pool is None or run_session_id is None:
            return
        # Spawn bounded (30s) background notification. Intentional: not
        # tracked in _pending_tasks — that flush exists for commit tasks
        # and would time out on this task's poll duration. The task body
        # runs under its own logfire span; asyncio holds a reference while
        # it is pending and the concrete task object is not referenced
        # further.
        notify_task = asyncio.create_task(
            self._notify_memory_diff(client, commit_result, session_pool, run_session_id)
        )
        logger.debug("remember: notification task %s spawned", id(notify_task))

    @staticmethod
    def _capture_steer_channel(ctx: RunContext[Any]) -> tuple[Any, str | None]:
        """Capture the session steer channel and run session id at drain time.

        Follows the DCP nudge pattern: ``ctx.deps.node.host_context`` →
        ``session_pool``. Defensive because the viking deps type is not
        guaranteed to carry ``node`` (tests use bare deps).

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            A ``(session_pool_or_None, run_session_id_or_None)`` tuple.
        """
        from wolfharness.capabilities.viking.tools import _get_session_id

        run_session_id = _get_session_id(ctx)
        try:
            host_ctx = ctx.deps.node.host_context
        except AttributeError:
            host_ctx = None
        session_pool = host_ctx.session_pool if host_ctx is not None else None
        return session_pool, run_session_id

    async def _notify_memory_diff(
        self,
        client: Any,
        commit_result: dict[str, Any],
        session_pool: Any,
        run_session_id: str,
    ) -> None:
        """Poll the extraction task, then steer the memory diff summary.

        Runs as a background task after a successful remember commit.
        Waits for the asynchronous Phase 2 extraction to finish, reads
        ``{archive_uri}/memory_diff.json``, formats the added/updated/
        deleted URIs, and steers them into the session via the session
        pool's background-task entry point (which survives run
        boundaries and falls back to the session's feedback queue).

        Args:
            client: The SDK client used for the commit.
            commit_result: The ``commit_session`` response.
            session_pool: The captured ``SessionPool`` (or compatible).
            run_session_id: The run session id to steer into.
        """
        with logfire.span("viking.memory_diff_notify"):
            archive_uri = commit_result.get("archive_uri")
            task_id = commit_result.get("task_id")
            try:
                if not await self._wait_for_extraction(client, task_id):
                    return
                diff = await read_memory_diff(client, str(archive_uri))
                summary = format_memory_diff_summary(diff)
                if not summary:
                    logger.debug("remember: memory diff empty — notify skipped")
                    return
                await session_pool.steer_from_background_task(run_session_id, summary)
            except Exception:
                logger.warning("remember: memory notification failed", exc_info=True)

    @staticmethod
    async def _wait_for_extraction(
        client: Any,
        task_id: str | None,
        *,
        timeout: float = _REMEMBER_NOTIFY_TIMEOUT,
        interval: float = 1.0,
    ) -> bool:
        """Poll the extraction task until it completes or times out.

        Args:
            client: The SDK client used for the commit.
            task_id: The extraction task id from ``commit_session``.
            timeout: Maximum total wait in seconds.
            interval: Poll interval in seconds.

        Returns:
            ``True`` when the task reached a completed state.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = await client._request("GET", f"/tasks/{task_id}")
            except Exception:
                logger.warning("remember: extraction poll failed", exc_info=True)
                return False
            status = _extract_task_status(resp)
            if status in ("completed", "succeeded", "done"):
                return True
            if status in ("failed", "error", "cancelled"):
                logger.warning("remember: extraction task ended with status %r", status)
                return False
            await asyncio.sleep(interval)
        logger.warning("remember: extraction poll timed out after %.0fs", timeout)
        return False

    async def _handle_multimodal_bridge(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Upload binary content to Viking and replace based on model capabilities.

        When ``multimodal_bridge`` is disabled, returns the original
        ``request_context`` unchanged.

        For each ``BinaryContent`` found in ``UserPromptPart`` content:

        - Text-only model → text reference with ``viking://`` URI
        - Multimodal + ``public_download_base_url`` → HTTP URL
        - Multimodal + no URL → keep original (but persisted in Viking)

        Args:
            ctx: The pydantic-ai run context (unused).
            request_context: The model request context containing messages.

        Returns:
            The (possibly modified) model request context.
        """
        if not self.multimodal_bridge:
            return request_context

        from dataclasses import replace

        from pydantic_ai.messages import (
            BinaryContent,
            ModelRequest,
            TextPart,
            UserPromptPart,
        )

        new_messages: list[Any] = []
        modified = False
        for msg in request_context.messages:
            if not isinstance(msg, ModelRequest):
                new_messages.append(msg)
                continue

            new_parts: list[Any] = []
            msg_modified = False
            for part in msg.parts:
                if not isinstance(part, UserPromptPart):
                    new_parts.append(part)
                    continue

                content = part.content
                if not isinstance(content, list):
                    new_parts.append(part)
                    continue

                new_content: list[Any] = []
                for item in content:
                    if not isinstance(item, BinaryContent):
                        new_content.append(item)
                        continue

                    viking_uri = await self._upload_binary(item)
                    if viking_uri is None:
                        new_content.append(item)
                        continue

                    supports = self._supports_modality(item.media_type)
                    if not supports:
                        new_content.append(
                            TextPart(
                                content=(
                                    f"[Content stored at {viking_uri}. Use viking_read to access.]"
                                ),
                            )
                        )
                        msg_modified = True
                    elif self.public_download_base_url:
                        http_url = f"{self.public_download_base_url}?uri={viking_uri}"
                        new_content.append(TextPart(content=http_url))
                        msg_modified = True
                    else:
                        new_content.append(item)

                if msg_modified:
                    new_parts.append(replace(part, content=new_content))
                    modified = True
                else:
                    new_parts.append(part)

            if msg_modified:
                new_messages.append(replace(msg, parts=new_parts))
            else:
                new_messages.append(msg)

        if not modified:
            return request_context
        return replace(request_context, messages=new_messages)

    def _should_return_image_bytes(self) -> bool:
        """Whether ``viking_read`` should return image bytes for image URIs.

        Decision order:

        1. ``support_vision`` explicitly set — return its value.
        2. ``model_capabilities`` injected — return ``image_input`` (``None``
           counts as text-only).
        3. Otherwise — text-only (safe degradation).

        Note: unlike ``ModalityFilterCapability._is_modality_supported``
        (which treats ``capabilities=None`` as pass-through), this treats an
        unavailable capability as text-only: this capability *produces*
        image content, so it must never emit ``BinaryImage`` it cannot
        guarantee the model accepts.

        Returns:
            ``True`` when image bytes should be returned, ``False`` for text.
        """
        if self.support_vision is not None:
            return self.support_vision
        caps = self.model_capabilities
        return bool(caps and caps.image_input)

    def _supports_modality(self, media_type: str) -> bool:
        """Check if the model supports the given media type.

        Dispatches on ``media_type`` prefix to the appropriate
        ``ModelCapabilities`` field.

        Args:
            media_type: The MIME type of the content (e.g. ``"image/png"``).

        Returns:
            ``True`` if the model supports this modality, ``False`` otherwise.
        """
        caps = self.model_capabilities
        if caps is None:
            return False
        if media_type.startswith("image/"):
            return bool(caps.image_input)
        if media_type.startswith("audio/"):
            return bool(caps.audio_input)
        if media_type.startswith("video/"):
            return bool(caps.video_input)
        if media_type in (
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            return bool(caps.document_input)
        return False

    async def _upload_binary(self, content: BinaryContent) -> str | None:
        """Upload binary content to Viking under ``uploads_uri``.

        Generates a unique URI and uploads via ``client.write()`` with
        base64-encoded content.

        Args:
            content: The ``BinaryContent`` to upload.

        Returns:
            The Viking URI of the uploaded content, or ``None`` on failure.
        """
        try:
            client = await self._ensure_client()
            user_id = (
                self._identity.user_id if self._identity is not None else (self.user or "default")
            )
            uploads_uri = self.uploads_uri or (f"viking://user/{user_id}/memories/uploads/")
            # Viking server only allows .md files; store binary as base64
            # text inside a .md container.
            uri = f"{uploads_uri}{uuid.uuid4().hex[:12]}.md"

            if self._check_uri_allowed(uri, tool_name="multimodal_bridge") is not None:
                logger.debug(
                    "multimodal_bridge: upload URI %s outside allowed prefixes — skipping",
                    uri,
                )
                return None

            # write() accepts text content; encode binary as base64
            import base64

            b64_data = base64.b64encode(content.data).decode("ascii")
            await client.write(uri, b64_data, mode="create")
            return uri
        except Exception:
            logger.warning("_upload_binary failed", exc_info=True)
            return None

    # ---- Auto Semantic Recall ----

    async def _handle_auto_recall(  # noqa: PLR0911
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Perform semantic recall and inject results before the model request.

        Extracts the latest user prompt, queries the Viking knowledge graph
        using ``client.search()`` (session-aware) or ``client.find()``
        (deduplicated), ranks/deduplicates results, formats them as an
        ``<openviking-recall>`` XML block, and injects a system message
        before the latest user message.

        When ``auto_recall_method`` is ``"search"``, first calls
        ``client.get_session_context()`` to retrieve existing session context
        for inclusion in the recall block.

        On any error, logs a warning and returns the original
        ``request_context`` unchanged.

        Args:
            ctx: The pydantic-ai run context (unused — session_id is
                extracted from ``ctx.deps``).
            request_context: The model request context containing messages.

        Returns:
            A new ``ModelRequestContext`` with the recall block injected,
            or the original context if recall is disabled, no user prompt
            is found, or an error occurs.
        """
        from wolfharness.capabilities.viking.recall import (
            _extract_latest_user_prompt,
            _format_recall_block,
            _inject_system_message,
            _rank_and_dedup,
        )

        if not self.auto_recall_enabled:
            return request_context

        memories_uri = self._resolve_memories_uri()
        if self._check_uri_allowed(memories_uri, tool_name="auto_recall") is not None:
            logger.debug(
                "auto_recall: memories_uri %s outside allowed prefixes — skipping",
                memories_uri,
            )
            return request_context

        # Extract the latest user prompt
        prompt = _extract_latest_user_prompt(request_context.messages)
        if prompt is None:
            return request_context

        try:
            client = await self._ensure_client()
        except Exception:
            logger.warning("auto_recall: failed to ensure client", exc_info=True)
            return request_context

        # Extract session_id from ctx.deps if available
        from wolfharness.capabilities.viking.tools import _get_session_id

        session_id = _get_session_id(ctx)

        try:
            session_context: dict[str, Any] | None = None

            if self.auto_recall_method == "search":
                # Pre-fetch session context
                if session_id is not None:
                    try:
                        session_context = await client.get_session_context(
                            session_id, token_budget=self.auto_recall_max_tokens
                        )
                    except Exception:
                        logger.debug(
                            "auto_recall: get_session_context failed, proceeding with search only",
                            exc_info=True,
                        )
                        session_context = None

                raw_results = await client.search(
                    prompt,
                    target_uri=memories_uri,
                    limit=self.auto_recall_limit,
                    session_id=session_id,
                )
            else:
                raw_results = await client.find(
                    prompt,
                    target_uri=memories_uri,
                    limit=self.auto_recall_limit,
                )

            # Normalize results to a list of hit dicts
            hits: list[dict[str, Any]] = _normalize_search_results(raw_results)

            # Rank and deduplicate
            ranked = _rank_and_dedup(
                hits,
                query=prompt,
                lexical_boost=self.auto_recall_lexical_boost,
                category_boost=self.auto_recall_category_boost,
                context_types=self.auto_recall_context_types,
                min_score=self.auto_recall_min_score,
            )

            # Format and inject
            recall_block = _format_recall_block(
                ranked,
                session_context=session_context,
                max_tokens=self.auto_recall_max_tokens,
            )

            if not recall_block.strip():
                return request_context

            return _inject_system_message(request_context, recall_block)

        except Exception:
            logger.warning("auto_recall: recall failed", exc_info=True)
            return request_context

    # ---- Auto Conversation Ingestion ----

    async def _handle_auto_ingest(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Ingest the previous turn's conversation to Viking.

        Called from ``before_model_request`` (wired in Group 7). Extracts
        new messages since ``_last_ingested_idx``, sanitizes them, and
        fires-and-forget an ``asyncio.create_task`` to write them to a
        new Viking session. Updates the ingestion cursor regardless of
        success or failure (to avoid retrying on every subsequent turn).

        When ``auto_ingest_mode`` is ``"sync"``, awaits the ingestion
        directly instead of spawning a background task.

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The original ``request_context`` unchanged (ingestion does
            not modify the request context).
        """
        if not self.auto_ingest_enabled:
            return request_context

        messages = request_context.messages
        current_count = len(messages)

        # No new messages since last ingestion
        if current_count <= self._last_ingested_idx:
            return request_context

        # Extract conversation pairs since the cursor
        pairs = _extract_conversation_pairs(messages, self._last_ingested_idx)
        if not pairs:
            # Still advance the cursor to avoid re-scanning
            self._last_ingested_idx = current_count
            return request_context

        # Sanitize messages if enabled
        if self.auto_ingest_sanitize:
            pairs = [{"role": p["role"], "content": _sanitize_message(p["content"])} for p in pairs]

        # Update cursor BEFORE ingestion to prevent retries on failure
        self._last_ingested_idx = current_count

        try:
            client = await self._ensure_client()
        except Exception:
            logger.warning("auto_ingest: failed to ensure client", exc_info=True)
            return request_context

        # Generate a unique session ID for this ingestion
        session_id = f"ingest-{uuid.uuid4().hex[:12]}"

        async def _do_ingest() -> None:
            try:
                await _ingest_conversation(
                    client,
                    pairs,
                    session_id=session_id,
                    source_type=self.auto_ingest_source_type,
                    keep_recent_turns=self.auto_ingest_keep_recent_turns,
                )
            except Exception:
                logger.warning("auto_ingest: ingestion failed", exc_info=True)

        if self.auto_ingest_mode == "sync":
            await _do_ingest()
        else:
            task = asyncio.create_task(_do_ingest())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        return request_context

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: Any,
    ) -> Any:
        """Flush pending ingestion before the run completes.

        Two-step close-out, order matters:

        1. Await all fire-and-forget commit tasks (existing 5-second
           flush). Awaited FIRST so an auto-ingest task that claimed a
           message range but failed can't leave a gap the advanced cursor
           says is covered.
        2. Tail-flush: drain any pending remember intent plus any
           un-ingested trailing messages (``[cursor, end]``) in a final
           synchronous commit. Closes the historical gap where the final
           assistant message of a run was never ingested.

        Args:
            ctx: The pydantic-ai run context.
            result: The agent run result (passed through unchanged).

        Returns:
            The unchanged ``result``.
        """
        if self._pending_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._pending_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                logger.warning(
                    "auto_ingest: flush timed out after 5s — %d tasks pending",
                    len(self._pending_tasks),
                )
            except Exception:
                logger.warning("auto_ingest: flush failed", exc_info=True)
        try:
            await self._flush_tail(ctx)
        except Exception:
            logger.warning("remember: tail-flush failed", exc_info=True)
        return result

    async def _flush_tail(self, ctx: RunContext[Any]) -> None:
        """Flush pending remember intent and trailing messages at run end.

        Captures ``[cursor, end]`` (including the final assistant message,
        which never existed at any earlier ``before_model_request``) plus
        any pending ``<memory-intent>`` markers, in one synchronous commit.
        Sanitizes unconditionally. On success advances the cursor; on
        failure logs — the run is over, so there is nothing to retry.

        Gated on active ingestion only: runs when ``auto_ingest_enabled``
        (closing the final-turn cursor gap for automatic capture) or when
        a remember intent is pending (flushing the last-moment capture).
        When both are off the run's conversation is never auto-captured.

        Args:
            ctx: The pydantic-ai run context.
        """
        if not self.auto_ingest_enabled and not self._remember_pending:
            return

        messages = ctx.messages
        current_count = len(messages)
        if current_count <= self._last_ingested_idx and not self._remember_pending:
            return

        reasons = list(self._remember_pending)
        self._remember_pending = []
        pairs = _extract_conversation_pairs(messages, self._last_ingested_idx)
        if not pairs:
            return
        pairs = [{"role": p["role"], "content": _sanitize_message(p["content"])} for p in pairs]
        for reason in reasons:
            if reason:
                pairs.append({
                    "role": "user",
                    "content": _MEMORY_INTENT_TEMPLATE.format(reason=reason),
                })

        client = await self._ensure_client()
        session_id = f"remember-{uuid.uuid4().hex[:12]}"
        await _ingest_conversation(
            client,
            pairs,
            session_id=session_id,
            source_type="remember",
        )
        self._last_ingested_idx = current_count


def _normalize_search_results(results: Any) -> list[dict[str, Any]]:
    """Normalize SDK search/find results into a flat list of hit dicts.

    Handles both dict responses (with ``hits``, ``results``, or Viking's
    grouped keys like ``memories``/``resources``/``skills``) and list
    responses.

    Args:
        results: Raw response from ``client.search()`` or ``client.find()``.

    Returns:
        A flat list of hit dicts.
    """
    if isinstance(results, dict):
        hits: list[dict[str, Any]] = (
            results.get("hits")
            or results.get("results")
            or (
                results.get("memories", [])
                + results.get("resources", [])
                + results.get("skills", [])
            )
        )
        return hits
    if isinstance(results, list):
        return results
    return []


def _extract_task_status(resp: Any) -> str | None:
    """Extract a task status string from a raw poll response.

    Handles both dict responses and ``httpx.Response`` objects (the
    identity resolver establishes that ``_request`` returns the latter),
    checking the common ``status`` / ``state`` keys.

    Args:
        resp: The raw response from ``client._request("GET", ...)``.

    Returns:
        The status string, or ``None`` when it cannot be determined.
    """
    if hasattr(resp, "json"):
        resp = resp.json()
    if isinstance(resp, dict):
        status = resp.get("status") or resp.get("state")
        return str(status) if status else None
    return None


def _guess_extension(media_type: str) -> str:
    """Guess a file extension from a media type.

    Args:
        media_type: The MIME type (e.g. ``"image/png"``).

    Returns:
        A file extension string (e.g. ``"png"``).
    """
    ext_map = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/svg+xml": "svg",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/ogg": "ogg",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "application/pdf": "pdf",
    }
    return ext_map.get(media_type, "bin")
