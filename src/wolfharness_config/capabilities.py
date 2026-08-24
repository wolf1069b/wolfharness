"""Capability configuration models.

Typed config models for each of the 6 built-in capabilities, forming a
discriminated union that validates YAML inputs at load time.

Three resolution paths:

1. **Built-in short names** (``loop_detection``, ``token_budget``, etc.)
   validated against typed config models.
2. **Entry-point names** (e.g. ``mermaid_lint``) resolved via the
   ``wolfharness.capabilities`` entry-point group using
   :class:`EntryPointCapabilityConfig`.
3. **Python import paths** (e.g. ``pydantic_ai.capabilities.Instrumentation``)
   resolved via :class:`GenericCapabilityConfig` using ``__import__``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


KNOWN_CAPABILITY_TYPES: frozenset[str] = frozenset({
    "loop_detection",
    "token_budget",
    "tool_output_budget",
    "dcp",
    "skill_activation",
    "memory",
    "modality_filter",
    "viking",
    "tool_arg_sanitize",
})

IMPORT_MAP: dict[str, str] = {
    "loop_detection": "wolfharness.capabilities.loop_detection.LoopDetectionCapability",
    "token_budget": "wolfharness.capabilities.token_budget.TokenBudgetCapability",
    "tool_output_budget": (
        "wolfharness.capabilities.tool_output_budget.ToolOutputBudgetCapability"
    ),
    "dcp": "wolfharness.capabilities.dcp.DynamicContextPruningCapability",
    "skill_activation": "wolfharness.capabilities.skill_manager_cap:SkillManagerCap",
    "memory": "wolfharness.capabilities.memory.MemoryCapability",
    "modality_filter": "wolfharness.capabilities.modality_filter.ModalityFilterCapability",
    "viking": "wolfharness.capabilities.viking.VikingCapability",
    "tool_arg_sanitize": ("wolfharness.capabilities.tool_arg_sanitize.ToolArgSanitizeCapability"),
}


# ---------------------------------------------------------------------------
# Typed config models for built-in capabilities
# ---------------------------------------------------------------------------


class LoopDetectionCapabilityConfig(BaseModel):
    """Config for ``LoopDetectionCapability``."""

    type: Literal["loop_detection"] = "loop_detection"
    max_depth: int = 10
    """Maximum delegation depth before raising ``LoopDetectionError``."""


class TokenBudgetCapabilityConfig(BaseModel):
    """Config for ``TokenBudgetCapability``."""

    type: Literal["token_budget"] = "token_budget"
    max_tokens: int = 100_000
    """Maximum cumulative token usage per agent run."""


class ToolOutputBudgetCapabilityConfig(BaseModel):
    """Config for ``ToolOutputBudgetCapability``."""

    type: Literal["tool_output_budget"] = "tool_output_budget"
    max_output_chars: int = 10_000
    """Maximum characters per tool output before truncation."""
    truncation_suffix: str = "\n[Tool output truncated by ToolOutputBudgetCapability]"
    """Suffix appended to truncated tool output to indicate truncation."""


class DCPCapabilityConfig(BaseModel):
    """Config for ``DynamicContextPruningCapability`` (DCP).

    DCP provides dynamic context pruning to manage context window usage
    by tracking watermarks, deduplicating tool outputs, and injecting
    nudges to steer the model toward context-efficient behavior.
    """

    type: Literal["dcp"] = "dcp"
    enabled: bool = True
    """Master switch — when ``False``, DCP is loaded but performs no pruning."""
    expose_tools: bool = True
    """Whether to expose DCP meta-tools (e.g. ``dcp_status``) to the model."""
    max_context_tokens: int = 128_000
    """Maximum context window size in tokens used for watermark calculations."""
    info_threshold: float = 0.60
    """Context fill ratio that triggers the ``info`` watermark level."""
    warning_threshold: float = 0.75
    """Context fill ratio that triggers the ``warning`` watermark level."""
    critical_threshold: float = 0.90
    """Context fill ratio that triggers the ``critical`` watermark level."""
    auto_dedup: bool = True
    """Automatically deduplicate repeated tool outputs."""
    auto_strategy_threshold: str = "info"
    """Watermark level at which automatic pruning strategies activate.

    One of ``"info"``, ``"warning"``, ``"critical"``.
    """
    purge_error_steps: int = 3
    """Number of error steps to retain before purging older ones."""
    nudge_turn_frequency: int = 3
    """Inject a context nudge every N turns (0 disables turn-based nudges)."""
    nudge_step_frequency: int = 0
    """Inject a context nudge every N tool-call steps (0 disables step-based nudges)."""
    nudge_role: Literal["system", "user"] = "user"
    """Message role for injected nudges."""
    nudge_visible: bool = True
    """Whether nudge messages are visible to the frontend via EventBus."""
    inject_role: Literal["system", "user"] = "user"
    """Message role for injected pruning notifications."""
    clear_thinking_enabled: bool = True
    """Whether to clear thinking/reasoning blocks from older messages."""
    meta_tool_retention: int = 1
    """Number of recent meta-tool invocations to retain in context."""
    step_protection: int = 2
    """Number of recent steps protected from pruning."""
    protected_tool_patterns: list[str] = Field(default_factory=list)
    """Glob patterns matching tool names that should never be pruned."""
    protected_tools: set[str] = Field(default_factory=set)
    """Literal tool names that should never be pruned."""


class SkillActivationCapabilityConfig(BaseModel):
    """Config for ``SkillActivationCapability``."""

    type: Literal["skill_activation"] = "skill_activation"


class MemoryCapabilityConfig(BaseModel):
    """Config for ``MemoryCapability``."""

    type: Literal["memory"] = "memory"


class ModalityFilterCapabilityConfig(BaseModel):
    """Config for ``ModalityFilterCapability``.

    Provides per-modality degradation strategies for models that lack
    certain input modalities. When a tool returns content the model
    cannot process (e.g. image for a text-only model), the capability
    degrades it according to the configured strategy.
    """

    type: Literal["modality_filter"] = "modality_filter"
    image_strategy: Literal["describe", "reference", "drop", "pass", "understand"] = "describe"
    """Degradation strategy for unsupported image content.

    ``"understand"`` replaces the image with a real text description
    produced by a vision LLM (see ``vision_model``). When no
    ``vision_model`` is configured, ``"understand"`` falls back to
    ``"describe"`` at runtime.
    """
    audio_strategy: Literal["describe", "reference", "drop", "pass"] = "describe"
    """Degradation strategy for unsupported audio content."""
    video_strategy: Literal["describe", "reference", "drop", "pass"] = "describe"
    """Degradation strategy for unsupported video content."""
    document_strategy: Literal["describe", "reference", "drop", "pass"] = "describe"
    """Degradation strategy for unsupported document content."""
    vision_model: str | None = None
    """Vision model used by the ``"understand"`` image strategy.

    Either a model variant name (resolved via the manifest) or a
    namespaced string such as ``"openai:gpt-4o"`` (resolved via
    ``infer_model``). When ``None`` and ``image_strategy ==
    "understand"``, the strategy falls back to ``"describe"`` at runtime.
    """


class ToolArgSanitizeCapabilityConfig(BaseModel):
    """Config for ``ToolArgSanitizeCapability``.

    Sanitizes invalid-JSON tool call arguments in message history before
    every model request. Some models (e.g. deepseek-v4-flash) occasionally
    emit tool call arguments that are not valid JSON; the provider rejects
    the poisoned history with HTTP 400 on the next request. This capability
    replaces such arguments with ``{}`` so bad JSON never reaches the provider.
    """

    type: Literal["tool_arg_sanitize"] = "tool_arg_sanitize"
    enabled: bool = True
    """Master switch. Set to ``false`` to observe without sanitizing."""


class VikingCapabilityConfig(BaseModel):
    """Config for ``VikingCapability``."""

    type: Literal["viking"] = "viking"
    mode: Literal["retrieve", "write", "graph", "all"] = "all"
    """Tool exposure mode — retrieve (7 tools), write (6 tools), graph (2 tools), all (15 tools)."""
    url: str | None = None
    """Viking server URL. If None, SDK resolves from OPENVIKING_URL env var
    or ~/.openviking/ovcli.conf."""
    api_key: str | None = None
    """Viking API key. If None, SDK resolves from env vars."""
    account: str | None = None
    """Viking account ID. If None, SDK resolves from env vars."""
    user: str | None = None
    """Viking user ID. If None, SDK resolves from env vars."""
    timeout: float | None = None
    """Request timeout in seconds. If None, SDK uses default (60s)."""
    skills_uri: str | None = None
    """Override for skills URI. Default: viking://user/{user or 'default'}/skills/"""
    resources_uri: str | None = None
    """Override for resources URI."""
    sessions_uri: str | None = None
    """Override for sessions URI. Default: viking://user/{user}/sessions/"""
    multimodal_bridge: bool = False
    """Enable multimodal bridge (Phase 6, not yet implemented)."""
    support_vision: bool | None = None
    """Result of viking_read for image URIs.

    Tri-state control over how image resources are returned to the model:

    - ``True`` — return image bytes (``BinaryImage``) regardless of model.
    - ``False`` — return a text URI description, never image bytes.
    - ``None`` (default) — auto-detect from resolved model capabilities
      (``image_input``); text-only when unknown.

    When forcing ``True`` on a model that does not actually accept image
    input, configure ``type: modality_filter`` as a safety net so the
    image is degraded before reaching the model API.
    """
    uploads_uri: str | None = None
    """Override for uploads URI."""
    public_download_base_url: str | None = None
    """Base URL for public download links."""
    enable_link: bool = False
    """Enable viking_link tool. Requires backend graph link API support.
    Disabled by default — not all Viking deployments support linking."""
    enable_memory: bool = False
    """Enable viking_remember and viking_recall tools. Requires backend
    session-based memory support. Disabled by default — not all Viking
    deployments support memory sessions."""
    resource_file_extensions: tuple[str, ...] = (
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".html",
    )
    """File extensions to include in list_resources(). Files with extensions
    not in this set are skipped. Set to an empty list to include all files."""
    resource_read_level: Literal["abstract", "overview", "read"] = "overview"
    """Default content level for read_resource() (ResourceAccess Protocol).
    "abstract" (L0), "overview" (L1, default), or "read" (L2, full)."""
    auto_resolve_identity: bool = True
    """When True (default), resolve account_id and user_id automatically
    from the API key or /health endpoint after client initialization."""
    memories_uri: str | None = None
    """Override for memories URI. Default: viking://user/{user_id}/memories/"""
    actor_peer_id: str | None = None
    """Explicit actor peer ID for multi-agent isolation. When None (default),
    the Viking server uses user_id for isolation. When set, passed to the
    SDK client for all requests."""
    auto_ingest_enabled: bool = False
    """Enable automatic conversation ingestion. When True, the capability
    ingests the previous turn's conversation at the start of the next
    ``before_model_request`` call (lazy ingestion)."""
    auto_ingest_mode: Literal["async", "sync"] = "async"
    """Ingestion mode — ``"async"`` (fire-and-forget background task) or
    ``"sync"`` (block until ingestion completes). Default is ``"async"``."""
    auto_ingest_sanitize: bool = True
    """Strip ``<openviking-recall>`` and ``<openviking-profile>`` XML blocks
    from messages before ingestion to prevent feedback loops."""
    auto_ingest_source_type: str = "wolfharness"
    """Source type metadata for ingested sessions."""
    auto_ingest_keep_recent_turns: int = 0
    """Number of recent turns to retain in the session after commit.
    When 0 (default), no retention parameter is passed to commit_session."""
    auto_recall_enabled: bool = False
    """When True, perform semantic recall before each model request using the
    latest user prompt as the query. Results are injected as an
    <openviking-recall> XML block into the system prompt."""
    auto_recall_method: Literal["search", "find"] = "search"
    """Recall retrieval method: "search" (default, session-aware, calls
    client.search() with session_id) or "find" (faster, deduplicated, calls
    client.find() without session context)."""
    auto_recall_max_tokens: int = 2000
    """Maximum token budget for the injected recall block. Content exceeding
    this budget is truncated with a [... truncated] indicator."""
    auto_recall_limit: int = 10
    """Maximum number of results to request from the Viking server per recall."""
    auto_recall_min_score: float = 0.3
    """Minimum composite score for a recall hit to be included in the result."""
    auto_recall_lexical_boost: float = 0.1
    """Score boost per overlapping word between the query and hit content."""
    auto_recall_category_boost: float = 0.05
    """Score boost applied to hits with context_type="memory"."""
    auto_recall_context_types: list[str] = Field(default_factory=lambda: ["memory", "resource"])
    """Context types to include in recall results. Hits with context_type not
    in this list are filtered out before ranking."""
    enable_forget: bool = False
    """Enable the viking_forget tool. This is a destructive operation that
    removes documents from the Viking knowledge graph. Disabled by default —
    an agent deleting memories without user confirmation is dangerous.
    Independent from enable_memory."""
    uri_guard_enabled: bool = False
    """When True, block file-access tools (read, bash, grep, glob) from
    accessing viking:// URIs in their arguments. Forces the agent to use
    dedicated Viking tools (viking_read, viking_search) instead."""
    uri_guard_protected_tools: list[str] = Field(
        default_factory=lambda: ["read", "bash", "grep", "glob"]
    )
    """Tool names protected by the URI guard. When uri_guard_enabled is True,
    these tools are blocked from accessing viking:// URIs. Customize to add
    or remove tools from the protected list."""
    allowed_uri_prefixes: list[str] = Field(default_factory=list)
    """URI prefix allowlist covering all viking:// namespaces. When
    non-empty:
    - knowledge-base access (all ``viking_*`` tools + the @-mention flow)
      rejects URIs outside the listed prefixes;
    - memory paths — auto-recall, profile injection, compaction, and
      multimodal-bridge uploads — are only active when their target URI
      (e.g. ``viking://user/{user}/memories/``) is inside the list;
    - skill discovery (``list_skills``/``read_skill``/``skill_exists``) is
      only active when the skills URI is inside the list.
    Since one list governs everything, include both the intended resource
    prefixes and the memory/skill prefixes when those features are needed.
    Empty list (default) means unrestricted — backward compatible."""
    compaction_enabled: bool = False
    """When True, archive old conversation messages to Viking before context
    overflow. Disabled by default."""
    compaction_threshold: int = 100_000
    """Estimated token count above which compaction is triggered. Only
    checked when compaction_enabled is True."""
    compaction_keep_recent_turns: int = 5
    """Number of recent turns to keep when compacting. Older messages are
    archived to viking://user/{user_id}/memories/compacted/."""
    compaction_expand_tool: bool = True
    """When True (and compaction_enabled is True), expose a viking_expand
    tool that loads the full content of a previously archived conversation."""
    profile_enabled: bool = False
    """Enable first-turn profile injection from Viking memories. When True,
    the capability queries Viking for memory search results on the first
    turn and injects them as an <openviking-profile> XML block."""
    profile_max_tokens: int = 1000
    """Maximum token budget for the injected profile block. Content exceeding
    this budget is truncated with a [... truncated] indicator."""
    profile_limit: int = 5
    """Maximum number of memory hits to retrieve for the profile block."""
    profile_first_turn_only: bool = True
    """When True (default), profile injection runs only on the first turn
    of a session (message count <= 2). When False, injection runs on every
    before_model_request call where _profile_injected is False."""
    enabled_tools: list[str] | None = Field(
        default=None,
        examples=[["viking_ls", "viking_read", "viking_grep"]],
        title="Enabled tools",
    )
    """If set, only these tools will be available (whitelist).
    Mutually exclusive with disabled_tools."""

    disabled_tools: list[str] | None = Field(
        default=None,
        examples=[["viking_search", "viking_find"]],
        title="Disabled tools",
    )
    """Tools to exclude from this capability (blacklist). Mutually exclusive
    with enabled_tools. For example, disable a slow knowledge-graph semantic
    search backend while keeping the deterministic tools:
    ``disabled_tools: ["viking_search", "viking_find"]``."""

    @model_validator(mode="after")
    def _validate_tool_filters(self) -> Self:
        """Validate that enabled_tools and disabled_tools are mutually exclusive."""
        if self.enabled_tools is not None and self.disabled_tools is not None:
            raise ValueError("Cannot specify both 'enabled_tools' and 'disabled_tools'")
        return self


# ---------------------------------------------------------------------------
# Entry-point-based config
# ---------------------------------------------------------------------------


class EntryPointCapabilityConfig(BaseModel):
    """Configuration for a capability loaded via entry-point name.

    Used when ``type`` matches a name registered in the
    ``wolfharness.capabilities`` entry-point group (e.g. ``"mermaid_lint"``).
    The entry-point registry maps short names to capability classes,
    enabling external packages to register capabilities without requiring
    users to know the full Python import path.
    """

    type: str
    """Entry-point name registered under ``wolfharness.capabilities``."""

    args: dict[str, Any] = Field(default_factory=dict)
    """Arguments to pass to the capability constructor."""

    def build(self) -> Any:
        """Resolve the entry-point name and instantiate the capability.

        Uses :mod:`importlib.metadata` directly to discover entry points
        in the ``wolfharness.capabilities`` group, avoiding a dependency
        on the :mod:`wolfharness` package (which would violate the
        import-linter contract that ``wolfharness_config`` must not import
        from ``wolfharness``).

        Returns:
            Instantiated capability object.

        Raises:
            ValueError: If the entry-point name is not registered.
            ImportError: If the capability class cannot be loaded.
        """
        from importlib.metadata import entry_points

        eps = entry_points(group="wolfharness.capabilities")
        for ep in eps:
            if ep.name == self.type:
                cls = ep.load()
                return cls(**self.args)

        available = sorted({ep.name for ep in eps})
        msg = (
            f"Unknown entry-point capability: {self.type!r}. "
            f"Available: {', '.join(available) if available else '(none)'}"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Generic / import-path-based config (backward compatible)
# ---------------------------------------------------------------------------


class GenericCapabilityConfig(BaseModel):
    """Configuration for a pydantic-ai capability loaded from YAML via import path.

    Used when ``type`` is a Python import path (e.g.
    ``'pydantic_ai.capabilities.Instrumentation'``) rather than a short name.
    """

    type: str
    """Import path to the capability class."""

    args: dict[str, Any] = Field(default_factory=dict)
    """Arguments to pass to the capability constructor."""

    def build(self) -> Any:
        """Import and instantiate the capability.

        Returns:
            Instantiated capability object.

        Raises:
            ImportError: If the module cannot be imported.
            ValueError: If the type path is invalid or the class not found.
        """
        try:
            module_path, class_name = self.type.rsplit(".", 1)
        except ValueError:
            msg = f"Invalid capability type path: {self.type!r}"
            raise ValueError(msg) from None

        try:
            module = __import__(module_path, fromlist=[class_name])
        except ImportError as e:
            msg = f"Cannot import module for capability {self.type!r}: {e}"
            raise ImportError(msg) from e

        try:
            cls = getattr(module, class_name)
        except AttributeError:
            msg = f"Class {class_name!r} not found in module {module_path!r}"
            raise ValueError(msg) from None

        return cls(**self.args)


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


BuiltinCapabilityConfig = Annotated[
    LoopDetectionCapabilityConfig
    | TokenBudgetCapabilityConfig
    | ToolOutputBudgetCapabilityConfig
    | DCPCapabilityConfig
    | SkillActivationCapabilityConfig
    | MemoryCapabilityConfig
    | ModalityFilterCapabilityConfig
    | ToolArgSanitizeCapabilityConfig
    | VikingCapabilityConfig,
    Field(discriminator="type"),
]

CapabilityConfig = BuiltinCapabilityConfig | EntryPointCapabilityConfig | GenericCapabilityConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_known_capability_type(raw_type: str) -> bool:
    """Check if a type string is a known short capability name.

    Args:
        raw_type: The ``type`` field value from a YAML dict.

    Returns:
        ``True`` if it's a known short name (``"loop_detection"``, etc.).
    """
    return raw_type in KNOWN_CAPABILITY_TYPES


def build_config_capabilities(capabilities: list[Any]) -> list[Any]:
    """Build capability instances from a config capabilities list.

    Handles three item types:
    - ``GenericCapabilityConfig`` / ``EntryPointCapabilityConfig`` → ``.build()``
    - Other Pydantic ``BaseModel`` (typed built-in configs) → ``build_capability()``
    - Pre-instantiated objects → used as-is

    This is the single canonical builder used by:
    - ``NativeAgent.__init__()`` for eager cap building
    - ``AgentFactory._compile_agent_capabilities()`` step 3b
    - ``AgentFactory.register_config_capabilities()`` for pool-init registration

    Args:
        capabilities: List from a ``NativeAgentConfig.capabilities`` field.
            Items may be ``None`` (skipped), config models, or pre-instantiated
            ``AbstractCapability`` instances.

    Returns:
        List of instantiated capability objects.
    """
    from pydantic import BaseModel

    built: list[Any] = []
    for cap in capabilities:
        if cap is None:
            continue
        if isinstance(cap, BaseModel):
            from typing import cast as _cast

            built.append(build_capability(_cast("CapabilityConfig", cap)))
        else:
            # Pre-instantiated AbstractCapability
            built.append(cap)
    return built


def build_capability(config: CapabilityConfig) -> Any:  # noqa: PLR0911, RET503
    """Build a capability from any config variant.

    For typed built-in configs, imports the corresponding capability class
    and constructs it with the config's fields. For generic configs, uses
    ``GenericCapabilityConfig.build()``.

    Args:
        config: A validated capability config (built-in or generic).

    Returns:
        An instantiated pydantic-ai ``AbstractCapability``.

    Raises:
        ImportError: If the module cannot be imported.
        ValueError: If the type is unknown or the class not found.
    """
    match config:
        case GenericCapabilityConfig():
            return config.build()
        case EntryPointCapabilityConfig():
            return config.build()
        case LoopDetectionCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["loop_detection"], config)
        case TokenBudgetCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["token_budget"], config)
        case ToolOutputBudgetCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["tool_output_budget"], config)
        case DCPCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["dcp"], config)
        case SkillActivationCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["skill_activation"], config)
        case MemoryCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["memory"], config)
        case ModalityFilterCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["modality_filter"], config)
        case ToolArgSanitizeCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["tool_arg_sanitize"], config)
        case VikingCapabilityConfig():
            return _import_and_instantiate(IMPORT_MAP["viking"], config)
        case _ as unreachable:
            from typing import assert_never

            assert_never(unreachable)


def _import_and_instantiate(import_path: str, config: BaseModel) -> Any:
    """Import a capability class and construct it from a config model.

    Args:
        import_path: The fully qualified import path (module.ClassName).
        config: A typed config model. All fields except ``type`` are passed
            as constructor kwargs.

    Returns:
        An instantiated capability object.

    Raises:
        ImportError: If the module cannot be imported.
        ValueError: If the class is not found.
    """
    try:
        module_path, class_name = import_path.rsplit(".", 1)
    except ValueError:
        msg = f"Invalid import path: {import_path!r}"
        raise ValueError(msg) from None

    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as e:
        msg = f"Cannot import module for capability {import_path!r}: {e}"
        raise ImportError(msg) from e

    try:
        cls = getattr(module, class_name)
    except AttributeError:
        msg = f"Class {class_name!r} not found in module {module_path!r}"
        raise ValueError(msg) from None

    # Pass all fields except "type" as constructor kwargs
    kwargs = {k: v for k, v in config.model_dump(exclude={"type"}).items() if v is not None}
    return cls(**kwargs)
