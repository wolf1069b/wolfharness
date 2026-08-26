"""SkillManagerCap — unified skill management as a pydantic-ai capability.

Replaces the deleted per-skill ``SkillCapability`` class and
:class:`~wolfharness.capabilities.skill_activation.SkillActivationCapability`
with a single capability that:

- Holds local skills as ``dict[str, Skill]`` (no per-skill capability wrappers).
- Queries child :class:`~wolfharness.capabilities.mcp_server_cap.McpServerCap`
  instances for remote skills and commands.
- Provides metadata-only instructions by default (``<available-skills>`` XML).
- Supports optional ``matcher_fn`` for dynamic per-turn skill injection.
- Supports ``always_active`` flag for skills that bypass the matcher.
- Aggregates ``SkillResource`` and ``CommandResource`` from local + remote.
- Imports per-skill Python tools via
  :class:`~wolfharness.skills.skill_tool_manager.SkillToolManager`.
- Creates per-skill :class:`~wolfharness.capabilities.mcp_server_cap.McpServerCap` instances.
- **Owns** the ``load_skill`` and ``list_skills`` agent-facing tools (D1).
- Applies ``allowed_tools`` filtering via ``get_wrapper_toolset()``.
- Guards against built-in tool name collisions (D10).
- Inherits change stream merging and lifecycle from
  :class:`~wolfharness.capabilities.combined_toolset.CombinedToolsetCapability`.
"""

from __future__ import annotations

import contextlib
import html
import inspect
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import (
    AbstractToolset,
    CombinedToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
)

from wolfharness.capabilities.combined_toolset import CombinedToolsetCapability
from wolfharness.capabilities.resource_protocols import (
    BlobResourceContent,
    ChangeObservable,
    CommandEntry,
    CommandResource,
    ResourceAccess,
    ResourceEntry,
    SkillEntry,
    SkillResource,
    TextResourceContent,
)
from wolfharness.log import get_logger
from wolfharness.skills.skill import Skill
from wolfharness.skills.skill_tool_manager import SkillToolManager
from wolfharness.skills.uri_resolver import ResolvedSkillURI, _name_alternatives


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from types import TracebackType

    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.capabilities.abstract import AgentInstructions  # type: ignore[attr-defined]

    from wolfharness.capabilities.agent_context import AgentContextDeps
    from wolfharness.capabilities.mcp_server_cap import McpServerCap
    from wolfharness.delegation.pool import AgentPool
    from wolfharness.skills.uri_resolver import SkillURIResolver
    from wolfharness.tools.base import Tool


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper functions (moved from wolfharness_toolsets/builtin/skills.py)
# ---------------------------------------------------------------------------


def _substitute_arguments(instructions: str, arguments: str | None) -> str:
    """Substitute argument placeholders in skill instructions.

    Supports:
    - $1, $2, ... - Nth argument
    - $@ - All arguments
    - $ARGUMENTS - All arguments

    Args:
        instructions: The skill instructions to process
        arguments: Space-separated arguments string

    Returns:
        Instructions with placeholders replaced
    """
    if arguments is None:
        return instructions

    args_list = arguments.split() if arguments else []

    # Replace positional arguments $1, $2, etc.
    for i, arg in enumerate(args_list, start=1):
        instructions = instructions.replace(f"${i}", arg)

    # Replace $@ and $ARGUMENTS with all arguments
    all_args = arguments if arguments else ""
    return instructions.replace("$@", all_args).replace("$ARGUMENTS", all_args)


async def _load_reference_content(
    skill: Skill, reference_path: str, pool: AgentPool[Any] | None = None
) -> str:
    """Load content from a skill reference file.

    Args:
        skill: The skill instance
        reference_path: Path to the reference file within the skill directory
        pool: Optional AgentPool for accessing remote skill providers

    Returns:
        The reference content with a header, or empty string if not found

    Raises:
        ReferenceNotFoundError: If the reference file cannot be loaded.
        SecurityError: If the reference path is a path traversal attempt.
    """
    from wolfharness.skills.exceptions import ReferenceNotFoundError

    # For virtual paths (PurePosixPath like skill:// URIs), route through
    # the cap's children (SkillResource providers) for remote reference reads.
    # Use exact type check (not isinstance) to avoid catching UPath subclasses.
    if type(skill.skill_path) is PurePosixPath:
        # Virtual/remote skills: try reading via pool's skill_resolver children.
        # The old pool.skill_provider.read_reference() path is removed;
        # remote skills typically don't have filesystem reference files.
        if pool is not None and pool.skill_resolver is not None:
            # Attempt to resolve via the resource_resolver's _resolve_skill_reference
            # which iterates SkillResource providers for UPath skills only.
            # Virtual skills with PurePosixPath are skipped there, so we
            # fall through to ReferenceNotFoundError.
            pass
        raise ReferenceNotFoundError(
            f"Cannot load reference {reference_path}: virtual/remote skills "
            "do not support filesystem reference reads"
        )

    # For filesystem paths (UPath), load from disk
    from upathtools import UPath

    skill_path = cast(UPath, skill.skill_path)

    # Validate reference_path to prevent path traversal attacks
    from wolfharness.skills.exceptions import SecurityError

    decoded_path = reference_path
    # Check for path traversal attempts and absolute paths
    if ".." in decoded_path.split("/") or decoded_path.startswith("/"):
        raise SecurityError(f"Path traversal detected in reference path: {reference_path}")

    ref_file = skill_path / reference_path
    # Resolve and verify the path is within the skill directory
    try:
        resolved_path = ref_file.resolve()
        resolved_skill_path = skill_path.resolve()
        if not str(resolved_path).startswith(str(resolved_skill_path)):
            raise SecurityError(f"Reference path escapes skill directory: {reference_path}")
    except (OSError, ValueError) as e:
        raise ReferenceNotFoundError(f"Invalid reference path: {reference_path}") from e

    if not ref_file.exists():
        raise ReferenceNotFoundError(str(ref_file))

    content = ref_file.read_text(encoding="utf-8")
    return f"\n\n## Reference: {reference_path}\n\n{content}"


def _is_skill_visible_to_node(
    pool: AgentPool[Any] | None, skill: Skill, node_name: str | None
) -> bool:
    """Check if a skill is visible to a given node's package scope."""
    if pool is None:
        return True
    return pool.is_skill_visible_to_node(skill, node_name)


def _visible_model_skills(
    pool: AgentPool[Any] | None,
    skills: list[Skill],
    node_name: str | None,
) -> list[Skill]:
    """Filter skills to those visible to a node and not disabled for model invocation."""
    return [
        skill
        for skill in skills
        if not getattr(skill, "disable_model_invocation", False)
        and _is_skill_visible_to_node(pool, skill, node_name)
    ]


# ---------------------------------------------------------------------------
# SkillManagerCap
# ---------------------------------------------------------------------------


class SkillManagerCap(
    CombinedToolsetCapability[AgentDepsT],
    SkillResource,
    CommandResource,
    ResourceAccess,
    ChangeObservable,
):
    """Unified skill management capability.

    Holds local skills directly as ``dict[str, Skill]`` and queries child
    ``McpServerCap`` instances for remote skills/commands. Provides
    metadata-only instructions by default, with optional ``matcher_fn``
    for dynamic per-turn injection.

    Owns the ``load_skill`` and ``list_skills`` agent-facing tools (D1),
    eliminating the need for a separate ``SkillsTools`` toolset.

    Attributes:
        _local_skills: Local skills keyed by name.
        _children: Child ``McpServerCap`` instances for remote access.
        _matcher_fn: Optional callable for skill selection.
        _always_active: Set of skill names that always inject.
        _inject_mode: Instruction injection mode (description/matcher/all).
    """

    @property
    def owned_schemes(self) -> frozenset[str]:
        """URI schemes this provider authoritatively handles.

        SkillManagerCap owns the ``skill`` URI scheme exclusively.

        Returns:
            ``frozenset({"skill"})``.
        """
        return frozenset({"skill"})

    def __init__(
        self,
        local_skills: dict[str, Skill] | None = None,
        children: list[AbstractCapability[AgentDepsT]] | None = None,
        *,
        matcher_fn: Callable[..., list[str]] | None = None,
        always_active: set[str] | None = None,
        registry: Any | None = None,
        name: str | None = None,
        tool_manager: SkillToolManager | None = None,
        inject_mode: str = "description",
    ) -> None:
        """Initialize the skill manager capability.

        Args:
            local_skills: Local skills keyed by name. Defaults to empty.
            children: Child ``McpServerCap`` instances for remote skills/commands.
            matcher_fn: Optional async or sync callable that receives the
                conversation context and returns a list of skill names to
                inject. Only used when ``inject_mode`` is ``"matcher"``.
                When ``None``, ``"matcher"`` falls back to ``description``.
            always_active: Set of skill names that always have their instructions
                injected, bypassing the matcher (matcher mode only).
            registry: Optional ``SkillsRegistry`` reference for hot-reload.
            name: Optional name override.
            tool_manager: Optional ``SkillToolManager`` for importing Python tools
                declared in skill frontmatter. When provided, tools are imported
                eagerly at construction time.
            inject_mode: Instruction injection mode — one of ``"description"``
                (default, catalog only), ``"matcher"``, or ``"all"``.
        """
        self._local_skills: dict[str, Skill] = dict(local_skills) if local_skills else {}
        self._children: list[AbstractCapability[AgentDepsT]] = list(children) if children else []
        self._matcher_fn = matcher_fn
        self._always_active: set[str] = set(always_active) if always_active else set()
        self._inject_mode = inject_mode
        self._registry = registry
        self._tool_manager: SkillToolManager | None = tool_manager

        # Per-skill Python tools: {skill_name: [Tool, ...]}
        self._skill_tools: dict[str, list[Tool]] = {}
        # Per-skill McpServerCap children: {skill_name: [McpServerCap, ...]}
        self._skill_mcp_children: dict[str, list[McpServerCap]] = {}

        # Import Python tools eagerly (D2).
        if self._tool_manager is not None:
            self._import_skill_tools()

        # Create per-skill McpServerCap instances (D3).
        self._create_skill_mcp_children()

        # Build the full children list: original children + skill MCP children.
        all_children: list[AbstractCapability[AgentDepsT]] = list(self._children)
        for caps in self._skill_mcp_children.values():
            all_children.extend(caps)

        # Initialize CombinedToolsetCapability with all child capabilities.
        super().__init__(all_children, name=name or "skill-manager")

    # ---- Properties ----

    @property
    def local_skills(self) -> dict[str, Skill]:
        """Return the local skills dict."""
        return self._local_skills

    @property
    def children(self) -> list[AbstractCapability[AgentDepsT]]:
        """Return the child capability list."""
        return list(self._children)

    def add_child(self, child: AbstractCapability[AgentDepsT]) -> None:
        """Add a child capability at runtime.

        Args:
            child: The capability to add.
        """
        self._children.append(child)
        self._capabilities.append(child)

    def remove_child(self, child: AbstractCapability[AgentDepsT]) -> bool:
        """Remove a child capability at runtime.

        Args:
            child: The capability to remove.

        Returns:
            ``True`` if the child was found and removed, ``False`` otherwise.
        """
        removed = False
        if child in self._children:
            self._children.remove(child)
            removed = True
        if child in self._capabilities:
            self._capabilities.remove(child)
            removed = True
        return removed

    def add_local_skill(self, skill: Skill) -> None:
        """Add a local skill.

        Args:
            skill: The Skill to add.
        """
        self._local_skills[skill.name] = skill

    # ---- Pool resolution from RunContext ----

    @staticmethod
    def _resolve_agent_context(ctx: RunContext[AgentDepsT]) -> AgentContextDeps:
        """Extract the ``AgentContextDeps`` from the run context deps.

        Delegates to the shared ``resolve_agent_context_from_deps`` utility
        which handles both the production path (``RuntimeAgentContext.data``)
        and the test path (direct ``AgentContextDeps``).

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            The ``AgentContextDeps`` instance from ``ctx.deps``.

        Raises:
            RuntimeError: If deps is None or AgentContextDeps is not found.
        """
        from wolfharness.capabilities.agent_context import resolve_agent_context_from_deps

        return resolve_agent_context_from_deps(ctx.deps, capability_name="SkillManagerCap")

    @staticmethod
    def _resolve_pool(ctx: RunContext[AgentDepsT]) -> tuple[AgentPool[Any] | None, str | None]:
        """Extract the ``AgentPool`` and current node name from the run context deps.

        In production, ``ctx.deps`` is ``AgentContext`` (which extends
        ``NodeContext`` and exposes ``.node`` — the agent node carrying a
        ``name``) plus a ``pool`` field. In tests, ``ctx.deps`` may be an
        ``AgentContextDeps`` directly (no pool/node).

        The node name is used for package-scoped skill visibility
        (``pool.is_skill_visible_to_node``); passing ``None`` would resolve
        against the default (host) scope and leak/withhold scoped skills.

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            A tuple of (``AgentPool`` if available else ``None``, node name
            if available else ``None``).
        """
        from wolfharness.agents.context import AgentContext as RuntimeAgentContext

        deps = ctx.deps
        if isinstance(deps, RuntimeAgentContext):
            node = deps.node
            node_name = getattr(node, "name", None)
            return deps.pool, node_name
        return None, None

    # ---- Per-skill tool import (D2) ----

    def _import_skill_tools(self) -> None:
        """Import Python tools from all local skills with ``tools`` frontmatter.

        Iterates ``local_skills``, calls ``tool_manager.import_tools(skill.tools)``
        for skills with non-empty ``tools``, and stores results in
        ``_skill_tools`` keyed by skill name.
        """
        if self._tool_manager is None:
            return
        for name, skill in self._local_skills.items():
            if not skill.tools:
                continue
            try:
                imported = self._tool_manager.import_tools(skill.tools)
            except Exception:
                logger.warning("Failed to import tools for skill %r", name, exc_info=True)
                continue
            if imported:
                self._skill_tools[name] = imported

    # ---- Per-skill MCP children (D3) ----

    def _create_skill_mcp_children(self) -> None:
        """Create ``McpServerCap`` instances for skills with ``mcp_servers`` frontmatter.

        For each skill with non-empty ``mcp_servers``, converts each
        ``SkillMcpServerConfig`` to a ``MCPServerConfig`` via
        ``to_mcp_server_config()``, creates a ``McpServerCap``, and stores
        it in ``_skill_mcp_children[skill_name]``.
        """
        from wolfharness.capabilities.mcp_server_cap import McpServerCap

        for name, skill in self._local_skills.items():
            if not skill.mcp_servers:
                continue
            caps: list[McpServerCap] = []
            for server_name, server_config in skill.mcp_servers.items():
                try:
                    mcp_config = server_config.to_mcp_server_config(f"{name}__{server_name}")
                except (ValueError, TypeError):
                    logger.warning(
                        "Failed to convert MCP server config %r for skill %r",
                        server_name,
                        name,
                        exc_info=True,
                    )
                    continue
                try:
                    cap = McpServerCap(config=mcp_config)
                    caps.append(cap)
                except Exception:
                    logger.warning(
                        "Failed to create McpServerCap for skill %r server %r",
                        name,
                        server_name,
                        exc_info=True,
                    )
            if caps:
                self._skill_mcp_children[name] = caps

    # ---- get_toolset() full override (D2 + D10) ----

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Return a unified toolset with built-in tools and per-skill prefixed tools.

        Fully overrides :meth:`CombinedToolsetCapability.get_toolset` — does
        NOT call ``super().get_toolset()``.

        1. Adds built-in tools ``load_skill`` and ``list_skills`` (non-prefixed)
           as closure functions with rich Annotated parameter descriptions.
        2. Creates ``PrefixedToolset("{skill_name}__tool__")`` for each skill
           with imported Python tools.
        3. Creates ``PrefixedToolset("{skill_name}__mcp__")`` for each
           per-skill ``McpServerCap`` child.
        """
        toolsets: list[AbstractToolset[AgentDepsT]] = []
        cap = self

        # 0. Built-in tools: load_skill and list_skills (non-prefixed).
        # Defined as closures so pydantic-ai sees the correct tool names
        # and Annotated parameter descriptions without conflicting with
        # the SkillResource.list_skills protocol method.

        async def load_skill(
            ctx: RunContext[AgentDepsT],
            skill_name: Annotated[
                str,
                Field(
                    description=(
                        "Skill name or skill:// URI to load. Use a bare name "
                        "(e.g. 'python-expert') or a skill:// URI "
                        "(e.g. 'skill://python-expert'). To load a reference "
                        "file, use 'skill://skill-name/references/file.md'."
                    )
                ),
            ],
            arguments: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional space-separated arguments for substitution. "
                        "Placeholders $1, $2, ... are replaced with the Nth "
                        "argument; $@ and $ARGUMENTS are replaced with all "
                        "arguments."
                    )
                ),
            ] = None,
        ) -> str:
            """Load a skill's full instructions, activating its MCP servers and Python tools.

            Calling this tool loads the complete skill instructions (not the
            truncated resource-surface content) and activates the skill's
            declared MCP servers and Python tools, making them available for
            subsequent tool calls.

            Skill URI format:
            - ``skill://skill-name`` — load by flat URI
            - ``skill://skill-name/references/file.md`` — load a reference file
            - Bare name (e.g. ``python-expert``) — backward compatible

            Argument substitution:
            - ``$1``, ``$2``, ... — replaced with the Nth argument
            - ``$@`` — replaced with all arguments
            - ``$ARGUMENTS`` — replaced with all arguments
            """
            return await cap._load_skill_impl(ctx, skill_name, arguments)

        async def list_skills(
            ctx: RunContext[AgentDepsT],
        ) -> str:
            """List all available skills with descriptions and URI information.

            Returns a formatted list of local and remote (MCP-provided) skills.
            Local skills are listed first, followed by remote skills discovered
            through child MCP capabilities. Remote skills are discovery-only —
            they are not auto-injected into system prompts.
            """
            return await cap._list_skills_impl(ctx)

        builtin_ts: FunctionToolset[AgentDepsT] = FunctionToolset(
            [load_skill, list_skills],
            id=f"{self._name}__builtin",
        )
        toolsets.append(builtin_ts)

        # 1. Python tools: PrefixedToolset per skill.
        for skill_name, tools in self._skill_tools.items():
            pa_tools: list[Any] = [t.to_pydantic_ai() for t in tools]
            if pa_tools:
                toolsets.append(
                    PrefixedToolset(
                        wrapped=FunctionToolset(pa_tools),
                        prefix=f"{skill_name}__tool__",
                    )
                )

        # 2. Per-skill McpServerCap children: PrefixedToolset per skill.
        for skill_name, child_caps in self._skill_mcp_children.items():
            for child in child_caps:
                child_ts = child.get_toolset()
                if child_ts is not None:
                    toolsets.append(
                        PrefixedToolset(
                            wrapped=child_ts,
                            prefix=f"{skill_name}__mcp__",
                        )
                    )

        return CombinedToolset(toolsets=toolsets)

    # ---- get_wrapper_toolset() override (D4) ----

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[AgentDepsT],
    ) -> AbstractToolset[AgentDepsT] | None:
        """Apply composite ``allowed_tools`` filtering across all skills.

        Builds a per-skill filter map from ``parsed_allowed_tools()``. If
        any skill has non-empty ``allowed_tools``, wraps the entire agent
        toolset in a ``FilteredToolset`` with a composite filter function.

        Filter semantics:
        - Non-skill tools (no ``{skill_name}__`` prefix) always pass.
        - Skill tools are checked against that skill's allowed set after
          stripping the prefix.
        """
        skill_filters: dict[str, set[str]] = {}
        for name, skill in self._local_skills.items():
            allowed = skill.parsed_allowed_tools()
            if allowed is not None:
                skill_filters[name] = set(allowed)

        if not skill_filters:
            return None

        def _filter(
            ctx: RunContext[AgentDepsT],
            tool_def: ToolDefinition,
        ) -> bool:
            tool_name = tool_def.name
            for skill_name, allowed_set in skill_filters.items():
                for category in ("tool", "mcp"):
                    prefix = f"{skill_name}__{category}__"
                    if tool_name.startswith(prefix):
                        bare = tool_name[len(prefix) :]
                        return bare in allowed_set
            return True  # Non-skill tools always pass.

        return FilteredToolset(wrapped=toolset, filter_func=_filter)

    # ---- AbstractCapability: instructions ----

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return metadata XML and a dynamic callable for skill content.

        Implements progressive disclosure:

        - **Static metadata** (``<available-skills>`` XML): returned as a
          plain ``str`` so pydantic-ai marks it ``dynamic=False`` — cacheable
          by Anthropic prompt caching and OpenAI prefix caching.
        - **Dynamic skill content** (``<skill_content>`` blocks): returned as
          a callable that receives ``RunContext`` at run time. The callable
          uses ``ctx.messages`` for matcher-based skill selection.
          Pydantic-ai marks callable results ``dynamic=True`` — correct for
          per-turn dynamic content.

        This replaces the old ``before_model_request`` approach that mutated
        ``SystemPromptPart.content`` in-place every turn (which invalidated
        the prefix cache).

        Returns ``None`` when there are no local skills.
        """
        if not self._local_skills:
            return None

        metadata = self._build_metadata_xml()
        return [metadata, self._build_dynamic_skill_content]

    def _build_metadata_xml(self) -> str:
        """Build the static ``<available-skills>`` XML block."""
        lines = ["<available-skills>"]
        for name, skill in self._local_skills.items():
            if skill.disable_model_invocation:
                continue
            desc = html.escape(skill.description)
            lines.append(f'<skill name="{html.escape(name)}" description="{desc}" />')
        lines.append("</available-skills>")
        return "\n".join(lines)

    async def _build_dynamic_skill_content(self, ctx: RunContext[AgentDepsT]) -> str | None:
        """Build ``<skill_content>`` blocks for skills per the inject mode.

        The injection mode (``description``/``matcher``/``all``) governs which
        skills get full instructions injected:

        - ``"description"``: nothing injected (catalog only).
        - ``"all"``: full instructions for every visible local skill.
        - ``"matcher"``: full instructions for skills selected by
          ``_matcher_fn``; falls back to ``description`` (catalog only) with a
          warning when no matcher is configured.
        """
        if not self._local_skills:
            return None

        # description mode → catalog only, no <skill_content> injected.
        if self._inject_mode == "description":
            return None

        messages = ctx.messages

        # matcher mode requires a configured matcher_fn; otherwise fall back
        # to description (catalog only) with a warning.
        if self._inject_mode == "matcher":
            if self._matcher_fn is None:
                logger.warning(
                    "inject=matcher configured but no matcher_fn provided; "
                    "falling back to description mode (catalog only)"
                )
                return None
            sig = inspect.signature(self._matcher_fn)
            if len(sig.parameters) >= 2:  # noqa: PLR2004
                result = self._matcher_fn(messages, list(self._local_skills.keys()))
            else:
                result = self._matcher_fn(messages)
            if inspect.isawaitable(result):
                result = await result
            matched: set[str] = {n for n in result if n in self._local_skills}
            # Always add always_active skills.
            matched |= self._always_active & set(self._local_skills.keys())
        else:
            # "all" mode (and any other): inject every skill.
            matched = set(self._local_skills.keys())

        if not matched:
            return None

        # Build injection text.
        parts: list[str] = []
        for name in sorted(matched):
            skill = self._local_skills[name]
            try:
                instructions = skill.load_instructions()
            except (ValueError, OSError):
                logger.warning("Failed to load instructions for skill %r", name)
                continue
            if instructions:
                escaped_name = html.escape(name)
                parts.append(
                    f'<skill_content name="{escaped_name}">\n{instructions}\n</skill_content>'
                )

        return "\n\n".join(parts) if parts else None

    # ---- ResourceAccess delegation (RFC-0058) ----

    async def list_resources(self) -> Sequence[ResourceEntry]:
        """List resources from all per-skill MCP children.

        Delegates to every per-skill ``McpServerCap`` child implementing
        :class:`ResourceAccess` and aggregates their ``ResourceEntry``
        instances. A failing child is skipped so one broken MCP server
        does not prevent resource listing from the others.

        Returns:
            Aggregated sequence of ``ResourceEntry`` descriptors.
        """
        entries: list[ResourceEntry] = []
        for caps in self._skill_mcp_children.values():
            for cap in caps:
                if isinstance(cap, ResourceAccess):
                    try:
                        entries.extend(await cap.list_resources())
                    except Exception:
                        logger.warning(
                            "Failed to list resources from per-skill MCP child",
                            server=cap.name,
                            exc_info=True,
                        )
                        continue
        return entries

    async def read_resource(
        self, uri: str
    ) -> list[TextResourceContent | BlobResourceContent] | None:
        """Read an MCP resource by URI from per-skill MCP children.

        Tries each per-skill ``McpServerCap`` child implementing
        :class:`ResourceAccess` until one returns a non-``None`` result.
        A failing child is skipped so a broken server does not abort the read.

        Args:
            uri: Resource URI to read.

        Returns:
            List of resource content instances, or ``None`` if no child
            has the resource.
        """
        for caps in self._skill_mcp_children.values():
            for cap in caps:
                if isinstance(cap, ResourceAccess):
                    try:
                        result = await cap.read_resource(uri)
                    except Exception:
                        logger.warning(
                            "Failed to read resource %r from per-skill MCP child",
                            uri,
                            exc_info=True,
                        )
                        continue
                    if result is not None:
                        return result
        return None

    async def resource_exists(self, uri: str) -> bool:
        """Check if a resource URI exists in any per-skill MCP child.

        Args:
            uri: Resource URI to check.

        Returns:
            ``True`` if any per-skill ``McpServerCap`` child reports the
            resource, ``False`` otherwise.
        """
        for caps in self._skill_mcp_children.values():
            for cap in caps:
                if isinstance(cap, ResourceAccess):
                    try:
                        if await cap.resource_exists(uri):
                            return True
                    except Exception:
                        logger.warning(
                            "Failed to check resource %r existence in per-skill MCP child",
                            uri,
                            exc_info=True,
                        )
                        continue
        return False

    @property
    def has_wrap_node_run(self) -> bool:
        """Return False — no node run wrapping needed."""
        return False

    async def for_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> SkillManagerCap[AgentDepsT]:
        """Create a per-run copy of this capability.

        Calls ``for_run()`` on each child capability so children are
        also per-run isolated.

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            A new ``SkillManagerCap`` sharing the same skills but with
            per-run copies of children.
        """
        children_for_run = [await child.for_run(ctx) for child in self._children]
        cap = SkillManagerCap(
            local_skills=self._local_skills,
            children=children_for_run,
            matcher_fn=self._matcher_fn,
            always_active=self._always_active,
            registry=self._registry,
            name=self._name,
            tool_manager=self._tool_manager,
            inject_mode=self._inject_mode,
        )
        return cap  # noqa: RET504

    # ---- Built-in tool implementations (D1 + D9) ----

    async def _load_skill_impl(  # noqa: PLR0911, PLR0915
        self,
        ctx: RunContext[AgentDepsT],
        skill_name: str,
        arguments: str | None = None,
        *,
        node_name: str | None = None,
        include_assembly: bool = True,
    ) -> str:
        """Implementation for the ``load_skill`` agent tool.

        Loads a skill's full instructions with argument substitution and
        reference-file support. Activates MCP servers and Python tools.

        Args:
            ctx: The run context providing agent dependencies.
            skill_name: Skill name or skill:// URI.
            arguments: Optional space-separated arguments for substitution.
            node_name: Optional node name override for package-scoped skill
                visibility. Defaults to the node carried by ``ctx``.
            include_assembly: When False, skip the MCP/tool status rendering
                and tool import (module side effects) entirely — returns pure
                instruction text for instruction-only injection (e.g. team
                member skills). Defaults to True, preserving the agent-facing
                ``load_skill`` tool behavior.
        """
        pool, resolved_node = self._resolve_pool(ctx)
        node_name_effective = node_name if node_name is not None else resolved_node

        # Determine if this is a URI or bare skill name
        is_uri = skill_name.startswith("skill://")

        try:
            resolved = ResolvedSkillURI.parse(skill_name)
        except Exception as e:  # noqa: BLE001
            return f"Invalid skill name or URI {skill_name!r}: {e}"

        if is_uri:
            # URI-based loading via skill_resolver
            resolver: SkillURIResolver | None = pool.skill_resolver if pool is not None else None
            if resolver is None:
                return "Skill URI resolution not available - skill_resolver not configured"

            try:
                skill = await resolver.resolve(skill_name)
            except Exception as e:  # noqa: BLE001
                return f"Failed to resolve skill URI {skill_name!r}: {e}"
            if not _is_skill_visible_to_node(pool, skill, node_name_effective):
                available = await self._available_skill_names(pool, node_name_effective)
                return f"Skill {resolved.skill_name!r} not found. Available skills: {available}"

            # Check for reference path first
            ref_path = skill.resolved_reference_path or resolved.reference_path

            if ref_path:
                # Reference-only loading: skip main SKILL.md content
                try:
                    ref_content = await _load_reference_content(skill, ref_path, pool=pool)
                    instructions = ref_content
                except Exception as e:  # noqa: BLE001
                    return f"Failed to load reference {ref_path!r}: {e}"
            else:
                instructions = skill.instructions or ""
        else:
            # Bare-name loading: check local skills first, then children.
            loaded = await self._load_visible_bare_skill(
                pool, node_name_effective, resolved.skill_name
            )
            if loaded is None:
                available = await self._available_skill_names(pool, node_name_effective)
                return f"Skill {resolved.skill_name!r} not found. Available skills: {available}"
            skill, instructions = loaded

        # Apply argument substitution
        instructions = _substitute_arguments(instructions, arguments)

        # Activate MCP servers and tools declared in the skill.
        # When ``include_assembly`` is False (e.g. instruction-only injection
        # into a team member prompt), skip the MCP/tool status appendix
        # entirely — tool modules are NOT imported (importing runs their
        # module-level side effects, out of scope for prompt injection).
        mcp_lines: list[str] = []
        tool_lines: list[str] = []

        if include_assembly and skill.mcp_servers:
            for server_name, config in skill.mcp_servers.items():
                server_desc = config.command or config.url or "configured"
                mcp_lines.append(f"- `{server_name}`: {server_desc}")

        if include_assembly and skill.tools:
            tool_manager = SkillToolManager()
            for tool_config in skill.tools:
                result = tool_manager.import_tool(tool_config)
                status = "✓" if result is not None else "✗"
                tool_lines.append(f"- `{tool_config.import_path}` ({status})")

        # Determine if this is a reference-only load
        effective_ref_path = skill.resolved_reference_path or (
            resolved.reference_path if is_uri else None
        )
        is_reference_load = is_uri and effective_ref_path is not None

        # Build the response
        if is_reference_load:
            header = f"# {skill.name} → Reference: {effective_ref_path}"
            parts: list[str] = [header]
            parts.append(instructions)
            parts.append(f"Skill URI: {skill.safe_uri}")
        else:
            header = f"# {skill.name}\n\n{skill.description}"
            meta_lines: list[str] = []
            if skill.license:
                meta_lines.append(f"License: {skill.license}")
            if skill.compatibility:
                meta_lines.append(f"Compatibility: {skill.compatibility}")
            meta = "\n".join(meta_lines)
            parts = [header]
            if meta:
                parts.append(meta)
            parts.append(instructions)
            parts.append(f"Skill URI: {skill.safe_uri}")

        # Append activated MCP servers section
        if mcp_lines:
            parts.append("## Activated MCP Servers\n" + "\n".join(mcp_lines))

        # Append activated tools section
        if tool_lines:
            parts.append("## Activated Tools\n" + "\n".join(tool_lines))

        return "\n\n".join(parts)

    async def _list_skills_impl(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> str:
        """Implementation for the ``list_skills`` agent tool.

        Lists local and remote (MCP-provided) skills with descriptions and
        URI information. Remote skills are discovery-only.

        Args:
            ctx: The run context providing agent dependencies.
        """
        pool, node_name = self._resolve_pool(ctx)

        # Get local skills from the cap's own _local_skills
        local_skill_list = list(self._local_skills.values())
        visible_local = _visible_model_skills(pool, local_skill_list, node_name)

        # Get remote skills from child SkillResource providers
        remote_skills: list[Skill] = []
        for child in self._children:
            if not isinstance(child, SkillResource):
                continue
            with contextlib.suppress(Exception):
                entries = await child.list_skills()
                remote_skills.extend(
                    Skill(
                        name=entry.name,
                        description=entry.description,
                        skill_path=PurePosixPath(entry.uri),
                        instructions="",
                    )
                    for entry in entries
                )

        visible_remote = _visible_model_skills(pool, remote_skills, node_name)

        # Deduplicate by name: local skills take priority
        seen: set[str] = {s.name for s in visible_local}
        all_skills = list(visible_local)
        for skill in visible_remote:
            if skill.name not in seen:
                seen.add(skill.name)
                all_skills.append(skill)

        if not all_skills:
            return "No skills available"

        lines = ["Available skills:", ""]

        for skill in all_skills:
            lines.append(f"- **{skill.name}**: {skill.description}")
            lines.append(f"  - URI: `skill://{skill.name}`")

        # Add usage guidance
        lines.append("")
        lines.append("## Usage")
        lines.append("")
        lines.append("Load a skill by name (backward compatible):")
        lines.append("```python")
        lines.append('await load_skill(ctx, "skill-name")')
        lines.append("```")
        lines.append("")
        lines.append("Or use a skill:// URI:")
        lines.append("```python")
        lines.append('await load_skill(ctx, "skill://skill-name")')
        lines.append("```")
        lines.append("")
        lines.append("With arguments for substitution:")
        lines.append("```python")
        lines.append('await load_skill(ctx, "skill://skill-name", "arg1 arg2")')
        lines.append("```")

        return "\n".join(lines)

    # ---- Internal helpers for bare-name skill loading ----

    async def _load_visible_bare_skill(
        self,
        pool: AgentPool[Any] | None,
        node_name: str | None,
        skill_name: str,
    ) -> tuple[Skill, str] | None:
        """Load a bare skill name from local skills or remote children.

        Local skills take precedence. A local skill that exists but is not
        visible to the current node does NOT shadow a matching visible remote
        skill — the lookup falls through to remote children (see
        ``test_hidden_package_skill_does_not_shadow_visible_provider_skill``).

        Args:
            pool: Optional AgentPool for visibility checks.
            node_name: The current node name for package-scoped visibility.
            skill_name: The bare skill name to load.

        Returns:
            Tuple of (Skill, instructions) if found, ``None`` otherwise.
        """
        # Local skills first.
        local_skill = self._local_skills.get(skill_name)
        if local_skill is None:
            # Fuzzy match: try underscore↔hyphen alternatives.
            for alt_name in _name_alternatives(skill_name):
                local_skill = self._local_skills.get(alt_name)
                if local_skill is not None:
                    break

        if local_skill is not None and _is_skill_visible_to_node(pool, local_skill, node_name):
            try:
                instructions = local_skill.load_instructions()
            except (ValueError, OSError):
                instructions = ""
            return local_skill, instructions
        # Local skill found but not visible to this node — fall through to
        # check remote children so a visible remote skill is not shadowed.

        # Remote skills from children
        for child in self._children:
            if not isinstance(child, SkillResource):
                continue
            try:
                provider_entries = await child.list_skills()
            except Exception:  # noqa: BLE001
                continue
            # Map SkillEntry objects to Skill instances and apply visibility
            provider_skills = [
                Skill(
                    name=entry.name,
                    description=entry.description,
                    skill_path=PurePosixPath(entry.uri),
                    instructions="",
                )
                for entry in provider_entries
            ]
            visible_skills = _visible_model_skills(pool, provider_skills, node_name)
            matching_skill = next(
                (s for s in visible_skills if s.name == skill_name),
                None,
            )
            if matching_skill is None:
                # Fuzzy match: try underscore↔hyphen alternatives.
                for alt_name in _name_alternatives(skill_name):
                    matching_skill = next(
                        (s for s in visible_skills if s.name == alt_name),
                        None,
                    )
                    if matching_skill is not None:
                        break
            if matching_skill is not None:
                try:
                    remote_instructions: str | None = await child.read_skill(matching_skill.name)
                except Exception:  # noqa: BLE001
                    remote_instructions = None
                if remote_instructions is None:
                    remote_instructions = ""
                matching_skill.instructions = remote_instructions
                return matching_skill, remote_instructions

        return None

    async def _available_skill_names(
        self,
        pool: AgentPool[Any] | None,
        node_name: str | None,
    ) -> str:
        """Return a comma-separated list of available skill names.

        Args:
            pool: Optional AgentPool for visibility checks.
            node_name: The current node name for skill package visibility.

        Returns:
            Sorted comma-separated string of skill names.
        """
        local_skill_list = list(self._local_skills.values())
        visible_local = _visible_model_skills(pool, local_skill_list, node_name)

        remote_skills: list[Skill] = []
        for child in self._children:
            if not isinstance(child, SkillResource):
                continue
            with contextlib.suppress(Exception):
                entries = await child.list_skills()
                remote_skills.extend(
                    Skill(
                        name=entry.name,
                        description=entry.description,
                        skill_path=PurePosixPath(entry.uri),
                        instructions="",
                    )
                    for entry in entries
                )

        visible_remote = _visible_model_skills(pool, remote_skills, node_name)
        all_names = {skill.name for skill in [*visible_local, *visible_remote]}
        return ", ".join(sorted(all_names))

    # ---- SkillManagerCap-specific API for backward compat ----

    async def get_skill_instructions(self, skill_name: str) -> str:
        """Get instructions for a skill by name (for protocol bridges).

        Args:
            skill_name: The skill name to look up.

        Returns:
            The skill instructions string, or empty string if not found.
        """
        # Local first
        if skill_name in self._local_skills:
            try:
                return self._local_skills[skill_name].load_instructions()
            except (ValueError, OSError):
                return ""

        # Remote
        for child in self._children:
            if isinstance(child, SkillResource):
                try:
                    content = await child.read_skill(skill_name)
                except Exception:  # noqa: BLE001
                    continue
                if content is not None:
                    return content

        return ""

    # ---- SkillResource ----

    async def list_skills(self) -> Sequence[SkillEntry]:
        """List all available skills (local + remote).

        Returns:
            Sequence of ``SkillEntry`` descriptors.
        """
        entries: list[SkillEntry] = []

        # Local skills.
        for name, skill in self._local_skills.items():
            entries.append(
                SkillEntry(
                    name=name,
                    description=skill.description,
                    uri=f"skill://{name}",
                    source="local",
                    skill_path=skill.skill_path,
                )
            )

        # Remote skills from child McpServerCap instances.
        for child in self._children:
            if isinstance(child, SkillResource):
                try:
                    remote_skills = await child.list_skills()
                    entries.extend(remote_skills)
                except Exception:
                    logger.warning(
                        "Failed to list skills from child %r",
                        child.get_serialization_name(),
                        exc_info=True,
                    )

        return entries

    async def read_skill(self, name: str) -> str | None:
        """Read skill content by name.

        Local skills take precedence over remote.

        Args:
            name: Skill name to read.

        Returns:
            Skill content as string, or ``None`` if not found.
        """
        # Local first.
        if name in self._local_skills:
            try:
                return self._local_skills[name].load_instructions()
            except (ValueError, OSError):
                return None

        # Remote.
        for child in self._children:
            if isinstance(child, SkillResource):
                try:
                    content = await child.read_skill(name)
                except Exception:
                    logger.warning(
                        "Failed to read skill %r from child %r",
                        name,
                        child.get_serialization_name(),
                        exc_info=True,
                    )
                    continue
                if content is not None:
                    return content

        return None

    async def skill_exists(self, name: str) -> bool:
        """Check if a skill exists (local or remote).

        Args:
            name: Skill name to check.

        Returns:
            ``True`` if the skill exists, ``False`` otherwise.
        """
        # Local.
        if name in self._local_skills:
            return True

        # Remote.
        for child in self._children:
            if isinstance(child, SkillResource):
                try:
                    if await child.skill_exists(name):
                        return True
                except Exception:  # noqa: BLE001
                    continue

        return False

    # ---- CommandResource ----

    async def list_commands(self) -> Sequence[CommandEntry]:
        """List all available commands (local + remote).

        Each local skill becomes a ``CommandEntry``. Remote commands come
        from child ``McpServerCap`` instances implementing ``CommandResource``.

        Returns:
            Sequence of ``CommandEntry`` descriptors.
        """
        entries: list[CommandEntry] = []

        # Local skills as commands.
        for name, skill in self._local_skills.items():
            if not skill.user_invocable:
                continue
            entries.append(
                CommandEntry(
                    name=name,
                    description=skill.description,
                    skill_uri=f"skill://{name}",
                    source="local",
                )
            )

        # Remote commands from child McpServerCap instances.
        for child in self._children:
            if isinstance(child, CommandResource):
                try:
                    remote_commands = await child.list_commands()
                    entries.extend(remote_commands)
                except Exception:
                    logger.warning(
                        "Failed to list commands from child %r",
                        child.get_serialization_name(),
                        exc_info=True,
                    )

        return entries

    async def get_command(self, name: str) -> CommandEntry | None:
        """Get a specific command by name.

        Local skills take precedence over remote.

        Args:
            name: Command name to retrieve.

        Returns:
            ``CommandEntry`` if found, ``None`` otherwise.
        """
        # Local first.
        if name in self._local_skills:
            skill = self._local_skills[name]
            if skill.user_invocable:
                return CommandEntry(
                    name=name,
                    description=skill.description,
                    skill_uri=f"skill://{name}",
                    source="local",
                )

        # Remote.
        for child in self._children:
            if isinstance(child, CommandResource):
                try:
                    entry = await child.get_command(name)
                except Exception:  # noqa: BLE001
                    continue
                if entry is not None:
                    return entry

        return None

    # ---- Lifecycle ----

    async def __aenter__(self) -> SkillManagerCap[AgentDepsT]:
        """Enter async context for all children."""
        await super().__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context for all children."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
