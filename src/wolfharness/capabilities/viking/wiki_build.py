"""WikiBuildCapability — wiki knowledge construction tools for agentpool.

Exposes the wiki construction toolkit (entity materialization, OPA
records, build lifecycle) as a pydantic-ai capability so agents can
drive manual → wiki entity builds in-process instead of through an
external MCP subprocess.  The concrete ``WikiBuildTools`` implementation
lives in the host application (``xeno_adp_agentic``) and is imported
lazily at runtime; agentpool itself stays framework-clean.

Tools are plain ``WikiBuildTools`` methods wrapped as async functions
inside a ``FunctionToolset``; synchronous implementations run via
``asyncio.to_thread``.  ``functools.wraps`` copies the bound method's
signature and docstring onto the wrapper, so pydantic-ai's tool schema
generation sees the real parameter list (``self`` already stripped for
bound methods).

When a ``role`` is set, the exposed toolset is wrapped with a
:class:`FilteredToolset` so the agent only sees the tools its role may
call (see ``ROLE_TOOLS`` below).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import copy
import inspect
import os
from typing import TYPE_CHECKING, Any, cast, get_type_hints

import logfire
from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.toolsets import AgentToolset


logger = get_logger(__name__)


def _is_viking_backend() -> bool:
    """Return whether the wiki storage backend is OpenViking (default).

    Mirrors ``create_wiki_store`` / ``create_raw_reader``: the backend is
    selected by ``WIKI_STORAGE_BACKEND`` (default ``viking``); any other
    value means a local filesystem backend.
    """
    return os.environ.get("WIKI_STORAGE_BACKEND", "viking") == "viking"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class WikiBuildConfig(BaseModel):
    """Configuration for :class:`WikiBuildCapability`.

    All roots default to ``None`` and resolve from the corresponding
    environment variables (``WIKI_ROOT``, ``LIBRARY_ROOT``, ``CASE_ROOT``,
    ``FAULTANNOTATED_ROOT``) at tool-creation time, so a host application
    can wire everything through its environment.
    """

    model_config = {"arbitrary_types_allowed": True}

    wiki_root: str | None = None
    """Wiki store root (local path, or a ``viking://`` URI in Viking mode)."""

    library_root: str | None = None
    """Raw manual library root (local path or ``viking://resources/...``)."""

    case_root: str | None = None
    """Optional local case-files root."""

    faultannotated_root: str | None = None
    """Optional local fault-annotated docs root."""

    bom_root: str | None = None
    """Optional global BOM tree root (``viking://resources/...``); Phase 0 scans it."""

    tool_names: tuple[str, ...] = ()
    """Optional allowlist of tool names to expose; empty exposes all."""

    include_helpers: bool = True
    """Whether to also expose navigation/snapshot helper tools."""

    role: str | None = None
    """Optional role name used to filter the exposed toolset."""

    index_enabled: bool = False
    """Enable first-turn build-root index injection. When ``True``, the
    capability injects an ``<openviking-index>`` block on the first turn
    listing the config-resolved wiki/raw/bom build roots."""
    index_max_tokens: int = 1000
    """Maximum token budget for the injected index block. Content is
    truncated if it exceeds this budget (chars-to-tokens 4:1 heuristic)."""
    index_limit: int = 20
    """Maximum number of build roots to include in the index block."""


# ---------------------------------------------------------------------------
# Tool inventory & role matrix
# ---------------------------------------------------------------------------

# All wiki build tools (safe superset; the role matrix filters per role).
ALL_WIKI_TOOLS: frozenset[str] = frozenset(
    {
        "list_documents",
        "list_chapters",
        "read_chapter",
        "read_chapters_batch",
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
        "patch_symptom_profile",
        "diff_symptom_profile",
        "list_symptom_profiles",
        "patch_entity",
        "diff_entity",
        "merge_entity",
        "delete_entity",
        "move_entity",
        "plan_component_classification",
        "get_bom_taxonomy",
        "register_bom_component",
        "create_subdir",
        "read_resource",
        "entity_uri",
        "list_children",
        "get_backlinks",
        "get_related_resources",
        "trace_diagnostic_path",
        "search_wiki",
        "find_wiki",
        "grep_wiki",
        "audit_wiki",
        "prune_stale_index_entries",
        "rebuild_backlinks",
        "finalize_wiki",
        "inspect_wiki_state",
        "inspect_build_checkpoint",
        "recover_build",
        "checkpoint_build",
        "preflight_build",
        "get_source_ledger",
        "source_coverage_status",
        "source_change_status",
        "record_source_packet",
        "register_no_entity_chapters",
        "register_case_uri",
        "score_chapters",
        "auto_repair",
        "get_schema",
        "create_opa",
        "create_ops",
        "get_ops",
        "create_opl",
        "get_opls",
        "op_flow_status",
        "discover_opa",
        "get_opas",
        "resolve_opa",
        "apply_opa",
        # In-process navigation/snapshot helpers exposed by the host tools
        # (read-only; safe for every role).
        "browse_chapters",
        "library_doc_ids",
        "read_chapter_map",
        "read_raw_resource",
        "source_snapshot",
    },
)

# In-process navigation helpers (excluded when include_helpers=False).
_HELPER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "browse_chapters",
        "library_doc_ids",
        "read_chapter_map",
        "read_raw_resource",
        "source_snapshot",
    },
)

# Source & entity reading (shared across content roles).
_READ_TOOLS: frozenset[str] = frozenset(
    {
        "list_chapters",
        "read_chapter",
        "read_chapters_batch",
        "read_resource",
        "entity_uri",
        "list_children",
        "get_related_resources",
        "search_wiki",
        "find_wiki",
        "grep_wiki",
        "get_schema",
        "inspect_wiki_state",
    },
)

# Formal entity write surface (materialization & repair).
_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
        "patch_entity",
        "patch_symptom_profile",
        "merge_entity",
    },
)

# Diff/compare helpers — any role that patches needs them.
_DIFF_TOOLS: frozenset[str] = frozenset(
    {
        "diff_entity",
        "diff_symptom_profile",
    },
)

_FILE_OP_TOOLS: frozenset[str] = frozenset(
    {
        "delete_entity",
        "move_entity",
        "plan_component_classification",
        "create_subdir",
        "prune_stale_index_entries",
    },
)

_FINALIZE_TOOLS: frozenset[str] = frozenset(
    {
        "rebuild_backlinks",
        "finalize_wiki",
    },
)

# Build/checkpoint lifecycle — conductor drives, workers inspect when needed.
_LIFECYCLE_TOOLS: frozenset[str] = frozenset(
    {
        "inspect_wiki_state",
        "inspect_build_checkpoint",
        "recover_build",
        "source_coverage_status",
        "source_change_status",
        "audit_wiki",
        "auto_repair",
    },
)

# OPA conflict/gap records.
_OPA_READ_TOOLS: frozenset[str] = frozenset({"get_opas"})
_OPA_WRITE_TOOLS: frozenset[str] = frozenset({"create_opa"})
_OPS_READ_TOOLS: frozenset[str] = frozenset({"get_ops"})
_OPS_WRITE_TOOLS: frozenset[str] = frozenset({"create_ops"})
_OPL_READ_TOOLS: frozenset[str] = frozenset({"get_opls"})
_OPL_WRITE_TOOLS: frozenset[str] = frozenset({"create_opl"})
_OP_FLOW_READ_TOOLS: frozenset[str] = frozenset({"op_flow_status"})
_OPA_DISCOVERY_TOOLS: frozenset[str] = frozenset({"discover_opa"})
_OPA_RESOLVE_TOOLS: frozenset[str] = frozenset({"resolve_opa"})
_OPA_APPLY_TOOLS: frozenset[str] = frozenset({"apply_opa"})

# Role → allowed tools mapping (mirrors the host role model).
ROLE_TOOLS: dict[str, frozenset[str]] = {
    # Conductor orchestrates: reads state/sources, decides via OPA + BOM,
    # runs finalize/repair.  Does not materialize entity content.
    "wiki_conductor": (
        _READ_TOOLS
        | _LIFECYCLE_TOOLS
        | _FINALIZE_TOOLS
        | _OPA_READ_TOOLS
        | _OPA_DISCOVERY_TOOLS
        | _OPS_READ_TOOLS
        | _OPL_READ_TOOLS
        | _OP_FLOW_READ_TOOLS
        | frozenset(
            {
                "get_bom_taxonomy",
                "register_bom_component",
                "record_source_packet",
                "register_no_entity_chapters",
                "score_chapters",
                "diff_entity",
                "merge_entity",
                "checkpoint_build",
                "preflight_build",
            },
        )
    ),
    # Extraction worker owns content materialization: reads sources, writes
    # entities, records source packets, files gap OPAs.
    "wiki_extraction_worker": (
        _READ_TOOLS
        | _WRITE_TOOLS
        | _DIFF_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_READ_TOOLS
        | _OPS_READ_TOOLS
        | _OPL_READ_TOOLS
        | _OP_FLOW_READ_TOOLS
        | frozenset({"record_source_packet"})
    ),
    # Relation worker: patches formal entities (frontmatter relation fields +
    # body cross-references).
    "wiki_relation_worker": (
        _READ_TOOLS
        | _DIFF_TOOLS
        | _OPA_READ_TOOLS
        | frozenset(
            {
                "get_backlinks",
                "patch_entity",
                "patch_symptom_profile",
            },
        )
    ),
    # OPA worker turns deterministic audit findings into readable, evidence-
    # bound OPA records. It cannot write formal entities or proposals.
    "wiki_opa_worker": (
        _READ_TOOLS | _OPA_WRITE_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPL_READ_TOOLS
    ),
    # OPS worker retrieves the target/raw neighbourhood and writes only expert
    # suggestions; formal entity writes remain unavailable.
    "wiki_ops_worker": (
        _READ_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPS_WRITE_TOOLS | _OPL_READ_TOOLS
    ),
    # OPL worker combines existing OPA + OPS into unconfirmed proposals.
    "wiki_opl_worker": (
        _READ_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPL_READ_TOOLS | _OPL_WRITE_TOOLS
    ),
    # File operator: structural repairs, OPA management.  Reads entities to
    # plan moves/repairs but does not need raw-source chapter reading.
    "wiki_file_operator": (
        frozenset(
            {
                "list_chapters",
                "read_resource",
                "entity_uri",
                "list_children",
                "get_related_resources",
                "search_wiki",
                "find_wiki",
                "grep_wiki",
                "get_schema",
                "inspect_wiki_state",
            },
        )
        | _WRITE_TOOLS
        | _DIFF_TOOLS
        | _FILE_OP_TOOLS
        | _OPA_READ_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_RESOLVE_TOOLS
        | _OPA_APPLY_TOOLS
        | frozenset(
            {
                "auto_repair",
                "get_bom_taxonomy",
                "register_bom_component",
                "register_no_entity_chapters",
            },
        )
    ),
}

# All wiki agent role names.
WIKI_AGENT_ROLES: tuple[str, ...] = (
    "wiki_conductor",
    "wiki_extraction_worker",
    "wiki_relation_worker",
    "wiki_file_operator",
    "wiki_opa_worker",
    "wiki_ops_worker",
    "wiki_opl_worker",
)


class RoleFilter:
    """Role-aware wiki tool permission enforcement (no capability plumbing).

    Splits the role/permission responsibility into a plain helper so the
    permission surface can be reused and tested independently.
    """

    def __init__(self, role: str = "wiki_conductor") -> None:
        if role not in WIKI_AGENT_ROLES:
            raise ValueError(
                f"Unknown wiki agent role '{role}'. Must be one of: {', '.join(WIKI_AGENT_ROLES)}",
            )
        self._role = role
        self._allowed_tools = ROLE_TOOLS.get(role, frozenset())

    @property
    def role(self) -> str:
        """The bound agent role name."""
        return self._role

    @property
    def allowed_tools(self) -> frozenset[str]:
        """The frozenset of wiki-build tools this role may call."""
        return self._allowed_tools

    @staticmethod
    def _wiki_tool_name(tool_name: str) -> str | None:
        """Strip MCP server prefixes from a tool name to the wiki tool name.

        A tool may surface as ``wiki-build_write_entity`` or plain
        ``write_entity``.  Returns the canonical wiki tool name, or ``None``
        when the tool is outside the wiki MCP boundary.
        """
        if tool_name in ALL_WIKI_TOOLS:
            return tool_name
        matches = [candidate for candidate in ALL_WIKI_TOOLS if tool_name.endswith(f"_{candidate}")]
        return max(matches, key=len) if matches else None

    def allows_tool(self, tool_name: str) -> bool:
        """Return whether this role may call a tool.

        Unknown tools are team/runtime tools outside the wiki tool boundary
        and remain available.
        """
        wiki_tool_name = self._wiki_tool_name(tool_name)
        return wiki_tool_name is None or wiki_tool_name in self._allowed_tools

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[Any],
    ) -> AbstractToolset[Any]:
        """Apply role permissions to the fully assembled agent toolset."""
        return FilteredToolset(
            wrapped=toolset,
            filter_func=lambda _ctx, tool_def: self.allows_tool(tool_def.name),
        )


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


def _build_tool_fns(
    tools: Any,
    *,
    tool_names: frozenset[str],
    include_helpers: bool = True,
) -> list[Callable[..., Any]]:
    """Wrap the public ``WikiBuildTools`` methods as async tool fns.

    Each wrapper keeps an explicit ``ctx: RunContext`` first parameter with
    a real annotation so wolfharness's tool wrapping recognises it and
    injects the run context at call time (via ``get_argument_key``).  The
    wrapper's ``__signature__`` is rebuilt from the bound method's own
    signature (``self`` stripped by binding, thus not leaked into the model
    schema) with ``ctx`` injected first — pydantic-ai then sees the real
    parameter list and names when generating the tool schema.

    Args:
        tools: The built tools instance whose methods back the tools.
        tool_names: Allowed tool names (superset; helper toggle applies
            on top).
        include_helpers: Whether to include the navigation helpers.

    Returns:
        A list of async callables suitable for ``FunctionToolset``.
    """
    from wolfharness.utils.signatures import (  # lazy: avoids import at module load
        create_modified_signature,
        update_signature,
    )

    names = sorted(tool_names - (_HELPER_TOOL_NAMES if not include_helpers else frozenset()))
    tool_fns: list[Callable[..., Any]] = []

    for name in names:
        attr: object = getattr(tools, name, None)
        if attr is None or not callable(attr):
            continue
        method: Callable[..., Any] = cast(Callable[..., Any], attr)
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):

            async def _async_wrapper(
                ctx: RunContext[Any],
                *args: Any,
                _m: Callable[..., Any] = method,
                **kwargs: Any,
            ) -> Any:
                return await _m(*args, **kwargs)

            wrapper: Callable[..., Any] = _async_wrapper
        else:

            async def _sync_wrapper(
                ctx: RunContext[Any],
                *args: Any,
                _m: Callable[..., Any] = method,
                **kwargs: Any,
            ) -> Any:
                return await asyncio.to_thread(_m, *args, **kwargs)

            wrapper = _sync_wrapper

        wrapper.__name__ = name
        wrapper.__qualname__ = name
        wrapper.__doc__ = inspect.getdoc(method)
        new_sig = create_modified_signature(
            inspect.signature(method),
            inject={"ctx": RunContext[Any]},
        )
        try:
            resolved_hints = get_type_hints(method, include_extras=True)
        except (NameError, TypeError):
            logger.warning("Could not resolve annotations for wiki tool %s", name)
            resolved_hints = {}
        resolved_parameters = [
            parameter.replace(
                annotation=(
                    RunContext[Any]
                    if parameter.name == "ctx"
                    else resolved_hints.get(parameter.name, Any)
                ),
            )
            for parameter in new_sig.parameters.values()
        ]
        new_sig = new_sig.replace(
            parameters=resolved_parameters,
            return_annotation=resolved_hints.get("return", Any),
        )
        update_signature(wrapper, new_sig)
        # Resolve the host method's postponed annotations in its own module,
        # then copy those concrete objects onto the wrapper.  Resolving the
        # strings in this module cannot work for host-only aliases, while
        # replacing them with ``Any`` removes JSON-schema validation and lets
        # models send integers/dicts as strings.
        wrapper.__annotations__ = {
            parameter.name: parameter.annotation
            for parameter in new_sig.parameters.values()
        } | {"return": new_sig.return_annotation}
        tool_fns.append(wrapper)
    return tool_fns


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class WikiBuildCapability(AbstractCapability[Any]):
    """Capability exposing wiki construction tools to an agent.

    Config fields mirror :class:`WikiBuildConfig` as constructor kwargs so
    the entry-point ``build()`` path (``cls(**args)``) can construct the
    capability directly from YAML ``args``.
    """

    _FIRST_TURN_MAX_MESSAGES = 2

    def __init__(
        self,
        config: WikiBuildConfig | None = None,
        *,
        wiki_root: str | None = None,
        library_root: str | None = None,
        case_root: str | None = None,
        faultannotated_root: str | None = None,
        bom_root: str | None = None,
        include_helpers: bool = True,
        role: str | None = None,
        index_enabled: bool = False,
        index_max_tokens: int = 1000,
        index_limit: int = 20,
    ) -> None:
        self._config = config or WikiBuildConfig(
            wiki_root=wiki_root,
            library_root=library_root,
            case_root=case_root,
            faultannotated_root=faultannotated_root,
            bom_root=bom_root,
            include_helpers=include_helpers,
            role=role,
            index_enabled=index_enabled,
            index_max_tokens=index_max_tokens,
            index_limit=index_limit,
        )
        self._tools: Any | None = None
        self._tool_fns: list[Callable[..., Any]] = []
        self._index_injected: bool = False

    @property
    def tools(self) -> Any | None:
        """The lazily-created host tools instance, if created."""
        return self._tools

    def _ensure_tools(self) -> None:
        """Lazily create the host ``WikiBuildTools`` instance.

        Imports the framework-clean host implementation at runtime only;
        agentpool never imports the host package at module load time.
        """
        if self._tools is not None:
            return

        # Lazy import keeps agentpool independent of the host package.
        from xeno_adp_agentic.wiki.serve.build_tools import (
            WikiBuildTools,
        )

        wiki_root = self._config.wiki_root or os.environ.get("WIKI_ROOT") or "output/wiki_newbuild"
        library_root = (
            self._config.library_root or os.environ.get("LIBRARY_ROOT") or "output/library"
        )
        fault_root = self._config.faultannotated_root or os.environ.get("FAULTANNOTATED_ROOT")
        self._tools = WikiBuildTools(
            wiki_root,
            library_root,
            case_root=self._config.case_root or os.environ.get("CASE_ROOT"),
            faultannotated_root=fault_root,
            bom_root=self._config.bom_root or os.environ.get("WIKI_BOM_ROOT"),
        )
        allowed = ALL_WIKI_TOOLS
        if self._config.tool_names:
            allowed = frozenset(self._config.tool_names)
        self._tool_fns = _build_tool_fns(
            self._tools,
            tool_names=allowed,
            include_helpers=self._config.include_helpers,
        )

    def get_instructions(self) -> str | None:
        """Return ``None`` — instructions come from the agent's system prompt."""
        return None

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build a ``FunctionToolset`` from the wrapped build tools.

        When ``config.role`` is set, the toolset is wrapped with the role
        filter so the agent only sees its role's allowed tools.
        """
        from pydantic_ai.toolsets import FunctionToolset

        self._ensure_tools()
        if not self._tool_fns:
            return None
        toolset: FunctionToolset[Any] = FunctionToolset(self._tool_fns, id="wiki_build")
        if self._config.role is not None:
            return RoleFilter(self._config.role).get_wrapper_toolset(toolset)
        return toolset

    async def get_tools(self) -> Sequence[Any]:
        """Return tools as ``Tool`` objects for listing endpoints."""
        from wolfharness.tools.base import FunctionTool

        self._ensure_tools()
        return [FunctionTool.from_callable(fn) for fn in self._tool_fns]

    def _resolve_index_roots(self) -> list[tuple[str, str]]:
        """Resolve the config-driven wiki/raw/bom build roots for the index.

        Mirrors ``_ensure_tools`` root resolution (config override, then the
        same environment keys: ``WIKI_ROOT``, ``LIBRARY_ROOT``,
        ``WIKI_BOM_ROOT``), keeps only roots whose URI starts with
        ``viking://`` (B-scheme: config roots, no server enumeration), and
        labels them ``wiki``/``raw``/``bom``. Missing or non-viking roots
        are omitted entirely.

        Returns:
            ``(label, uri)`` pairs for each resolved ``viking://`` root.
        """
        candidates: list[tuple[str, str | None]] = [
            ("wiki", self._config.wiki_root or os.environ.get("WIKI_ROOT")),
            ("raw", self._config.library_root or os.environ.get("LIBRARY_ROOT")),
            ("bom", self._config.bom_root or os.environ.get("WIKI_BOM_ROOT")),
        ]
        resolved: list[tuple[str, str]] = []
        for label, uri in candidates:
            if uri is None:
                continue
            if uri.startswith("viking://"):
                resolved.append((label, uri))
            elif _is_viking_backend() and label == "wiki" and os.environ.get(
                "VIKING_NAMESPACE",
            ):
                # Launchers pass a local wiki_root (used by the CLI build
                # runner and local backends), but in viking mode the store
                # actually writes to viking://resources/{VIKING_NAMESPACE}
                # (see create_wiki_store). Advertise the real remote root so
                # the agent addresses the live store, not the local stub.
                resolved.append(
                    (label, f"viking://resources/{os.environ['VIKING_NAMESPACE']}"),
                )
            elif _is_viking_backend() and label == "raw" and os.environ.get(
                "VIKING_RAW_NAMESPACE",
            ):
                # Same for the raw chapter reader: viking mode reads from
                # viking://resources/{VIKING_RAW_NAMESPACE} (create_raw_reader).
                resolved.append(
                    (label, f"viking://resources/{os.environ['VIKING_RAW_NAMESPACE']}"),
                )
        return resolved

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Inject a config-driven ``<openviking-index>`` block on the first turn.

        When ``index_enabled`` is ``True`` and this is the first
        ``before_model_request`` call for the session (``_index_injected``
        is ``False`` and message count is <= 2), the method builds the index
        block from the config-resolved wiki/raw/bom build roots and injects
        it as a ``SystemPromptPart`` before the latest user message using
        ``dataclasses.replace()``. No server call — this is pure formatting
        plus a message rewrite.

        On any error, logs a warning and returns the original
        ``request_context`` unchanged (flag already set, no retry).

        Args:
            ctx: The pydantic-ai run context.
            request_context: The model request context containing messages.

        Returns:
            The (possibly modified) model request context.
        """
        if not self._config.index_enabled or self._index_injected:
            return request_context

        if len(request_context.messages) > self._FIRST_TURN_MAX_MESSAGES:
            self._index_injected = True
            return request_context

        self._index_injected = True

        with logfire.span("wiki_build.before_model_request"):
            try:
                from wolfharness.capabilities.viking.wiki_index import (
                    _format_index_block,
                )

                index_block = _format_index_block(
                    self._resolve_index_roots(),
                    max_tokens=self._config.index_max_tokens,
                    limit=self._config.index_limit,
                )
                if not index_block.strip():
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

                system_msg = ModelRequest(parts=[SystemPromptPart(content=index_block)])
                new_messages = [*messages[:insert_idx], system_msg, *messages[insert_idx:]]
                return replace(request_context, messages=new_messages)
            except Exception:
                logger.warning("Wiki index injection failed", exc_info=True)
                return request_context

    async def for_run(self, ctx: RunContext[Any]) -> WikiBuildCapability:
        """Create a per-run copy that resets the first-turn index flag.

        ``AbstractCapability.for_run`` is called once per agent run; its
        default returns ``self`` (shared across runs). We return a shallow
        copy that shares ``_config``/``_tools``/``_tool_fns`` with the
        parent but resets ``_index_injected`` so each run re-injects the
        index block on its first turn.

        Args:
            ctx: The pydantic-ai run context.

        Returns:
            A new ``WikiBuildCapability`` sharing configuration and tools.
        """
        run_copy = copy.copy(self)
        run_copy._index_injected = False
        return run_copy
