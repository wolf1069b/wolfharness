"""Tool functions for the WikiBuildCapability.

Follows the same pattern as ``viking/tools.py``: a ``build_tools(cap)``
function returns a list of async tool closures that capture the
capability instance.  The capability class (``WikiBuildCapability``)
delegates to this module in ``get_toolset()``.

Two categories of tools:

1. **Method wrappers** — public ``WikiBuildTools`` methods wrapped as
   async closures via ``_build_method_wrappers``.  Filtered by the
   role matrix (``ROLE_TOOLS``).
2. **Helper tools** — in-process navigation helpers (browse, read
   chapter map, etc.) toggled by ``include_helpers``.

External OP capabilities (OPA/OPS/OPL as tickets, the
``wiki_external_expert`` role) now live in :mod:`wolfharness.capabilities.wiki.tickets.ticket`;
this module routes that role to ``ticket.build_ticket_tools``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from pydantic_ai.tools import RunContext  # runtime: needed by create_modified_signature
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.capabilities.wiki.build import WikiBuildCapability

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------

ALL_WIKI_TOOLS: frozenset[str] = frozenset(
    {
        "read_chapter",
        "read_chapters_batch",
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
        "patch_symptom_profile",
        "diff_symptom_profile",
        "patch_entity",
        "patch_entities_batch",
        "diff_entity",
        "merge_entity",
        "delete_entity",
        "move_entity",
        "plan_component_classification",
        "get_bom_taxonomy",
        "register_bom_component",
        "register_bom_identity_batch",
        "bom_enrichment_status",
        "plan_bom_enrichment",
        "register_model_mapping",
        "get_model_mappings",
        "model_mapping_report",
        "create_subdir",
        "wiki_read_resource",
        "entity_uri",
        "list_children",
        "get_backlinks",
        "get_related_resources",
        "find_wiki",
        "grep_wiki",
        "audit_wiki",
        "rebuild_backlinks",
        "rebuild_all_backlinks",
        "finalize_wiki",
        "build_change_report",
        "inspect_wiki_state",
        "inspect_build_checkpoint",
        "recover_build",
        "checkpoint_build",
        "plan_chapter_work",
        "record_source_packet",
        "plan_materialization_work",
        "materialize_template_batch",
        "record_materialization_receipt",
        "plan_relation_work",
        "plan_relation_shards",
        "register_no_entity_chapters",
        "auto_repair",
        "get_schema",
        "create_opa",
        "create_ops",
        "get_ops",
        "get_opls",
        "get_expert_authority",
        "get_wiki_change_events",
        "op_flow_status",
        "op_flow_report",
        "discover_opa",
        "get_opas",
        "resolve_opa",
        "apply_opa",
        "refine_opa_reason_code",
        "ops_dispatch_plan",
        "browse_chapters",
        "browse",
        "migrate_source_uri_references",
        "sync_device_system_chapters",
    },
)

_HELPER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "browse_chapters",
        "browse",
    },
)

# ---------------------------------------------------------------------------
# Role matrix
# ---------------------------------------------------------------------------

_READ_TOOLS: frozenset[str] = frozenset(
    {
        "list_chapters",
        "browse_chapters",
        "browse",
        "read_chapter",
        "read_chapters_batch",
        "wiki_read_resource",
        "entity_uri",
        "list_children",
        "get_related_resources",
        "find_wiki",
        "grep_wiki",
        "get_schema",
        "inspect_wiki_state",
    },
)

# Extraction is deliberately chapter-granular.  ``read_chapters_batch`` is
# retained in the general read set for conductor/backward-compatible callers,
# but exposing it to an extraction worker lets a small task expand into an
# unbounded model context before the worker can persist a packet.
_EXTRACTION_READ_TOOLS: frozenset[str] = _READ_TOOLS - frozenset({"read_chapters_batch"})

_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "write_entity",
        "write_entities_batch",
        "write_symptom_profile",
        "patch_entity",
        "patch_entities_batch",
        "patch_symptom_profile",
        "merge_entity",
    },
)

_DIFF_TOOLS: frozenset[str] = frozenset({"diff_entity", "diff_symptom_profile"})

_FILE_OP_TOOLS: frozenset[str] = frozenset(
    {
        "delete_entity",
        "move_entity",
        "plan_component_classification",
        "create_subdir",
        "migrate_source_uri_references",
        "sync_device_system_chapters",
    },
)

_FINALIZE_TOOLS: frozenset[str] = frozenset(
    {
        "rebuild_backlinks",
        "rebuild_all_backlinks",
        "finalize_wiki",
        "build_change_report",
    },
)

_LIFECYCLE_TOOLS: frozenset[str] = frozenset(
    {
        "inspect_wiki_state",
        "inspect_build_checkpoint",
        "recover_build",
        "audit_wiki",
        "auto_repair",
        "build_change_report",
    },
)

_OPA_READ_TOOLS: frozenset[str] = frozenset({"get_opas"})
_OPA_WRITE_TOOLS: frozenset[str] = frozenset({"create_opa"})
_OPS_READ_TOOLS: frozenset[str] = frozenset({"get_ops"})
_OPS_WRITE_TOOLS: frozenset[str] = frozenset({"create_ops"})
_OPL_READ_TOOLS: frozenset[str] = frozenset({"get_opls"})
_OP_AUTHORITY_READ_TOOLS: frozenset[str] = frozenset(
    {"get_expert_authority", "get_wiki_change_events"},
)
_OP_FLOW_READ_TOOLS: frozenset[str] = frozenset({"op_flow_status", "op_flow_report"})
_OPA_DISCOVERY_TOOLS: frozenset[str] = frozenset({"discover_opa"})
_OPA_RESOLVE_TOOLS: frozenset[str] = frozenset({"resolve_opa"})
_OPA_REFINE_TOOLS: frozenset[str] = frozenset({"refine_opa_reason_code"})
_OPA_APPLY_TOOLS: frozenset[str] = frozenset({"apply_opa"})
_OPS_DISPATCH_TOOLS: frozenset[str] = frozenset({"ops_dispatch_plan"})
# OPS resolver closes relation_missed gaps only; materialization stays with file_operator.
_OPS_RESOLVE_WRITE_TOOLS: frozenset[str] = frozenset({"patch_entity", "rebuild_backlinks"})

ROLE_TOOLS: dict[str, frozenset[str]] = {
    "wiki_conductor": (
        _READ_TOOLS
        | _LIFECYCLE_TOOLS
        | _FINALIZE_TOOLS
        | _OPA_READ_TOOLS
        | _OPA_DISCOVERY_TOOLS
        | _OPS_READ_TOOLS
        | _OPL_READ_TOOLS
        | _OP_AUTHORITY_READ_TOOLS
        | _OP_FLOW_READ_TOOLS
        | _OPS_DISPATCH_TOOLS
        | frozenset(
            {
                "get_bom_taxonomy",
                "register_bom_component",
                "register_bom_identity_batch",
                "bom_enrichment_status",
                "plan_bom_enrichment",
                "register_model_mapping",
                "get_model_mappings",
                "model_mapping_report",
                "record_source_packet",
                "plan_materialization_work",
                "materialize_template_batch",
                "record_materialization_receipt",
                "plan_relation_work",
                "plan_relation_shards",
                "register_no_entity_chapters",
                "diff_entity",
                "checkpoint_build",
                "plan_chapter_work",
                "migrate_source_uri_references",
                "sync_device_system_chapters",
            },
        )
    ),
    "wiki_extraction_worker": (
        _EXTRACTION_READ_TOOLS
        | _WRITE_TOOLS
        | _DIFF_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_READ_TOOLS
        | _OPS_READ_TOOLS
        | _OPL_READ_TOOLS
        | _OP_AUTHORITY_READ_TOOLS
        | _OP_FLOW_READ_TOOLS
        | frozenset({"record_source_packet", "record_materialization_receipt"})
        | frozenset({"get_model_mappings", "model_mapping_report"})
    ),
    "wiki_relation_worker": (
        _READ_TOOLS
        | _DIFF_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_READ_TOOLS
        | _OP_AUTHORITY_READ_TOOLS
        | frozenset(
            {
                "get_backlinks",
                "patch_entity",
                "patch_symptom_profile",
            },
        )
    ),
    "wiki_opa_worker": (
        _READ_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_READ_TOOLS
        | _OPS_READ_TOOLS
        | _OPL_READ_TOOLS
        | _OP_AUTHORITY_READ_TOOLS
    ),
    "wiki_ops_worker": (
        _READ_TOOLS
        | _OPA_READ_TOOLS
        | _OPS_READ_TOOLS
        | _OPS_WRITE_TOOLS
        | _OPL_READ_TOOLS
        | _OP_AUTHORITY_READ_TOOLS
        | _OPA_REFINE_TOOLS
        | _OPA_RESOLVE_TOOLS
        | _OPS_RESOLVE_WRITE_TOOLS
    ),
    "wiki_file_operator": (
        frozenset(
            {
                "list_chapters",
                "wiki_read_resource",
                "entity_uri",
                "list_children",
                "get_related_resources",
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
        | _OP_AUTHORITY_READ_TOOLS
        | frozenset(
            {
                "auto_repair",
                "get_bom_taxonomy",
                "register_no_entity_chapters",
            },
        )
    ),
    "wiki_external_expert": (
        frozenset(
            {
                "browse",
                "wiki_read_resource",
                "find_wiki",
                "get_schema",
                "get_opas",
                "get_ops",
                "get_opls",
                "get_expert_authority",
                "get_wiki_change_events",
                "diff_entity",
                "plan_entity_move",
            },
        )
    ),
}

WIKI_AGENT_ROLES: tuple[str, ...] = (
    "wiki_conductor",
    "wiki_extraction_worker",
    "wiki_relation_worker",
    "wiki_file_operator",
    "wiki_opa_worker",
    "wiki_ops_worker",
    "wiki_external_expert",
)

_REQUIRED_ROLE_TOOLS: dict[str, frozenset[str]] = {
    "wiki_relation_worker": frozenset({"patch_entity", "find_wiki", "get_related_resources"}),
}


# ---------------------------------------------------------------------------
# RoleFilter
# ---------------------------------------------------------------------------


class RoleFilter:
    """Role-aware wiki tool permission enforcement."""

    def __init__(self, role: str = "wiki_conductor") -> None:
        if role not in WIKI_AGENT_ROLES:
            raise ValueError(
                f"Unknown wiki agent role '{role}'. Must be one of: {', '.join(WIKI_AGENT_ROLES)}",
            )
        self._role = role
        self._allowed_tools = ROLE_TOOLS.get(role, frozenset())

    @property
    def role(self) -> str:
        return self._role

    @property
    def allowed_tools(self) -> frozenset[str]:
        return self._allowed_tools

    @staticmethod
    def _wiki_tool_name(tool_name: str) -> str | None:
        if tool_name in ALL_WIKI_TOOLS:
            return tool_name
        matches = [candidate for candidate in ALL_WIKI_TOOLS if tool_name.endswith(f"_{candidate}")]
        return max(matches, key=len) if matches else None

    def allows_tool(self, tool_name: str) -> bool:
        wiki_tool_name = self._wiki_tool_name(tool_name)
        return wiki_tool_name is None or wiki_tool_name in self._allowed_tools

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[Any],
    ) -> AbstractToolset[Any]:
        return FilteredToolset(
            wrapped=toolset,
            filter_func=lambda _ctx, tool_def: self.allows_tool(tool_def.name),
        )


# ---------------------------------------------------------------------------
# Method wrapper builder (for WikiBuildTools methods)
# ---------------------------------------------------------------------------


# Tool name → bound method name. Some capability methods must keep their
# ``ResourceAccess`` protocol name (e.g. ``read_resource``); the agent-facing
# tool name is prefixed to avoid clashing with the generic ``resource_access``
# toolset that a native agent may also host.
_TOOL_NAME_BY_METHOD_NAME: dict[str, str] = {
    "wiki_read_resource": "read_resource",
}


def _build_method_wrappers(
    tools: Any,
    *,
    tool_names: frozenset[str],
    include_helpers: bool = True,
) -> list[Callable[..., Any]]:
    """Wrap public ``WikiBuildTools`` methods as async tool fns.

    Each wrapper keeps an explicit ``ctx: RunContext`` first parameter so
    wolfharness's tool wrapping recognises it and injects the run context
    at call time.  The wrapper's ``__signature__`` is rebuilt from the
    bound method's own signature (``self`` stripped) with ``ctx`` injected
    first.
    """
    from wolfharness.utils.signatures import (  # lazy: avoids import at module load
        create_modified_signature,
        update_signature,
    )

    excluded = _HELPER_TOOL_NAMES if not include_helpers else frozenset()
    names = sorted(tool_names - excluded)
    tool_fns: list[Callable[..., Any]] = []

    for name in names:
        method_name = _TOOL_NAME_BY_METHOD_NAME.get(name, name)
        attr: object = getattr(tools, method_name, None)
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
        wrapper.__annotations__ = {
            parameter.name: parameter.annotation for parameter in new_sig.parameters.values()
        } | {"return": new_sig.return_annotation}
        tool_fns.append(wrapper)
    return tool_fns


# ---------------------------------------------------------------------------
# Main entry point — like tools.py's build_tools(cap)
# ---------------------------------------------------------------------------


def build_tools(cap: WikiBuildCapability) -> list[Callable[..., Any]]:
    """Build the list of tool functions for the WikiBuildCapability.

    Dispatches based on ``cap._config.role``:

    - ``wiki_external_expert`` → the ticket tools from
      :mod:`wolfharness.capabilities.wiki.tickets.ticket`
      (``build_ticket_tools``).
    - Any other role (or no role) → method wrappers from
      ``WikiBuildTools``.

    Args:
        cap: The ``WikiBuildCapability`` instance that owns these tools.

    Returns:
        A list of async tool functions suitable for ``FunctionToolset``.
    """
    cap._ensure_tools()
    tools = cap.tools
    if tools is None:
        return []

    role = cap.config.role

    # External expert role: read method-wrappers + ticket tools.
    if role == "wiki_external_expert":
        from wolfharness.capabilities.wiki.tickets.ticket import build_ticket_tools

        read_names = ROLE_TOOLS["wiki_external_expert"]
        if cap.config.tool_names:
            # tool_names 白名单收窄只读工具集, ticket 写工具始终保留。
            read_names = read_names & frozenset(cap.config.tool_names)
        read_tools = _build_method_wrappers(
            tools,
            tool_names=read_names,
            include_helpers=True,
        )
        return read_tools + build_ticket_tools(cap)

    # Standard roles: method wrappers filtered by role.
    allowed = ALL_WIKI_TOOLS
    if cap.config.tool_names:
        allowed = frozenset(cap.config.tool_names)

    tool_fns = _build_method_wrappers(
        tools,
        tool_names=allowed,
        include_helpers=cap.config.include_helpers,
    )

    available = {getattr(fn, "__name__", "") for fn in tool_fns}
    missing = sorted(_REQUIRED_ROLE_TOOLS.get(role or "", frozenset()) - available)
    if missing:
        raise RuntimeError(
            f"Wiki build capability for role {role!r} is incomplete; "
            f"missing required tools: {', '.join(missing)}",
        )

    return tool_fns


def get_instructions(role: str | None) -> str | None:
    """Return guidance for the agent on available tools.

    For the external expert role, returns the ticket flow instructions
    from :mod:`wolfharness.capabilities.wiki.tickets.ticket`.  For all other
    roles, returns ``None`` (instructions come from the agent's system
    prompt).
    """
    if role == "wiki_external_expert":
        from wolfharness.capabilities.wiki.tickets.ticket import get_ticket_instructions

        return get_ticket_instructions()
    return None
