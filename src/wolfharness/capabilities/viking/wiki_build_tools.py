"""Tool functions for the WikiBuildCapability.

Follows the same pattern as ``viking/tools.py``: a ``build_tools(cap)``
function returns a list of async tool closures that capture the
capability instance.  The capability class (``WikiBuildCapability``)
delegates to this module in ``get_toolset()``.

Three categories of tools:

1. **Method wrappers** — public ``WikiBuildTools`` methods wrapped as
   async closures via ``_build_method_wrappers``.  Filtered by the
   role matrix (``ROLE_TOOLS``).
2. **External OP closures** — three custom-signature tools
   (``submit_external_opa``, ``submit_external_ops``,
   ``apply_external_opl``) that translate simplified expert-facing
   parameters into the full ``WikiBuildTools`` API.  Built by
   ``_build_external_op_fns``.
3. **Helper tools** — in-process navigation helpers (browse, read
   chapter map, etc.) toggled by ``include_helpers``.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast, get_type_hints

from pydantic_ai.toolsets import AbstractToolset, FilteredToolset
from pydantic_ai.tools import RunContext  # runtime: needed by create_modified_signature

from wolfharness.log import get_logger

if TYPE_CHECKING:
    from wolfharness.capabilities.viking.wiki_build import WikiBuildCapability

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------

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
        "patch_entities_batch",
        "diff_entity",
        "merge_entity",
        "delete_entity",
        "move_entity",
        "plan_component_classification",
        "get_bom_taxonomy",
        "register_bom_component",
        "register_bom_identity_batch",
        "register_model_mapping",
        "get_model_mappings",
        "model_mapping_report",
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
        "rebuild_all_backlinks",
        "finalize_wiki",
        "build_change_report",
        "inspect_wiki_state",
        "inspect_build_checkpoint",
        "recover_build",
        "checkpoint_build",
        "preflight_build",
        "get_source_ledger",
        "record_source_packet",
        "build_relation_closure",
        "register_no_entity_chapters",
        "register_case_uri",
        "score_chapters",
        "auto_repair",
        "get_schema",
        "create_opa",
        "create_ops",
        "get_ops",
        "create_opl",
        "ingest_external_opl",
        "apply_opl",
        "get_opls",
        "op_flow_status",
        "op_flow_report",
        "discover_opa",
        "get_opas",
        "resolve_opa",
        "apply_opa",
        "browse_chapters",
        "browse",
        "next_chapter_window",
        "library_doc_ids",
        "read_chapter_map",
        "read_raw_resource",
        "source_snapshot",
        # External expert tools (custom closures, not method wrappers)
        "submit_external_opa",
        "submit_external_ops",
        "apply_external_opl",
    },
)

_HELPER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "browse_chapters",
        "browse",
        "next_chapter_window",
        "library_doc_ids",
        "read_chapter_map",
        "read_raw_resource",
        "source_snapshot",
    },
)

_EXTERNAL_OP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "submit_external_opa",
        "submit_external_ops",
        "apply_external_opl",
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
        "next_chapter_window",
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
        "prune_stale_index_entries",
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
_OPL_WRITE_TOOLS: frozenset[str] = frozenset({"create_opl"})
_OP_FLOW_READ_TOOLS: frozenset[str] = frozenset({"op_flow_status", "op_flow_report"})
_OPA_DISCOVERY_TOOLS: frozenset[str] = frozenset({"discover_opa"})
_OPA_RESOLVE_TOOLS: frozenset[str] = frozenset({"resolve_opa"})
_OPA_APPLY_TOOLS: frozenset[str] = frozenset({"apply_opa"})

ROLE_TOOLS: dict[str, frozenset[str]] = {
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
                "register_bom_identity_batch",
                "register_model_mapping",
                "get_model_mappings",
                "model_mapping_report",
                "record_source_packet",
                "register_no_entity_chapters",
                "score_chapters",
                "next_chapter_window",
                "diff_entity",
                "checkpoint_build",
                "preflight_build",
                "ingest_external_opl",
                "apply_opl",
            },
        )
    ),
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
        | frozenset({"get_model_mappings", "model_mapping_report"})
    ),
    "wiki_relation_worker": (
        _READ_TOOLS
        | _DIFF_TOOLS
        | _OPA_WRITE_TOOLS
        | _OPA_READ_TOOLS
        | frozenset(
            {
                "get_backlinks",
                "build_relation_closure",
                "patch_entity",
                "patch_symptom_profile",
            },
        )
    ),
    "wiki_opa_worker": (
        _READ_TOOLS | _OPA_WRITE_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPL_READ_TOOLS
    ),
    "wiki_ops_worker": (
        _READ_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPS_WRITE_TOOLS | _OPL_READ_TOOLS
    ),
    "wiki_opl_worker": (
        _READ_TOOLS | _OPA_READ_TOOLS | _OPS_READ_TOOLS | _OPL_READ_TOOLS | _OPL_WRITE_TOOLS
    ),
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
                "register_no_entity_chapters",
            },
        )
    ),
    "wiki_external_expert": (
        _EXTERNAL_OP_TOOL_NAMES
        | frozenset(
            {
                "browse",
                "read_resource",
                "search_wiki",
                "find_wiki",
                "get_schema",
                "get_opas",
                "get_ops",
                "get_opls",
                "op_flow_status",
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
    "wiki_opl_worker",
    "wiki_external_expert",
)

_REQUIRED_ROLE_TOOLS: dict[str, frozenset[str]] = {
    "wiki_relation_worker": frozenset({"build_relation_closure"}),
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
    excluded = excluded | _EXTERNAL_OP_TOOL_NAMES
    names = sorted(tool_names - excluded)
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
        wrapper.__annotations__ = {
            parameter.name: parameter.annotation
            for parameter in new_sig.parameters.values()
        } | {"return": new_sig.return_annotation}
        tool_fns.append(wrapper)
    return tool_fns


# ---------------------------------------------------------------------------
# External OP tool builders
# ---------------------------------------------------------------------------


def _sync_entity_to_remote(tools: Any, target_uri: str, opl_uri: str) -> dict[str, object]:
    """Push the patched entity page and OPL record to remote Viking.

    Best-effort: if the SDK is unavailable or the API key is missing,
    returns a ``skipped`` status without raising.
    """
    if not target_uri.startswith("viking://"):
        return {"sync_status": "skipped", "sync_reason": "target is not a viking:// URI"}

    api_key = os.environ.get("VIKING_API_KEY", "")
    if not api_key:
        return {"sync_status": "skipped", "sync_reason": "VIKING_API_KEY not set"}

    try:
        from openviking_sdk import SyncHTTPClient
    except ImportError:
        return {"sync_status": "skipped", "sync_reason": "openviking-sdk not installed"}

    base_url = os.environ.get("VIKING_BASE_URL", "http://viking.ai.rootcloud.info/")
    client = SyncHTTPClient(url=base_url, api_key=api_key)
    initialize = getattr(client, "initialize", None)
    if callable(initialize):
        initialize()

    result: dict[str, object] = {"sync_status": "ok"}
    errors: list[str] = []

    entity_content = tools.read_resource(target_uri)
    if entity_content:
        try:
            try:
                stat = getattr(client, "stat", None)
                if callable(stat):
                    stat(target_uri)
                write_mode = "replace"
            except Exception:
                write_mode = "create"
            client.write(target_uri, entity_content, mode=write_mode, wait=True)
            result["synced_entity_uri"] = target_uri
        except Exception as exc:
            errors.append(f"entity: {exc}")
            logger.warning("Sync entity to remote failed: %s", exc)

    if opl_uri.startswith("viking://"):
        opl_content = tools.read_resource(opl_uri)
        if opl_content:
            try:
                try:
                    stat = getattr(client, "stat", None)
                    if callable(stat):
                        stat(opl_uri)
                    write_mode = "replace"
                except Exception:
                    write_mode = "create"
                client.write(opl_uri, opl_content, mode=write_mode, wait=True)
                result["synced_opl_uri"] = opl_uri
            except Exception as exc:
                errors.append(f"opl: {exc}")
                logger.warning("Sync OPL to remote failed: %s", exc)

    if errors:
        result["sync_status"] = "partial"
        result["sync_errors"] = errors

    return result


def _build_external_op_fns(
    tools: Any,
    *,
    sync_after_apply: bool = False,
) -> list[Callable[..., Any]]:
    """Build the three external OP tool closures capturing the tools instance."""

    async def submit_external_opa(
        ctx: RunContext[Any],
        *,
        description: str,
        related_uris: list[str],
        target_uri: str = "",
        opa_uri: str = "",
        title: str = "",
    ) -> dict[str, Any]:
        """File or revise an expert OPA (problem/feedback) record.

        Creates a ``category=feedback``, ``reason_code=expert_feedback``
        OPA targeting the given ``target_uri``, or — when ``opa_uri`` is
        provided — revises that existing OPA record in place.

        All URIs use the ``viking://resources/<namespace>/...`` scheme.
        Use ``browse`` to discover available pages and ``search_wiki`` or
        ``find_wiki`` to locate relevant entities by keyword. Use
        ``get_opas`` to check whether an OPA already exists for a target.

        Args:
            description: The problem / feedback content.
            related_uris: ``viking://`` URI references related to this
                problem (evidence pages, source chapters, etc.).
            target_uri: ``viking://`` URI of the wiki entity page this
                OPA targets (e.g.
                ``viking://resources/814/Component/发动机/洋马4TNV98C``).
            opa_uri: ``viking://`` URI of an existing OPA to revise
                instead of creating a new one.
            title: Optional short title; derived from ``description``.

        Returns:
            Dict with ``opa_id``, ``uri``, ``path``, ``title``,
            ``target_uri``. Keep the returned ``uri`` for revision.
        """
        effective_title = title.strip() or description.strip().splitlines()[0][:60]
        opa_id = ""
        if opa_uri:
            opa_id = str(opa_uri).rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
        return await asyncio.to_thread(
            tools.create_opa,
            opa_id=opa_id,
            title=effective_title,
            description=description,
            category="feedback",
            reason_code="expert_feedback",
            target_uri=target_uri,
            evidence_uris=related_uris,
            status="pending",
            finding=description,
            missing=description,
            recommendation=description,
        )

    async def submit_external_ops(
        ctx: RunContext[Any],
        *,
        suggestion: str,
        related_uris: list[str],
        ops_uri: str = "",
        parent_opa: str = "",
        title: str = "",
    ) -> dict[str, Any]:
        """Submit or revise an expert OPS (modification suggestion).

        When ``ops_uri`` is provided, the existing OPS record is revised
        in place. Otherwise a new OPS is created under ``parent_opa``.

        Use ``get_ops`` to retrieve existing suggestions for an OPA
        before deciding whether to create a new OPS or revise an existing
        one.

        Args:
            suggestion: The expert's modification suggestion. First line
                becomes the title; remaining lines become the analysis.
            related_uris: ``viking://`` URI references (evidence pages,
                source chapters, etc.).
            ops_uri: ``viking://`` URI of an existing OPS to revise
                instead of creating a new one.
            parent_opa: OPA ID or ``viking://`` URI this OPS attaches to.
                Required when creating a new record.
            title: Optional short title; derived from ``suggestion``.

        Returns:
            Dict with ``ops_id``, ``uri``, ``parent_opa``,
            ``target_uri``, ``status``. Keep the returned ``uri`` for
            revision.
        """
        lines = [line.strip() for line in suggestion.strip().splitlines() if line.strip()]
        effective_title = title.strip() or (lines[0][:60] if lines else "专家建议")
        effective_analysis = "\n".join(lines[1:]) if len(lines) > 1 else suggestion.strip()
        effective_solution = suggestion.strip()
        return await asyncio.to_thread(
            tools.ingest_external_ops,
            parent_opa=parent_opa,
            title=effective_title,
            analysis=effective_analysis,
            solution=effective_solution,
            evidence_uris=related_uris,
            ops_uri=ops_uri,
        )

    async def apply_external_opl(
        ctx: RunContext[Any],
        *,
        target_uri: str,
        proposal: str,
        related_uris: list[str],
        candidate_content: str = "",
        expected_sha256: str = "",
        expert_id: str = "",
        expert_name: str = "",
        title: str = "",
        opl_uri: str = "",
        auto_apply: bool = True,
    ) -> dict[str, Any]:
        """Apply an expert knowledge proposal (OPL) to a wiki page.

        When ``opl_uri`` is provided, the existing OPL record is revised
        in place instead of creating a new one.

        To compute ``expected_sha256``, first call ``read_resource`` on
        ``target_uri`` to get the current page content, then SHA-256
        hash the UTF-8 encoded bytes. Without a matching hash,
        auto-apply is skipped (``apply_status=needs_review``).

        Args:
            target_uri: ``viking://`` URI of the wiki entity page to
                patch (e.g.
                ``viking://resources/814/Component/发动机/洋马4TNV98C``).
            proposal: The proposed knowledge change.
            related_uris: ``viking://`` URI references (evidence).
            candidate_content: Full replacement page markdown. When
                provided with a matching ``expected_sha256``, the page
                is patched automatically.
            expected_sha256: SHA-256 hex digest of the current content at
                ``target_uri``. Use ``read_resource`` to fetch current
                content, then hash it.
            expert_id: Expert identifier.
            expert_name: Expert display name.
            title: Optional short title; derived from ``proposal``.
            opl_uri: ``viking://`` URI of an existing OPL to revise.
            auto_apply: When ``True`` (default), apply the patch
                immediately if hash matches.

        Returns:
            Dict with ``opl_id``, ``uri``, ``parent_opa``, ``ops_uris``,
            ``target_uri``, ``status``, ``source_type``,
            ``apply_status``, and (when applied) ``applied_at``,
            ``applied_entity_sha256``. Keep ``uri`` for revision.
        """
        effective_title = title.strip() or (
            proposal.strip().splitlines()[0][:60] if proposal.strip() else "专家提案"
        )
        opl_result = await asyncio.to_thread(
            tools.ingest_external_opl,
            title=effective_title,
            target_uri=target_uri,
            proposal=proposal,
            rationale=proposal,
            evidence_uris=related_uris,
            expert_id=expert_id,
            expert_name=expert_name,
            candidate_content=candidate_content,
            expected_sha256=expected_sha256,
            opl_uri=opl_uri,
            auto_apply=auto_apply,
        )
        if sync_after_apply and opl_result.get("apply_status") == "applied":
            sync_result = await asyncio.to_thread(
                _sync_entity_to_remote,
                tools,
                str(opl_result.get("target_uri", "")),
                str(opl_result.get("uri", "")),
            )
            opl_result["sync"] = sync_result
        return opl_result

    return [submit_external_opa, submit_external_ops, apply_external_opl]


# ---------------------------------------------------------------------------
# Main entry point — like tools.py's build_tools(cap)
# ---------------------------------------------------------------------------


def build_tools(cap: WikiBuildCapability) -> list[Callable[..., Any]]:
    """Build the list of tool functions for the WikiBuildCapability.

    Dispatches based on ``cap._config.role``:

    - ``wiki_external_expert`` → only the 3 external OP closures.
    - Any other role (or no role) → method wrappers from
      ``WikiBuildTools``, optionally supplemented with external OP
      closures when ``include_external_ops`` is True.

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

    # External expert role: read tools (method wrappers) + 3 OP closures.
    if role == "wiki_external_expert":
        read_tools = _build_method_wrappers(
            tools,
            tool_names=ROLE_TOOLS["wiki_external_expert"] - _EXTERNAL_OP_TOOL_NAMES,
            include_helpers=True,
        )
        op_tools = _build_external_op_fns(
            tools,
            sync_after_apply=cap.config.sync_after_apply,
        )
        return read_tools + op_tools

    # Standard roles: method wrappers filtered by role.
    allowed = ALL_WIKI_TOOLS - _EXTERNAL_OP_TOOL_NAMES
    if cap.config.tool_names:
        allowed = frozenset(cap.config.tool_names)

    tool_fns = _build_method_wrappers(
        tools,
        tool_names=allowed,
        include_helpers=cap.config.include_helpers,
    )

    # Optionally supplement with external OP closures.
    if cap.config.include_external_ops:
        tool_fns.extend(
            _build_external_op_fns(tools, sync_after_apply=cap.config.sync_after_apply),
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

    For the external expert role, returns the OP flow instructions.
    For all other roles, returns ``None`` (instructions come from the
    agent's system prompt).
    """
    if role == "wiki_external_expert":
        return (
            "You are a wiki external expert. All wiki resources use the "
            "viking:// URI scheme (e.g. viking://resources/814/Component/...).\n"
            "\n"
            "Discovery tools:\n"
            "- browse(uri) — list one level of children under a viking:// path.\n"
            "- search_wiki(query) / find_wiki(query) — semantic search for "
            "relevant wiki pages.\n"
            "- read_resource(uri) — read the full content of a wiki page "
            "(needed to compute expected_sha256 for apply_external_opl).\n"
            "- get_opas() / get_ops(opa_id) / get_opls() — check existing "
            "OP records before submitting new ones.\n"
            "- op_flow_status() — see the current OP flow state.\n"
            "\n"
            "Submission tools:\n"
            "1. submit_external_opa — file a problem/feedback record (OPA) "
            "with a description and related viking:// URIs. Pass opa_uri to "
            "revise an existing OPA.\n"
            "2. submit_external_ops — submit a modification suggestion (OPS) "
            "with suggestion text and related URIs. Pass ops_uri to revise "
            "the same suggestion until accepted.\n"
            "3. apply_external_opl — apply a knowledge proposal (OPL) to a "
            "target page: provide candidate_content (full replacement "
            "markdown) and expected_sha256 (SHA-256 of current page content, "
            "computed from read_resource output). Pass opl_uri to revise.\n"
            "\n"
            "Typical flow:\n"
            "  browse/search → read_resource(target) → submit_external_opa "
            "→ submit_external_ops → apply_external_opl.\n"
            "Each submission tool returns a URI; keep it and pass it back "
            "on the next revision."
        )
    return None
