"""ExternalOPCapability — external expert OP submission tools for WolfHarness.

Exposes three tools that let an external agent submit expert feedback
into the wiki OP flow (OPA → OPS → OPL) and optionally apply the
resulting OPL to update wiki pages.  Built on top of the host
application's ``WikiBuildTools`` (lazily imported, same pattern as
:class:`WikiBuildCapability`).

Tools:

- ``submit_external_opa`` — file an expert problem/feedback record (OPA)
  against a wiki page.  Forces ``category=feedback``,
  ``reason_code=expert_feedback``.
- ``submit_external_ops`` — attach an expert modification suggestion (OPS)
  to a pre-existing OPA, with server-side retrieval receipt validation.
- ``apply_external_opl`` — create an external-expert OPL from OPA + OPS
  and optionally auto-apply it to update the formal wiki entity page.
  Requires ``candidate_content`` or ``candidate_operations`` plus a
  matching ``expected_sha256`` for automatic application.

All three tools delegate to the same ``WikiBuildTools`` instance; in
``viking`` storage mode (default) every write reaches the remote
OpenViking immediately — no separate sync step.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext

from wolfharness.log import get_logger

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AgentToolset


logger = get_logger(__name__)


class ExternalOPConfig(BaseModel):
    """Configuration for :class:`ExternalOPCapability`.

    Roots default to ``None`` and resolve from the same environment
    variables as :class:`WikiBuildConfig` (``WIKI_ROOT``, ``LIBRARY_ROOT``,
    ``CASE_ROOT``, ``FAULTANNOTATED_ROOT``).
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
    """Optional global BOM tree root (``viking://resources/...``)."""

    build_log_dir: str | None = None
    """Optional local directory for structured Wiki build events."""

    sync_after_apply: bool = False
    """When True, push the patched wiki page to remote Viking after a
    successful ``apply_external_opl``.  Intended for ``local_viking``
    storage mode: the OPL is stored and the entity page is patched
    locally first, then the patched page is written to the remote
    Viking server via the SDK client.  In ``viking`` mode this is a
    no-op (writes are already remote).  Requires ``VIKING_API_KEY``
    env var."""


def _sync_entity_to_remote(tools: Any, target_uri: str, opl_uri: str) -> dict[str, object]:
    """Push the patched entity page and OPL record to remote Viking.

    Best-effort: if the SDK is unavailable or the API key is missing,
    returns a ``skipped`` status without raising.  Only syncs
    ``viking://`` URIs (no-op for ``file:///`` local mode).

    Args:
        tools: The ``WikiBuildTools`` instance (for reading local content).
        target_uri: The patched entity's viking:// URI.
        opl_uri: The OPL record's viking:// URI.

    Returns:
        Dict with ``sync_status`` (``ok``/``partial``/``skipped``) and
        details about what was synced.
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
        sync_uri = target_uri
        try:
            try:
                stat = getattr(client, "stat", None)
                if callable(stat):
                    stat(sync_uri)
                    write_mode = "replace"
                else:
                    write_mode = "replace"
            except Exception:
                write_mode = "create"
            client.write(sync_uri, entity_content, mode=write_mode, wait=True)
            result["synced_entity_uri"] = sync_uri
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
                    else:
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


def _build_tool_fns(tools: Any, *, sync_after_apply: bool = False) -> list[Callable[..., Any]]:
    """Build the three external OP tool closures capturing the tools instance.

    Each wrapper keeps an explicit ``ctx: RunContext[Any]`` first parameter
    so wolfharness's tool wrapping recognises it and injects the run
    context at call time.  Synchronous host methods run via
    ``asyncio.to_thread``.
    """

    async def submit_external_opa(
        ctx: RunContext[Any],
        *,
        title: str,
        description: str,
        target_uri: str,
        evidence_uris: list[str],
        finding: str,
        missing: str,
        recommendation: str,
        target_section: str = "",
    ) -> dict[str, Any]:
        """File an expert OPA (problem/feedback) record about a wiki page.

        Creates a ``category=feedback``, ``reason_code=expert_feedback``
        OPA targeting the given ``target_uri``.  The OPA is persisted to
        ``OP/OpA/专家反馈/<opa_id>.md`` in the wiki store.

        Expert identity is not recorded at the OPA level; it is captured
        later in ``apply_external_opl`` when the OPL is created.

        Args:
            title: Short title for the problem record.
            description: Detailed description of the problem.
            target_uri: URI of the wiki page this OPA targets.
            evidence_uris: List of evidence URI references supporting this OPA.
            finding: What was found (the observed problem).
            missing: What is missing or incorrect.
            recommendation: Suggested fix or further investigation.
            target_section: Optional section within the target page.

        Returns:
            Dict with ``opa_id``, ``uri``, ``path``, ``title``,
            ``target_uri``.
        """
        return await asyncio.to_thread(
            tools.create_opa,
            title=title,
            description=description,
            category="feedback",
            reason_code="expert_feedback",
            target_uri=target_uri,
            target_section=target_section,
            evidence_uris=evidence_uris,
            status="pending",
            finding=finding,
            missing=missing,
            recommendation=recommendation,
        )

    async def submit_external_ops(
        ctx: RunContext[Any],
        *,
        parent_opa: str,
        title: str,
        retrieval_query: str,
        retrieved_uris: list[str],
        analysis: str,
        solution: str,
        evidence_uris: list[str] | None = None,
        ops_uri: str = "",
    ) -> dict[str, Any]:
        """Attach an expert OPS (modification suggestion) to a pre-existing OPA.

        The OPA must already exist in the wiki store.  The server re-runs
        ``retrieval_query`` over the wiki + raw scopes and validates that
        every URI in ``retrieved_uris`` appears in the server-side hit
        set.

        When ``ops_uri`` is provided the existing OPS record at that URI is
        revised in place with the new ``analysis`` / ``solution`` /
        ``evidence_uris`` (expert keeps revising one opinion until it is
        accepted); otherwise a new OPS is created.

        Args:
            parent_opa: The OPA ID (slug) this OPS attaches to.
            title: Short title for the suggestion.
            retrieval_query: The search query used to find supporting evidence.
            retrieved_uris: URIs the expert cites, must be in the server
                retrieval results.
            analysis: Expert analysis of the problem.
            solution: Proposed solution or modification.
            evidence_uris: Additional evidence URI references.
            ops_uri: Optional URI of an existing OPS record to revise in
                place instead of creating a new one.

        Returns:
            Dict with ``ops_id``, ``uri``, ``parent_opa``,
            ``target_uri``, ``status``.
        """
        if ops_uri:
            return await asyncio.to_thread(
                tools.update_ops,
                ops_uri,
                title=title,
                analysis=analysis,
                solution=solution,
                evidence_uris=evidence_uris,
            )
        return await asyncio.to_thread(
            tools.create_ops,
            parent_opa=parent_opa,
            title=title,
            retrieval_query=retrieval_query,
            retrieved_uris=retrieved_uris,
            analysis=analysis,
            solution=solution,
            evidence_uris=evidence_uris,
        )

    async def apply_external_opl(
        ctx: RunContext[Any],
        *,
        parent_opa: str,
        ops_uris: list[str],
        title: str,
        proposal: str,
        rationale: str,
        evidence_uris: list[str] | None = None,
        expert_id: str = "",
        expert_name: str = "",
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        auto_apply: bool = True,
        opl_uri: str = "",
    ) -> dict[str, Any]:
        """Create an external-expert OPL from OPA + OPS and optionally apply it.

        Creates an OPL with ``source_type="external_expert"`` linking the
        parent OPA and one or more OPS records.  When ``auto_apply`` is
        ``True`` and a machine-readable candidate is provided
        (``candidate_content`` or ``candidate_operations`` plus a matching
        ``expected_sha256``), the OPL is immediately applied to update
        the formal wiki entity page via ``merge_entity`` with the
        ``external_authority`` conflict policy.

        Without a machine candidate, the OPL is stored with
        ``apply_status="needs_review"`` and must be manually applied
        later.

        When ``opl_uri`` is provided the existing OPL record at that URI is
        revised in place (expert revises the same proposal across rounds)
        instead of creating a new OPL.

        Args:
            parent_opa: The OPA ID (slug) this OPL derives from.
            ops_uris: URIs of one or more OPS records attached to the OPA.
            title: Short title for the proposal.
            proposal: The proposed knowledge change (free-form prose).
            rationale: Why this change is correct and necessary.
            evidence_uris: Additional evidence URI references.
            expert_id: Expert identifier.
            expert_name: Expert display name.
            candidate_content: Full replacement page markdown.  Mutually
                exclusive with ``candidate_operations``.
            candidate_operations: Deterministic patch operations.  Mutually
                exclusive with ``candidate_content``.
            expected_sha256: SHA-256 of the current target page content
                (64 hex chars).  Required for auto-apply on existing pages.
            auto_apply: When ``True`` (default), attempt to apply the OPL
                immediately after creation.
            opl_uri: Optional URI of an existing OPL record to revise in
                place instead of creating a new one.

        Returns:
            Dict with ``opl_id``, ``uri``, ``parent_opa``, ``ops_uris``,
            ``target_uri``, ``status``, ``source_type``,
            ``apply_status``, and (when auto-applied) ``applied_at``,
            ``applied_entity_sha256``.
        """
        opl_result = await asyncio.to_thread(
            tools.create_opl,
            parent_opa=parent_opa,
            ops_uris=ops_uris,
            title=title,
            proposal=proposal,
            rationale=rationale,
            evidence_uris=evidence_uris,
            source_type="external_expert",
            expert_id=expert_id,
            expert_name=expert_name,
            candidate_content=candidate_content,
            candidate_operations=candidate_operations,
            expected_sha256=expected_sha256,
            opl_uri=opl_uri,
        )
        if auto_apply and opl_result.get("opl_id"):
            apply_result = await asyncio.to_thread(
                tools.apply_opl,
                opl_result["opl_id"],
            )
            opl_result.update(apply_result)
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


class ExternalOPCapability(AbstractCapability[Any]):
    """Capability exposing external expert OP submission tools to an agent.

    Provides three tools (``submit_external_opa``, ``submit_external_ops``,
    ``apply_external_opl``) backed by the host application's
    ``WikiBuildTools`` instance.  The host tools are imported lazily at
    runtime; agentpool stays framework-clean.

    Config fields mirror :class:`ExternalOPConfig` as constructor kwargs
    so the entry-point ``build()`` path (``cls(**args)``) can construct
    the capability directly from YAML ``args``.
    """

    def __init__(
        self,
        config: ExternalOPConfig | None = None,
        *,
        wiki_root: str | None = None,
        library_root: str | None = None,
        case_root: str | None = None,
        faultannotated_root: str | None = None,
        bom_root: str | None = None,
        build_log_dir: str | None = None,
        sync_after_apply: bool = False,
    ) -> None:
        self._config = config or ExternalOPConfig(
            wiki_root=wiki_root,
            library_root=library_root,
            case_root=case_root,
            faultannotated_root=faultannotated_root,
            bom_root=bom_root,
            build_log_dir=build_log_dir,
            sync_after_apply=sync_after_apply,
        )
        self._tools: Any | None = None
        self._tool_fns: list[Callable[..., Any]] = []

    @property
    def tools_instance(self) -> Any | None:
        """The lazily-created host tools instance, if created."""
        return self._tools

    def _ensure_tools(self) -> None:
        """Lazily create the host ``WikiBuildTools`` instance.

        Imports the framework-clean host implementation at runtime only;
        agentpool never imports the host package at module load time.
        """
        if self._tools is not None:
            return

        from xeno_adp_agentic.wiki.build.build_logger import WikiBuildLogger
        from xeno_adp_agentic.wiki.serve.build_tools import WikiBuildTools

        wiki_root = self._config.wiki_root or os.environ.get("WIKI_ROOT") or "output/wiki_newbuild"
        library_root = (
            self._config.library_root or os.environ.get("LIBRARY_ROOT") or "output/library"
        )
        fault_root = self._config.faultannotated_root or os.environ.get("FAULTANNOTATED_ROOT")
        log_dir = self._config.build_log_dir or os.environ.get("WIKI_BUILD_LOG_DIR")
        if not log_dir:
            log_dir = "logs" if "://" in str(wiki_root) else str(Path(wiki_root) / "logs")
        build_logger = WikiBuildLogger(log_dir)
        self._tools = WikiBuildTools(
            wiki_root,
            library_root,
            case_root=self._config.case_root or os.environ.get("CASE_ROOT"),
            faultannotated_root=fault_root,
            bom_root=self._config.bom_root or os.environ.get("WIKI_BOM_ROOT"),
            build_logger=build_logger,
        )
        self._tool_fns = _build_tool_fns(
            self._tools,
            sync_after_apply=self._config.sync_after_apply,
        )

    def get_instructions(self) -> str | None:
        """Return guidance for the external agent on the OP flow."""
        return (
            "You have three tools for the wiki OP (Open Problem) flow:\n"
            "1. submit_external_opa — file a problem/feedback record (OPA) "
            "against a wiki page.\n"
            "2. submit_external_ops — attach a modification suggestion (OPS) "
            "to an existing OPA, citing evidence URIs.\n"
            "3. apply_external_opl — create a knowledge proposal (OPL) from "
            "OPA + OPS and optionally apply it to update the wiki page.\n"
            "Typical flow: submit_external_opa → submit_external_ops → "
            "apply_external_opl. Each step returns an ID you pass to the next."
        )

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build a ``FunctionToolset`` from the three external OP tools."""
        from pydantic_ai.toolsets import FunctionToolset

        self._ensure_tools()
        if not self._tool_fns:
            return None
        toolset: FunctionToolset[Any] = FunctionToolset(self._tool_fns, id="external_op")
        return toolset
