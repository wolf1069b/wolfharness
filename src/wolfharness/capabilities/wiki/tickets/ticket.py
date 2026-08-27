"""Ticket surface for the WikiBuildCapability (external expert / eval flow).

Exposes the OPA → OPS → OPL knowledge loop as *tickets* so an outside
system (an evaluator, an external expert) can submit a problem, attach an
expert recommendation, form an integrated proposal, and apply it to the
Wiki through seven tool closures:

- ``read_resource`` — read a wiki entity page and return its content
  plus SHA-256 hash (for ``candidate_content`` preparation and
  ``expected_sha256`` optimistic locking).
- ``create_opa_ticket`` — file or revise a problem/feedback record (OPA).
- ``create_ops_ticket`` — attach an expert recommendation (OPS) to an OPA.
- ``create_opl_ticket`` — integrate one OPA + its OPS into a proposal
  (OPL) snapshot.
- ``apply_opl_ticket`` — apply the OPL patch to the Wiki, close the OPA,
  and return a terminal status.
- ``get_ticket_status`` — read the current state of OPA / OPS / OPL
  tickets for one target or OPA.
- ``submit_eval_payload`` — one-shot ingestion of an eval revision
  payload (``EvalPayload``) that materializes OPA + OPS tickets from it.

The ticket models mirror the eval revision JSON (``target``,
``cited_references``, ``expert_opinion``, ``suggested_resolution``, ...);
``cited_references[].uri`` and ``evidence`` (修改关联的uri) are the
tractable engine evidence, and ``target.entity_uri`` is the source/target.
All persistence goes through the underlying ``WikiBuildTools`` engine —
nothing here re-implements OP storage.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    Callable,  # noqa: TC003 — runtime: function_schema evaluates closure annotations
)
import contextlib
import os
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai.tools import (
    RunContext,  # noqa: TC002 — runtime: function_schema evaluates closure annotations
)

from wolfharness.log import get_logger


if TYPE_CHECKING:
    from wolfharness.capabilities.wiki.build import WikiBuildCapability

logger = get_logger(__name__)

# Viking metadata files that are not real entity pages.
_OP_METADATA_FILENAMES = frozenset({".overview.md", ".abstract.md", "entities.json"})

# ---------------------------------------------------------------------------
# Ticket inventory
# ---------------------------------------------------------------------------

TICKET_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_resource",
        "create_opa_ticket",
        "create_ops_ticket",
        "create_opl_ticket",
        "apply_opl_ticket",
        "get_ticket_status",
        "submit_eval_payload",
    },
)


# ---------------------------------------------------------------------------
# Ticket models (mirror the eval revision JSON contract)
# ---------------------------------------------------------------------------


class CitedReference(BaseModel):
    """One cited reference inside an eval revision (technical evidence URI)."""

    ref_id: str = ""
    title: str = ""
    uri: str = ""


class TicketTarget(BaseModel):
    """The wiki entity a revision is anchored to (``target`` block)."""

    entity_uri: str = ""
    entity_type: str = ""
    matched_section_title: str = ""
    content_snippet: str = ""


class EvalRevision(BaseModel):
    """One eval-generated revision; sediments into an OPA and an OPS.

    Field mapping to the engine:
    - ``target.entity_uri`` → OPA/OPS ``target_uri`` (the source).
    - ``cited_references[].uri`` + ``evidence`` → ``evidence_uris``.
    - ``target.content_snippet`` → OPA problem ``description`` /
      ``finding`` / ``missing``.
    - ``suggested_resolution`` → OPA ``recommendation`` / OPS ``analysis``.
    - ``expert_opinion`` → OPS ``solution``.
    """

    ticket_id: str = ""
    kind: str = "OPA"
    ops_ref: str = ""
    annotation_ref: str = ""
    anchor_status: str = ""
    resolution_path: str = ""
    target: TicketTarget | None = None
    cited_references: list[CitedReference] = Field(default_factory=list)
    expert_opinion: str = ""
    evidence: list[str] = Field(default_factory=list)
    suggested_resolution: str = ""
    source: str = "reflect"
    status: str = "pending_external_submission"


class EvalPayload(BaseModel):
    """Top-level eval output: one entity + N revisions."""

    eval: str = ""
    round: int = 0
    entity_uri: str = ""
    revisions: list[EvalRevision] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_create_opa(tools: Any, /, **kwargs: Any) -> dict[str, Any]:
    """Call ``tools.create_opa`` with ``skip_dedupe_lookup`` if supported.

    Older ``WikiBuildTools`` versions may not accept ``skip_dedupe_lookup``;
    fall back to calling without it on ``TypeError``.
    """
    try:
        return tools.create_opa(**kwargs)
    except TypeError:
        kwargs.pop("skip_dedupe_lookup", None)
        return tools.create_opa(**kwargs)


def _record_id(value: str, prefix: str) -> str:
    """Accept either a record id or a backend URI and return its id."""
    token = value.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md")
    return token if token.startswith(prefix + "-") else value


async def _notify_opl_applied(ctx: RunContext[Any], result: dict[str, Any]) -> dict[str, str]:
    """Best-effort steer notification backed by the durable event ledger."""
    event = result.get("event")
    if (
        result.get("apply_status") != "applied"
        or result.get("idempotent") is True
        or not isinstance(event, dict)
    ):
        return {"status": "skipped", "reason": "no_applied_event"}
    from wolfharness.capabilities.viking.tools import _get_session_id

    session_id = _get_session_id(ctx)
    try:
        session_pool = ctx.deps.node.host_context.session_pool
    except AttributeError:
        session_pool = None
    if session_pool is None or session_id is None:
        return {"status": "unavailable", "event_id": str(event.get("event_id", ""))}
    message = (
        "Wiki expert update applied: "
        f"event={event.get('event_id', '')}, target={event.get('target_uri', '')}, "
        f"opl={event.get('opl_uri', '')}, scopes={event.get('authority_scopes', [])}. "
        f"Continue from change-event cursor {event.get('sequence', 0)}."
    )
    try:
        message_id = await asyncio.wait_for(
            session_pool.steer_from_background_task(session_id, message),
            timeout=5,
        )
    except (TimeoutError, RuntimeError) as error:
        logger.warning("OPL apply notification failed: %s", error)
        return {
            "status": "failed",
            "event_id": str(event.get("event_id", "")),
            "error": str(error),
        }
    return {
        "status": "delivered" if message_id is not None else "queued",
        "event_id": str(event.get("event_id", "")),
        "message_id": str(message_id or ""),
    }


def _title_from_text(text: str, fallback: str) -> str:
    first_line = next(
        (line.strip() for line in text.strip().splitlines() if line.strip()),
        "",
    )
    return (first_line[:60] or fallback) if first_line else fallback


def _ticket_evidence(
    revision: EvalRevision | None,
    *,
    is_uri_valid: Callable[[str], bool] | None = None,
) -> list[str]:
    """Collect tractable evidence URIs from an eval revision.

    Combines ``cited_references[].uri`` (引用的证据) and ``evidence``
    (修改关联的uri) — both are user contracts that must flow into the
    OPA/OPS/OPL ``evidence_uris``.

    ``cited_references[].uri`` entries are trusted references and are always
    kept.  ``evidence`` entries are free-form user expressions: they may
    carry provider URIs or plain audit text (for example ``QuotedText: ...``
    or ``Matched knowledge snippet: ...``).  When *is_uri_valid* is supplied
    (the engine's ``is_valid_op_uri`` predicate), only entries that pass are
    treated as URIs; plain-text entries are dropped instead of being
    submitted as invalid evidence URIs.
    """
    if revision is None:
        return []
    cited = [ref.uri.strip() for ref in revision.cited_references if ref.uri.strip()]
    if is_uri_valid is None:
        linked = [uri.strip() for uri in revision.evidence if uri.strip()]
    else:
        linked = [
            uri.strip() for uri in revision.evidence if uri.strip() and is_uri_valid(uri.strip())
        ]
    return list(dict.fromkeys([*cited, *linked]))


async def _async_push_to_remote(tools: Any, target_uri: str, opl_uri: str) -> dict[str, object]:
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
        from openviking_sdk import AsyncHTTPClient
    except ImportError:
        return {"sync_status": "skipped", "sync_reason": "openviking-sdk not installed"}

    base_url = os.environ.get("VIKING_BASE_URL", "http://viking.ai.rootcloud.info/")
    client = AsyncHTTPClient(url=base_url, api_key=api_key)
    await client.initialize()

    result: dict[str, object] = {"sync_status": "ok"}
    errors: list[str] = []

    # Entity URIs from store.entity_uri() lack .md extension; Viking SDK requires it for create mode
    push_uri = target_uri if target_uri.endswith(".md") else target_uri + ".md"
    entity_content = await asyncio.to_thread(tools.read_resource, target_uri)
    if entity_content:
        try:
            try:
                await client.stat(push_uri)
                write_mode = "replace"
            except Exception:
                write_mode = "create"
            await client.write(push_uri, entity_content, mode=write_mode, wait=True)
            result["synced_entity_uri"] = push_uri
        except Exception as exc:
            errors.append(f"entity: {exc}")
            logger.warning("Sync entity to remote failed: %s", exc)

    if opl_uri.startswith("viking://"):
        opl_content = await asyncio.to_thread(tools.read_resource, opl_uri)
        if opl_content:
            try:
                try:
                    await client.stat(opl_uri)
                    write_mode = "replace"
                except Exception:
                    write_mode = "create"
                await client.write(opl_uri, opl_content, mode=write_mode, wait=True)
                result["synced_opl_uri"] = opl_uri
            except Exception as exc:
                errors.append(f"opl: {exc}")
                logger.warning("Sync OPL to remote failed: %s", exc)

    if errors:
        result["sync_status"] = "partial"
        result["sync_errors"] = errors

    return result


# ---------------------------------------------------------------------------
# Ticket tool closures
# ---------------------------------------------------------------------------


def _build_ticket_fns(tools: Any, *, sync_after_apply: bool = False) -> list[Callable[..., Any]]:
    """Build the six ticket tool closures capturing the tools instance."""
    # OPA/OPS ticket closures re-derive evidence URIs from the eval revision
    # dict, bypassing the provider-side adapter filter.  Capture the engine's
    # URI-validity predicate so plain-text ``evidence`` entries are not
    # blindly submitted as evidence URIs.  Falls back to the historical
    # blind merge when the engine does not expose ``is_valid_op_uri``.
    raw_predicate = getattr(tools, "is_valid_op_uri", None)
    is_uri_valid: Callable[[str], bool] | None = raw_predicate if callable(raw_predicate) else None

    async def create_opa_ticket(
        ctx: RunContext[Any],
        *,
        description: str = "",
        related_uris: list[str] | None = None,
        target_uri: str = "",
        opa_uri: str = "",
        title: str = "",
        target_section: str = "",
        finding: str = "",
        missing: str = "",
        recommendation: str = "",
        ticket: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """File or revise an OPA ticket (problem/feedback for one target).

        Creates a ``category=feedback``, ``reason_code=expert_feedback``
        OPA targeted at ``target_uri``, or — when ``opa_uri`` is given —
        revises that existing OPA record in place.

        Each submission without ``opa_uri`` creates its own distinct OPA
        record: the same problem reported again (possibly by another
        expert) is filed as a new record, never merged into a prior one.

        When ``ticket`` is provided, fields are derived from an eval
        revision: ``target.entity_uri`` → source/target,
        ``target.content_snippet`` or ``suggested_resolution`` → problem
        description, ``cited_references[].uri`` plus ``evidence``
        (修改关联的uri) → evidence URIs. Explicit args override derived
        values.

        All URIs use the ``viking://resources/<namespace>/...`` scheme.
        Create the OPA for the given ``target_uri`` directly — the returned
        ``uri`` is the audit anchor that later OPS/OPL steps must reuse.
        Do not call ``get_ticket_status`` first; the create result is the
        source of truth for what was written.

        Args:
            description: The problem / feedback content. First line
                becomes the title when ``title`` is empty.
            related_uris: ``viking://`` URI references (evidence pages,
                source chapters, cited references, 修改关联的uri).
            target_uri: ``viking://`` URI of the wiki entity this ticket
                targets (the source).
            opa_uri: ``viking://`` URI of an existing OPA ticket to
                revise instead of creating a new one.
            title: Optional short title; derived from ``description``.
            target_section: Optional page section this problem refers to.
            finding: Optional structured finding; defaults to
                ``description``.
            missing: Optional description of what is missing; defaults to
                ``description``.
            recommendation: Optional suggested fix; falls back to
                ``finding``.
            ticket: Optional ``EvalRevision``-shaped dict to derive the
                fields above from (eval revision output).

        Returns:
            Dict with ``opa_id``, ``uri``, ``title``, ``target_uri``,
            ``status``. Keep the returned ``uri`` for revision.
        """
        revision = EvalRevision.model_validate(ticket) if ticket is not None else None
        target = revision.target if revision is not None else None
        derived_evidence = _ticket_evidence(revision, is_uri_valid=is_uri_valid)
        description = description or (
            (target.content_snippet if target is not None and target.content_snippet else "")
            or (revision.suggested_resolution if revision is not None else "")
            or ""
        )
        effective_related = list(
            dict.fromkeys(uri.strip() for uri in (related_uris or []) if uri.strip()),
        )
        evidence = list(dict.fromkeys([*derived_evidence, *effective_related]))
        effective_target = target_uri or (target.entity_uri if target is not None else "")
        effective_section = target_section or (
            target.matched_section_title if target is not None else ""
        )
        effective_title = (
            title.strip()
            or (revision.ticket_id if revision is not None and revision.ticket_id else "")
            or _title_from_text(description, "专家反馈")
        )
        opa_id = (
            _record_id(opa_uri, "opa")
            if opa_uri
            else (
                _record_id(revision.ticket_id, "opa")
                if revision is not None and revision.ticket_id
                else ""
            )
        )
        effective_finding = finding or description
        return await asyncio.to_thread(
            _safe_create_opa,
            tools,
            opa_id=opa_id,
            title=effective_title,
            description=description,
            category="feedback",
            reason_code="expert_feedback",
            target_uri=effective_target,
            target_section=effective_section,
            evidence_uris=evidence,
            status="pending",
            finding=effective_finding,
            missing=missing or description,
            recommendation=recommendation or effective_finding,
            skip_dedupe_lookup=True,
        )

    async def create_ops_ticket(
        ctx: RunContext[Any],
        *,
        suggestion: str = "",
        related_uris: list[str] | None = None,
        parent_opa: str = "",
        ops_uri: str = "",
        title: str = "",
        expert_id: str = "",
        expert_name: str = "",
        ticket: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """Submit or revise an OPS ticket (expert recommendation / draft).

        When ``ops_uri`` is provided, the existing OPS record is revised
        in place; otherwise a new OPS is created under ``parent_opa`` as
        an unconfirmed draft.

        When ``ticket`` is provided, the suggestion and evidence URIs are
        derived from an eval revision: ``expert_opinion`` /
        ``suggested_resolution`` → suggestion, ``cited_references[].uri``
        plus ``evidence`` (修改关联的uri) → evidence URIs.

        Create or revise the OPS under ``parent_opa`` — the ``viking://`` URI
        returned by ``create_opa_ticket``. Do not call ``get_ticket_status``
        to rediscover the parent; pass the returned URI directly.

        Args:
            suggestion: The expert's recommendation. First line becomes
                the title; remaining lines become the analysis.
            related_uris: ``viking://`` URI references (evidence —
                cited_references from an eval revision).
            parent_opa: OPA id or ``viking://`` URI this ticket attaches
                to. Required when creating a new record.
            ops_uri: ``viking://`` URI of an existing OPS to revise
                instead of creating a new one.
            title: Optional short title; derived from ``suggestion``.
            expert_id: Expert identifier recorded on the draft.
            expert_name: Expert display name recorded on the draft.
            ticket: Optional ``EvalRevision``-shaped dict to derive the
                suggestion / evidence from.

        Returns:
            Dict with ``ops_id``, ``uri``, ``parent_opa``,
            ``target_uri``, ``status``. Keep the returned ``uri`` for
            revision.
        """
        revision = EvalRevision.model_validate(ticket) if ticket is not None else None
        suggestion = suggestion or (
            (revision.expert_opinion if revision is not None else "")
            or (revision.suggested_resolution if revision is not None else "")
            or ""
        )
        derived_evidence = _ticket_evidence(revision, is_uri_valid=is_uri_valid)
        effective_related = list(
            dict.fromkeys(uri.strip() for uri in (related_uris or []) if uri.strip()),
        )
        evidence = list(dict.fromkeys([*derived_evidence, *effective_related]))
        lines = [line.strip() for line in suggestion.strip().splitlines() if line.strip()]
        effective_title = (
            title.strip()
            or (revision.ops_ref if revision is not None and revision.ops_ref else "")
            or _title_from_text(suggestion, "专家建议")
        )
        effective_analysis = "\n".join(lines[1:]) if len(lines) > 1 else suggestion.strip()
        return await asyncio.to_thread(
            tools.ingest_external_ops,
            parent_opa=parent_opa,
            title=effective_title,
            analysis=effective_analysis,
            solution=suggestion.strip(),
            evidence_uris=evidence,
            expert_id=expert_id,
            expert_name=expert_name,
            ops_uri=ops_uri,
        )

    async def update_ops_ticket(
        ctx: RunContext[Any],
        *,
        ops_uri: str,
        title: str = "",
        analysis: str = "",
        solution: str = "",
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        status: str = "",
        reviewed_by: str = "",
        review_notes: str = "",
    ) -> dict[str, Any]:
        """Patch one OPS ticket in place, leaving untouched fields intact.

        A true patch, unlike ``create_ops_ticket`` with ``ops_uri`` (which
        rewrites the record from the full ``suggestion`` text). Only the
        fields you pass are changed; every omitted field keeps its current
        value. This is the capability's external-facing update surface,
        mirroring the engine's ``update_ops``.

        Pass ``status`` to transition the draft — ``unconfirmed`` |
        ``confirmed`` | ``rejected``; confirming or rejecting requires
        ``reviewed_by``. Reuse the ``ops_uri`` returned by
        ``create_ops_ticket``; do not rediscover it via ``get_ticket_status``.

        Args:
            ops_uri: ``viking://`` URI of the existing OPS ticket.
            title: New short title (kept when omitted).
            analysis: New analysis body (kept when omitted).
            solution: New expert solution text (kept when omitted).
            evidence_uris: New ``viking://`` evidence references.
            related_uris: New ``viking://`` related references.
            candidate_content: New full replacement markdown candidate.
            candidate_operations: List of deterministic patch operations. Each
                is a dict with an ``op`` key; supported ops:
                - Content patch ops (mutually exclusive with
                  ``candidate_content``): ``line_replace``/``line_insert``/
                  ``line_delete``, ``section_replace``/``section_insert_after``,
                  ``fm_append``/``fm_set``/``fm_set_list``.
                - Storage relocation op ``move_entity``:
                  ``{"op": "move_entity", "dst_class_name": "...",
                  "dst_object_name": "..."}``. Relocates the target entity file
                  to ``Component/<dst_class_name>/<dst_object_name>.md``
                  (omitted values carry over the current class/name). The
                  target file is physically moved (neither
                  ``candidate_content`` nor ``expected_sha256`` is used); an
                  emptied source class directory is pruned, and the old URI is
                  kept via a redirect. Use this for class/prefix (BOM 归类)
                  changes, not content edits.
            expected_sha256: New locked-content SHA-256 for apply-time
                optimistic locking (required for content patches on an
                existing target; not needed for a pure ``move_entity``).
            status: New status — ``unconfirmed`` | ``confirmed`` |
                ``rejected``.
            reviewed_by: Reviewer identifier; required to confirm/reject.
            review_notes: Notes recorded with a status transition.

        Returns:
            Dict with ``ops_id``, ``uri``, ``parent_opa``, ``target_uri``,
            ``status``. Keep the returned ``uri`` for further patches.
        """
        ops_id = _record_id(ops_uri, "ops")
        return await asyncio.to_thread(
            tools.update_ops,
            ops_id,
            title=title or None,
            analysis=analysis or None,
            solution=solution or None,
            evidence_uris=evidence_uris,
            related_uris=related_uris,
            candidate_content=candidate_content or None,
            candidate_operations=candidate_operations,
            expected_sha256=expected_sha256 or None,
            status=status or None,
            reviewed_by=reviewed_by,
            review_notes=review_notes,
        )

    async def create_opl_ticket(
        ctx: RunContext[Any],
        *,
        ticket: dict[str, object] | None = None,
        parent_opa: str,
        ops_uris: list[str],
        proposal: str = "",
        rationale: str = "",
        title: str = "",
        evidence_uris: list[str] | None = None,
        related_uris: list[str] | None = None,
        target_uri: str = "",
        candidate_content: str = "",
        candidate_operations: list[dict[str, object]] | None = None,
        expected_sha256: str = "",
        expert_id: str = "",
        expert_name: str = "",
        opl_uri: str = "",
    ) -> dict[str, Any]:
        """Integrate one OPA and its OPS records into an OPL ticket.

        Each referenced OPS is confirmed (``reviewed_by`` = expert or
        ``external_ticket``) and then archived into a single OPL
        snapshot.  ``candidate_content`` (full replacement markdown) or
        ``candidate_operations`` (deterministic patch ops) is what
        ``apply_opl_ticket`` will later merge into the Wiki.

        When ``ticket`` is provided, ``proposal`` (suggested_resolution),
        ``rationale`` (expert_opinion + content_snippet) and
        ``evidence_uris`` (cited_references + evidence, 修改关联的uri) are
        derived from an eval revision.

        Args:
            ticket: Optional ``EvalRevision``-shaped dict (eval revision).
            parent_opa: OPA id or ``viking://`` URI the ticket attaches
                to.
            ops_uris: ``viking://`` URIs of the OPS records to integrate.
            proposal: The proposed knowledge change.
            rationale: Why the change is correct (expert opinion /
                source reasoning).
            title: Optional short title; derived from ``proposal``.
            evidence_uris: Extra ``viking://`` evidence beyond what the
                OPA/OPS already carry.
            related_uris: ``viking://`` related URIs.
            target_uri: Must match the OPA's ``target_uri`` when given.
            candidate_content: Optional full replacement page markdown.
            candidate_operations: Optional list of deterministic patch
                operations. Each is a dict with an ``op`` key; supported ops:
                - Content patch ops (mutually exclusive with
                  ``candidate_content``): ``line_replace``/``line_insert``/
                  ``line_delete``, ``section_replace``/``section_insert_after``,
                  ``fm_append``/``fm_set``/``fm_set_list``.
                - Storage relocation op ``move_entity``:
                  ``{"op": "move_entity", "dst_class_name": "...",
                  "dst_object_name": "..."}``. Physically relocates the target
                  entity file to ``Component/<dst_class_name>/<dst_object_name>.md``
                  (omitted values carry over the current class/name); the
                  emptied source class directory is pruned and the old URI kept
                  via a redirect. Use this for class/prefix (BOM 归类) changes,
                  not content edits. A pure ``move_entity`` carries no
                  ``candidate_content`` and needs no ``expected_sha256``.
            expected_sha256: SHA-256 hex of current target content (for
                optimistic locking at apply time; required for content patches
                on an existing target, not needed for a pure ``move_entity``).
            expert_id: Expert identifier (also the OPS reviewer).
            expert_name: Expert display name.
            opl_uri: ``viking://`` URI of an existing OPL to revise.

        Returns:
            Dict with ``opl_id``, ``uri``, ``parent_opa``, ``ops_uris``,
            ``target_uri``, ``status``, ``apply_status``.
        """
        revision = EvalRevision.model_validate(ticket) if ticket is not None else None
        target = revision.target if revision is not None else None
        proposal = proposal or (revision.suggested_resolution if revision is not None else "")
        if not rationale.strip() and revision is not None:
            rationale = "".join(
                piece
                for piece in (revision.expert_opinion, target.content_snippet if target else "")
                if piece
            )
        effective_derived_evidence = _ticket_evidence(revision, is_uri_valid=is_uri_valid)
        effective_evidence = list(
            dict.fromkeys(
                [
                    *effective_derived_evidence,
                    *(uri.strip() for uri in (evidence_uris or []) if uri.strip()),
                ],
            ),
        )
        effective_target_uri = target_uri or (target.entity_uri if target is not None else "")
        effective_title = title.strip() or _title_from_text(proposal, "专家提案")
        ops_ids = [_record_id(uri, "ops") for uri in ops_uris if uri.strip()]
        reviewer = expert_id.strip() or "external_ticket"
        for ops_id in ops_ids:
            await asyncio.to_thread(
                tools.update_ops,
                ops_id,
                status="confirmed",
                reviewed_by=reviewer,
                review_notes=f"Confirmed by ticket OPL: {expert_name or expert_id}",
            )
        opl_id = _record_id(opl_uri, "opl") if opl_uri else ""
        return await asyncio.to_thread(
            tools.create_opl,
            parent_opa=parent_opa,
            ops_uris=ops_uris,
            title=effective_title,
            proposal=proposal,
            rationale=rationale,
            evidence_uris=effective_evidence,
            related_uris=related_uris,
            target_uri=effective_target_uri,
            opl_id=opl_id,
            source_type="external_expert",
            expert_id=expert_id,
            expert_name=expert_name,
            candidate_content=candidate_content,
            candidate_operations=candidate_operations,
            expected_sha256=expected_sha256,
        )

    async def apply_opl_ticket(
        ctx: RunContext[Any],
        *,
        ticket: dict[str, object] | None = None,
        opl_uri: str,
        expert_id: str = "",
        expert_name: str = "",
    ) -> dict[str, Any]:
        """Apply an OPL ticket to the Wiki and close its ticket chain.

        Merges the stored ``candidate_content`` / ``candidate_operations``
        into the target entity (ownership enforced by the engine), resolves
        the parent OPA as ``closed``, persists a cursor-addressable change
        event, notifies the current agent session when available, and — when
        the capability was built with ``sync_after_apply`` — pushes the patched page and OPL to
        remote Viking.  Returns the terminal status for the caller.

        Args:
            ticket: Optional ``EvalRevision``-shaped dict (used for
                traceability in the returned status; not required to
                apply an already-created OPL).
            opl_uri: ``viking://`` URI of the OPL ticket to apply
                (returned by ``create_opl_ticket``).
            expert_id: Expert identifier (recorded when closing the OPA).
            expert_name: Expert display name.

        Returns:
            Dict with ``opl_id``, ``status``, ``apply_status``,
            ``entity_uri``, ``opa`` (closed record), ``event``,
            ``notification`` and — when sync ran — ``sync``.
        """
        opl_id = _record_id(opl_uri, "opl")
        result: dict[str, Any] = await asyncio.to_thread(tools.apply_opl, opl_id)
        opl_rows = await asyncio.to_thread(tools.get_opls, limit=200)
        opl_row = next(
            (row for row in opl_rows if row["opl_id"] == opl_id),
            None,
        )
        if result.get("apply_status") == "applied" and opl_row is not None:
            parent_id = _record_id(str(opl_row.get("parent_opa", "")), "opa")
            if parent_id:
                entity_uri = str(result.get("entity_uri", ""))
                with contextlib.suppress(FileNotFoundError):
                    result["opa"] = await asyncio.to_thread(
                        tools.resolve_opa,
                        parent_id,
                        solution=str(opl_row.get("proposal", "")),
                        evidence_uris=[entity_uri] if entity_uri else None,
                        closure_status="closed",
                        closure_reason="applied via ticket OPL",
                    )
        result["notification"] = await _notify_opl_applied(ctx, result)
        if sync_after_apply and result.get("apply_status") == "applied":
            entity_uri = str(result.get("entity_uri", "")) or (
                str(opl_row.get("target_uri", "")) if opl_row else ""
            )
            opl_uri_effective = str(opl_row.get("uri", opl_uri)) if opl_row else opl_uri
            result["sync"] = await _async_push_to_remote(
                tools,
                entity_uri,
                opl_uri_effective,
            )
        return result

    async def get_ticket_status(
        ctx: RunContext[Any],
        *,
        parent_opa: str = "",
        target_uri: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Read the current state of OPA / OPS / OPL tickets.

        Filters OPA records by ``target_uri`` and OPS/OPL records by
        ``parent_opa``.  Always includes the engine-level OP flow report
        (``open_opa_count`` / ``review_pending``).

        Args:
            parent_opa: OPA id or ``viking://`` URI to scope OPS/OPL
                records to.
            target_uri: ``viking://`` entity URI to scope OPA records to.
            limit: Maximum records per category (1..200).

        Returns:
            Dict with ``opas``, ``ops``, ``opls``, and the merged
            ``op_flow`` state.
        """
        opas = await asyncio.to_thread(tools.get_opas, target_uri=target_uri, limit=limit)
        ops = await asyncio.to_thread(
            tools.get_ops, parent_opa=parent_opa, target_uri=target_uri, limit=limit
        )
        opls = await asyncio.to_thread(
            tools.get_opls, parent_opa=parent_opa, target_uri=target_uri, limit=limit
        )
        # op_flow_status is a global scan (get_opas 500 + get_ops 10000) — skip
        # it when a single target_uri is queried; it's irrelevant to per-target
        # ticket lookup and dominates latency on remote backends.
        if target_uri:
            op_flow: dict[str, Any] = {}
        else:
            op_flow = await asyncio.to_thread(tools.op_flow_status, limit=limit)
        authority = await asyncio.to_thread(
            tools.get_expert_authority,
            target_uri=target_uri,
            limit=limit,
        )
        events = await asyncio.to_thread(
            tools.get_wiki_change_events,
            target_uri=target_uri,
            limit=limit,
        )
        return {
            "opas": opas,
            "ops": ops,
            "opls": opls,
            "op_flow": op_flow,
            "expert_authority": authority,
            "events": events,
        }

    async def submit_eval_payload(
        ctx: RunContext[Any],
        *,
        payload: dict[str, Any],
        expert_id: str = "",
        expert_name: str = "",
    ) -> dict[str, Any]:
        """Sediment an eval payload as OPA + OPS-draft tickets (链路 C).

        Pure ingestion: each ``revisions[]`` entry is materialized through
        the engine — ``target.entity_uri`` is the source,
        ``cited_references[].uri`` plus ``evidence`` (修改关联的uri) the
        evidence, ``content_snippet`` the OPA problem statement and
        ``expert_opinion`` the expert assessment.

        An OPA ticket is always created. An OPS draft is created only when
        the revision carries an actionable solution (a non-empty
        ``suggested_resolution``); otherwise the ticket stays as OPA only
        (no ``ops`` entry in its record). The capability never judges
        whether an assessment is actionable — that decision stays on the
        business side, which decides whether to supply a
        ``suggested_resolution``.

        This is sedimentation only: no OPL snapshot, confirmation or apply
        happens here. Iterating the draft (``create_ops_ticket`` with
        ``ops_uri``), integrating into an OPL proposal
        (``create_opl_ticket``) and applying to the Wiki
        (``apply_opl_ticket``) are explicit frontend-driven steps.

        Args:
            payload: An ``EvalPayload``-shaped dict:
                ``{"eval", "round", "entity_uri", "revisions": [...]}``.
            expert_id: Expert identifier recorded on the sedimented tickets.
            expert_name: Expert display name recorded on the sedimented
                tickets.

        Returns:
            Dict with ``round``, ``eval``, ``tickets`` (one entry per
            revision: ticket id, kind, OPA/OPS ids and URIs, statuses).
        """
        payload_obj = EvalPayload.model_validate(payload)
        records: list[dict[str, Any]] = []
        for revision in payload_obj.revisions:
            target = revision.target
            target_uri = (target.entity_uri if target is not None else "") or payload_obj.entity_uri
            evidence_uris = _ticket_evidence(revision, is_uri_valid=is_uri_valid)
            description = (
                (target.content_snippet if target is not None else "")
                or revision.suggested_resolution
                or revision.expert_opinion
                or revision.ticket_id
            )
            opa_result = await create_opa_ticket(
                ctx,
                ticket=revision.model_dump(mode="json"),
                description=description,
                related_uris=evidence_uris,
                target_uri=target_uri,
                title=revision.ticket_id,
                finding=description,
                missing=description,
                recommendation=revision.suggested_resolution,
            )
            record: dict[str, Any] = {
                "ticket_id": revision.ticket_id,
                "kind": revision.kind or "OPA",
                "annotation_ref": revision.annotation_ref,
                "source": revision.source,
                "status": revision.status,
                "opa": {
                    "opa_id": opa_result["opa_id"],
                    "uri": opa_result["uri"],
                    "target_uri": opa_result.get("target_uri", target_uri),
                },
            }
            if revision.suggested_resolution.strip():
                ops_result = await create_ops_ticket(
                    ctx,
                    ticket=revision.model_dump(mode="json"),
                    suggestion=revision.suggested_resolution.strip(),
                    related_uris=evidence_uris,
                    parent_opa=str(opa_result["opa_id"]),
                    title=revision.ops_ref or "",
                    expert_id=expert_id,
                    expert_name=expert_name,
                )
                record["ops"] = {
                    "ops_id": ops_result["ops_id"],
                    "uri": ops_result["uri"],
                    "parent_opa": ops_result.get("parent_opa", ""),
                }
            records.append(record)
        return {
            "eval": payload_obj.eval,
            "round": payload_obj.round,
            "entity_uri": payload_obj.entity_uri,
            "ticket_count": len(records),
            "tickets": records,
        }

    async def find_op(
        ctx: RunContext[Any],
        *,
        query: str = "",
        prefix: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """Fast precise lookup over ticket & component identifiers in the wiki store.

        Ticket-domain search — directly uses the OpenViking client grep
        primitive against the configured wiki root, NOT the slower semantic
        ``find_wiki``. Resolves a ``bom_path`` prefix, an ``object_name``, a
        class keep, or a ``OP/`` ticket id before assembling a candidate.
        Each match carries the matched line content (e.g. the full
        ``bom_path: ...`` value) and its URI.
        """
        if not query and not prefix:
            return {"entries": [], "error": "find_op requires query or prefix"}
        _fs = getattr(getattr(tools, "store", None), "_fs", None)
        if _fs is None:
            return {"entries": [], "error": "wiki store not available"}
        client = getattr(_fs, "_client", None)
        target = str(getattr(_fs, "root_uri", "") or "").rstrip("/")
        entries: list[dict[str, object]] = []

        if client is not None:
            result = await asyncio.to_thread(
                client.grep,
                target or "",
                query or prefix,
                node_limit=limit,
            )
            for match in (result or {}).get("matches", []) or []:
                uri = str(match.get("uri", "")) if isinstance(match, dict) else ""
                if not uri or not uri.startswith(target + "/"):
                    continue
                leaf = uri.rsplit("/", 1)[-1]
                if leaf in _OP_METADATA_FILENAMES:
                    continue
                entries.append(
                    {
                        "uri": uri,
                        "content": match.get("content", "") if isinstance(match, dict) else "",
                    },
                )
                if len(entries) >= limit:
                    break
            return {"entries": entries}
        keys = await asyncio.to_thread(_fs.list_dir, target, recursive=True)
        for key in keys:
            if prefix and not str(key).startswith(prefix):
                continue
            entries.append({"key": str(key), "uri": f"{target}/{key}"})
            if len(entries) >= limit:
                break
        return {"entries": entries}

    async def read_resource(
        ctx: RunContext[Any],
        *,
        uri: str,
    ) -> dict[str, Any]:
        """Read a wiki entity page by its ``viking://`` URI.

        Returns the page content and its SHA-256 hash so the caller can
        prepare ``candidate_content`` for ``create_opl_ticket`` and use
        the hash as ``expected_sha256`` for optimistic locking at apply
        time.

        Args:
            uri: ``viking://`` URI of the wiki entity to read.

        Returns:
            Dict with ``uri``, ``content`` (page markdown), and
            ``sha256`` (hex digest of the UTF-8 encoded content).
        """
        content = await asyncio.to_thread(tools.read_resource, uri)
        if not content and uri.startswith("viking://resources/"):
            from wolfharness.capabilities.wiki.storage import async_viking_read

            content = await async_viking_read(uri)
        if not content:
            return {"uri": uri, "content": "", "sha256": "", "error": "resource not found"}
        return {
            "uri": uri,
            "content": content,
            "sha256": sha256(content.encode("utf-8")).hexdigest(),
        }

    return [
        read_resource,
        create_opa_ticket,
        create_ops_ticket,
        update_ops_ticket,
        create_opl_ticket,
        apply_opl_ticket,
        get_ticket_status,
        submit_eval_payload,
        find_op,
    ]


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def build_ticket_tools(cap: WikiBuildCapability) -> list[Callable[..., Any]]:
    """Build the ticket tool closures for the WikiBuildCapability.

    Args:
        cap: The ``WikiBuildCapability`` instance that owns the tools.

    Returns:
        A list of async tool functions suitable for ``FunctionToolset``.
    """
    cap._ensure_tools()
    tools = cap.tools
    if tools is None:
        return []
    return _build_ticket_fns(tools, sync_after_apply=cap.config.sync_after_apply)


def get_ticket_instructions() -> str:
    """Return guidance for an agent using the ticket surface.

    Returns:
        A multi-line instruction block describing the seven ticket tools
        and the recommended OPA → OPS → OPL → apply flow.
    """
    return (
        "You are a wiki ticket interface. All wiki resources use the "
        "viking:// URI scheme (e.g. viking://resources/814/Component/...).\n"
        "\n"
        "Discovery:\n"
        "- read_resource(uri) — read a wiki entity page; returns content "
        "and sha256. Use the content to prepare candidate_content for "
        "create_opl_ticket and the sha256 as expected_sha256.\n"
        "- get_ticket_status(target_uri=...) — check existing OPA/OPS/OPL "
        "tickets, expert authority claims, and recent apply events before "
        "submitting new ones.\n"
        "- get_expert_authority(target_uri=...) — inspect sections/fields "
        "protected by confirmed or applied expert knowledge.\n"
        "- get_wiki_change_events(after_sequence=...) — consume durable "
        "apply notifications after reconnect/restart.\n"
        "\n"
        "Ticket tools:\n"
        "1. create_opa_ticket — file a problem/feedback ticket (OPA) with a "
        "description and related viking:// URIs. Pass opa_uri to revise.\n"
        "2. create_ops_ticket — attach an expert recommendation (OPS) to an "
        "OPA: suggestion text + related URIs. Pass ops_uri to revise the "
        "same recommendation until accepted.\n"
        "3. update_ops_ticket — patch an existing OPS in place (only the "
        "fields you pass change). Pass status to confirm or reject the "
        "draft, with reviewed_by.\n"
        "4. create_opl_ticket — integrate one OPA + its OPS into an OPL "
        "proposal. Provide candidate_content (full replacement markdown) "
        "and expected_sha256 (SHA-256 of current page content, computed "
        "from read_resource output) when the patch is machine-applicable. "
        "For a class/prefix (BOM 归类) change, pass candidate_operations "
        "instead: [{'op': 'move_entity', 'dst_class_name': '...', "
        "'dst_object_name': '...'}] to physically relocate the target file "
        "to Component/<dst_class_name>/<dst_object_name>.md (no "
        "candidate_content/expected_sha256; emptied source dir pruned, old "
        "URI kept via redirect).\n"
        "5. apply_opl_ticket — merge the OPL candidate into the target page, "
        "activate scoped expert authority, emit a durable event, notify the "
        "current agent, and close the ticket chain.\n"
        "6. get_ticket_status — read OPS/OPL lifecycle state plus authority "
        "and durable event state.\n"
        "7. submit_eval_payload — ingest a full eval revision payload "
        "({eval, round, entity_uri, revisions: [...]}) and materialize "
        "OPA/OPS tickets from it. Per revision: target.entity_uri → OPA "
        "target_uri, content_snippet → OPA problem description, "
        "suggested_resolution → OPA recommendation and OPS suggestion, "
        "cited_references[].uri + evidence → OPA/OPS evidence_uris, "
        "expert_opinion → fallback description. An empty "
        "suggested_resolution leaves the ticket OPA-only (no OPS).\n"
        "\n"
        "Typical flow:\n"
        "  find_wiki → read_resource(target) → create_opa_ticket → "
        "create_ops_ticket → create_opl_ticket → apply_opl_ticket.\n"
        "Each submission tool returns a URI; keep it and pass it back on "
        "the next revision."
    )
