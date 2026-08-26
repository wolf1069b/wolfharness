"""WikiBuildTools — composition root for wiki build tools.

Split from the original mcp_server.py into mixins. This module defines the
core class composing all mixins and containing init/lifecycle, hashing,
validation, and doc-prefix methods.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
import logging
import os
from pathlib import Path
import re
from threading import RLock
import time
from urllib.parse import unquote, urlsplit

from httpx import HTTPError
from openviking_sdk.errors import OpenVikingError

from wolfharness.capabilities.wiki.auto_repair import batch_auto_repair
from wolfharness.capabilities.wiki.quality import (
    BuildProfile,
    RawSourceKind,
    SourceReadResult,
    SourceReadStatus,
    WikiAuditReport,
    classify_raw_source_uri,
    entity_status,
    extract_malformed_wiki_uris,
    extract_source_uris,
    extract_wiki_uris,
    parse_frontmatter,
    set_raw_source_root_uri,
)
from wolfharness.capabilities.wiki.schema_loader import get_concept_schema, get_schema_version
from wolfharness.capabilities.wiki.storage import (
    FSBackend,
    create_raw_reader,
    create_wiki_store,
    viking_read,
)
from wolfharness.capabilities.wiki.validation import (
    require_valid_entity,
)

from .io.audit import AuditCache, AuditMixin
from .io.children import ChildrenMixin
from .io.model_mapping import ModelMappingMixin
from .io.text_parsers import TextParsersMixin
from .tickets.opa import OPAMixin


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from ._helpers import _FORMAL_WRITE_HOOKS, _io_worker_limit
from .entities.entities import EntityWriteMixin
from .entities.finalize import FinalizeMixin
from .entities.packets import PacketMixin
from .entities.patches import PatchMixin
from .io.migration import MigrationMixin
from .planning.bom import BomMixin
from .planning.chapters import ChapterMixin
from .planning.materialization import MaterializationMixin
from .planning.relations import RelationMixin


if TYPE_CHECKING:
    from collections.abc import Callable

    from wolfharness.capabilities.wiki.build_logger import WikiBuildLogger


class WikiBuildTools(
    # New mixins first to avoid WikiBuildDeps Protocol shadowing in MRO
    PacketMixin,
    MaterializationMixin,
    ChapterMixin,
    RelationMixin,
    EntityWriteMixin,
    PatchMixin,
    BomMixin,
    MigrationMixin,
    FinalizeMixin,
    # Existing mixins (inherit from WikiBuildDeps Protocol)
    ModelMappingMixin,
    AuditMixin,
    ChildrenMixin,
    OPAMixin,
    TextParsersMixin,
):
    """All wiki build I/O operations as callable methods.

    One instance owns a ``WikiStore`` (write target) and a library root
    (chapter source).  Every method is self-contained and can be trivially
    exposed by an in-process capability or a compatibility adapter.
    """

    _relation_lock_registry_guard = RLock()
    _relation_lock_registry: dict[str, RLock] = {}

    def __init__(
        self,
        wiki_root: str | Path,
        library_root: str | Path,
        *,
        build_logger: WikiBuildLogger | None = None,
        case_root: str | Path | None = None,
        faultannotated_root: str | Path | None = None,
        bom_root: str | Path | None = None,
    ) -> None:
        self.store = create_wiki_store(wiki_root)
        # In Viking mode the raw library is a resource namespace, not a local
        # filesystem path.  Keeping the URI as a Path silently turns
        # ``viking://resources/<raw_namespace>/`` into a local-looking path and
        # makes manifest discovery report an empty library even though the
        # remote chapters are available through ``_raw_fs``.
        self._library_root_uri = str(library_root)
        self._library_root = (
            None if self._library_root_uri.startswith("viking://") else Path(library_root).resolve()
        )
        self._raw_fs = create_raw_reader(library_root)
        set_raw_source_root_uri(self._raw_fs.root_uri)
        # Global BOM registry (e.g. ``viking://resources/<bom_namespace>/bom/component/挖掘机部件_BOM_清单``)
        # is a separate read-only namespace spanning every machine class; keep a
        # dedicated reader so Phase 0 can scan the full remote BOM tree.
        self._bom_fs: FSBackend | None = None
        if bom_root is not None and str(bom_root).strip():
            self._bom_fs = create_raw_reader(bom_root)
        # The runner exposes the build scope as WIKI_EXPECTED_DOCS. Treat it
        # as the build source allowlist when the explicit source setting is
        # absent, so unrelated manuals are not reported as missing coverage.
        configured_docs = os.environ.get("WIKI_SOURCE_DOCS", "") or os.environ.get(
            "WIKI_EXPECTED_DOCS",
            "",
        )
        self._source_doc_allowlist = tuple(
            sorted({doc_id.strip() for doc_id in configured_docs.split(",") if doc_id.strip()}),
        )
        self._log = build_logger
        self._case_root = Path(case_root).resolve() if case_root is not None else None
        self._faultannotated_root = (
            Path(faultannotated_root).resolve() if faultannotated_root is not None else None
        )
        self._recover_finalize_transactions()
        # Raw citations are resolved many times during a full audit.  Keep the
        # immutable chapter map and resolution result in-process so one bad
        # model-generated URI cannot trigger a full JSON parse and chapter
        # tree walk for every referencing entity.
        self._chapter_map_cache: dict[str, dict[str, str]] = {}
        self._chapter_path_aliases: dict[tuple[str, str], str] = {}
        self._raw_doc_prefixes: dict[str, str] = {}
        # Lazy doc_id → path-prefix index over the raw namespace. Built once
        # by _build_doc_prefix_index; lets list_chapters resolve nested docs
        # without blind DFS.
        self._doc_prefix_index: dict[str, str] | None = None
        self._root_is_doc: bool = False
        self._chapters_cache: dict[str, list[dict[str, str]]] = {}
        # Chapter batch plans are built once per build scope and then served
        # from memory; the persisted copy is only the restart fallback.
        self._chapter_plan_cache: dict[str, list[dict[str, object]]] = {}
        self._library_doc_ids_cache: tuple[str, ...] | None = None
        self._raw_resource_cache: dict[str, str | None] = {}
        # ponytail: source_snapshot does N HTTP reads per doc; raw content
        # is normally immutable during a build session, so cache by doc_ids
        # tuple.  The fingerprint map below keeps the cache correct when an
        # incremental build observes a changed, added, or removed chapter.
        self._source_snapshot_cache: dict[tuple[str, ...], dict[str, str]] = {}
        self._source_snapshot_fingerprints: dict[
            tuple[str, ...],
            dict[str, tuple[int | None, int | None]],
        ] = {}
        self._raw_source_hash_cache: dict[str, str | None] = {}
        # ponytail: simple audit cache — invalidated on any write op.
        # avoids re-scanning 1000+ entities when conductor paginates
        # issues by different code/concept filters.
        self._audit_cache: AuditCache | None = None
        # Relation workers may run in parallel, but reverse narrative links
        # are a read-modify-write against one Component page. Serialize that
        # critical section so one worker cannot overwrite another worker's
        # newly materialized links.
        with self._relation_lock_registry_guard:
            self._relation_sync_lock = self._relation_lock_registry.setdefault(
                self.store.root_uri,
                RLock(),
            )

    def _finalize_transaction_key(self, build_id: str) -> str:
        """Return the legacy rollback key used by pre-atomic builds."""
        return f".staging/finalize/{build_id}"

    def _read_finalize_transaction(self, txn_key: str) -> dict[str, object] | None:
        """Read one durable finalize transaction, ignoring malformed debris."""
        transaction = self.store.read_json(f"{txn_key}/transaction.json")
        if transaction is None:
            return None
        return transaction if isinstance(transaction, dict) else None

    def _restore_finalize_transaction(self, txn_key: str, transaction: dict[str, object]) -> None:
        """Restore all files captured by a prepared finalize transaction."""
        targets = transaction.get("targets")
        if not isinstance(targets, list):
            raise TypeError(f"Invalid finalize transaction targets: {txn_key}")
        backup_key = f"{txn_key}/backup"
        for raw_target in targets:
            if not isinstance(raw_target, dict):
                raise TypeError(f"Invalid finalize transaction target: {txn_key}")
            relative = raw_target.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"Invalid finalize transaction path: {txn_key}")
            backup = raw_target.get("backup")
            if isinstance(backup, str) and backup:
                backup_content = self.store.read_text(f"{backup_key}/{backup}")
                if backup_content is None:
                    raise FileNotFoundError(f"Finalize rollback backup is missing: {backup}")
                self.store.write_text(relative, backup_content)
            elif self.store.exists(relative):
                self.store.delete(relative)

    def _mark_finalize_transaction(self, txn_key: str, state: str, **extra: str) -> None:
        """Update a transaction state atomically after commit or recovery."""
        transaction = self._read_finalize_transaction(txn_key)
        if transaction is None:
            return
        transaction["state"] = state
        transaction.update(extra)
        self.store.write_json(f"{txn_key}/transaction.json", transaction)

    def _recover_finalize_transactions(self) -> int:
        """Recover interrupted finalize operations before serving the wiki."""
        root_key = ".staging/finalize"
        keys = self.store.list_dir(root_key, recursive=True)
        transaction_keys = {
            key.removesuffix("/transaction.json")
            for key in keys
            if key.endswith("/transaction.json")
        }
        recovered = 0
        for txn_key in sorted(transaction_keys):
            transaction = self._read_finalize_transaction(txn_key)
            if transaction is None or transaction.get("state") != "prepared":
                continue
            self._restore_finalize_transaction(txn_key, transaction)
            self._mark_finalize_transaction(txn_key, "recovered", reason="interrupted_finalize")
            raw_input_docs = transaction.get("input_docs", [])
            recovered_input_docs = (
                tuple(str(doc_id) for doc_id in raw_input_docs if str(doc_id).strip())
                if isinstance(raw_input_docs, list)
                else ()
            )
            self.checkpoint_build(
                str(transaction.get("doc_id", "")),
                str(transaction.get("device_id", "")),
                str(transaction.get("series_id", "")),
                "recovered",
                build_id=str(transaction.get("build_id", "")),
                input_hash=str(transaction.get("input_hash", "")),
                config_hash=str(transaction.get("config_hash", "")),
                snapshot_id=str(transaction.get("snapshot_id", "")),
                source_snapshot_id=str(transaction.get("source_snapshot_id", "")),
                audit_profile=self._validate_audit_profile(
                    str(transaction.get("audit_profile", "manual")),
                ),
                input_docs=recovered_input_docs,
                schema_version=get_schema_version(),
            )
            recovered += 1
            logger.warning("Recovered interrupted Wiki finalize transaction: %s", txn_key)
        return recovered

    def recover_build(self) -> dict[str, int]:
        """Recover interrupted finalize work and return the recovery count."""
        return {"recovered_transactions": self._recover_finalize_transactions()}

    def _invalidate_audit_cache(self) -> None:
        self._audit_cache = None

    def _record_phase_timing(self, phase: str, started: float) -> None:
        """Persist one measured Wiki-side phase duration when logging is enabled."""
        if self._log:
            self._log.phase_timing(phase, (time.perf_counter() - started) * 1000)

    @staticmethod
    def _validate_audit_profile(profile: str) -> BuildProfile:
        """Return a supported build profile or reject configuration drift."""
        if profile == "manual":
            return "manual"
        if profile == "case":
            return "case"
        raise ValueError("audit_profile must be 'manual' or 'case'")

    def read_raw_source(self, uri: str) -> SourceReadResult:
        """Read a supported provenance URI through its owning backend."""
        uri = self._canonicalize_local_raw_uri(uri)
        kind = classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri)
        if kind is None:
            return SourceReadResult(
                uri=uri,
                kind=None,
                status=SourceReadStatus.INVALID_URI,
                error_code="invalid_raw_source_uri",
            )
        if kind is RawSourceKind.EXTERNAL:
            return SourceReadResult(
                uri=uri,
                kind=kind,
                status=SourceReadStatus.UNAVAILABLE,
                error_code="external_source_not_locally_readable",
            )
        content: str | None
        if uri.startswith(self._raw_fs.root_uri.rstrip("/") + "/"):
            key = unquote(uri.removeprefix(self._raw_fs.root_uri.rstrip("/") + "/"))
            content = self._raw_fs.read_text(key)
        elif uri.startswith("viking://resources/"):
            try:
                content = viking_read(uri, propagate_unavailable=True)
            except (HTTPError, OpenVikingError) as exc:
                return SourceReadResult(
                    uri=uri,
                    kind=kind,
                    status=SourceReadStatus.UNAVAILABLE,
                    error_code=type(exc).__name__,
                )
        else:
            content = None
        if content is None:
            return SourceReadResult(
                uri=uri,
                kind=kind,
                status=SourceReadStatus.NOT_FOUND,
                error_code="source_not_found",
            )
        return SourceReadResult(
            uri=uri,
            kind=kind,
            status=SourceReadStatus.OK,
            content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )

    def _canonicalize_local_raw_uri(self, uri: str) -> str:
        """Normalize equivalent local path spellings for raw-source URIs."""
        root_uri = self._raw_fs.root_uri.rstrip("/")
        if not root_uri.startswith("file://") or not uri.startswith("file://"):
            return uri
        parsed = urlsplit(uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            return uri
        try:
            root = Path(self._raw_fs.root).resolve()
            path = Path(unquote(parsed.path)).resolve()
            relative = path.relative_to(root)
        except (OSError, ValueError):
            return uri
        return f"{root_uri}/{relative.as_posix()}"

    def _source_hash(self, uri: str, *, refresh: bool = False) -> str | None:
        """Read and cache one source hash for packet validation.

        Packet registration only needs the membership and hashes of the
        packet's own sources.  It must not rebuild the complete document
        snapshot for every packet; explicit coverage/change checks perform the
        full freshness scan when required.
        """
        canonical = self._canonicalize_local_raw_uri(uri)
        if not refresh and canonical in self._raw_source_hash_cache:
            return self._raw_source_hash_cache[canonical]
        result = self.read_raw_source(canonical)
        value = result.content_hash if result.status is SourceReadStatus.OK else None
        self._raw_source_hash_cache[canonical] = value
        return value

    def _lookup_recorded_source_hash(
        self, uri: str, *, coverage: dict[str, object] | None = None
    ) -> str | None:
        """Return the content hash recorded for *uri* in any source packet.

        Pass a pre-fetched ``coverage`` dict (from ``get_source_ledger()``) to
        avoid re-reading all packets when called in a loop.
        """
        if coverage is None:
            ledger = self.get_source_ledger()
            cov = ledger.get("coverage")
            if not isinstance(cov, dict):
                return None
            coverage = cov
        record = coverage.get(uri)
        if isinstance(record, dict):
            source_hash = record.get("source_hash")
            if isinstance(source_hash, str) and source_hash:
                return source_hash
        return None

    def _current_entity_snapshot_id(self) -> str:
        """Hash current formal files so finalize cannot publish a stale audit."""
        records = [
            f"{record['uri']}\x1f{record['content_hash']}"
            for record in self._formal_entity_snapshot_records()
        ]
        return sha256("\n".join(sorted(records)).encode()).hexdigest()

    def _current_source_snapshot_id(self) -> str:
        """Re-read every cited raw source and hash the publication snapshot."""
        source_hashes: dict[str, str] = {}
        ledger_coverage = self.get_source_ledger().get("coverage")
        if not isinstance(ledger_coverage, dict):
            ledger_coverage = {}
        for record in self._formal_entity_snapshot_records():
            if entity_status(record["content"]) == "deprecated":
                continue
            for uri in extract_source_uris(record["content"]):
                kind = classify_raw_source_uri(
                    uri,
                    raw_root_uri=self._raw_fs.root_uri,
                )
                if kind is None:
                    continue
                result = self.read_raw_source(uri)
                if result.status is SourceReadStatus.OK and result.content_hash is not None:
                    source_hashes[uri] = result.content_hash
                # External (kb:// etc.) and other locally-unreadable
                # sources are excluded from the source snapshot exactly
                # like audit does (audit only hashes what it can read).
                # The server cannot verify external content or its hash;
                # only the reference itself matters, and unresolved
                # references surface as audit gaps, not snapshot drift.
                elif result.status is not SourceReadStatus.OK:
                    logger.warning(
                        "Skipping unresolvable source URI in snapshot: %s (%s)",
                        uri,
                        result.status.value,
                    )
        return sha256(
            "\n".join(
                f"{uri}\x1f{content_hash}" for uri, content_hash in sorted(source_hashes.items())
            ).encode(),
        ).hexdigest()

    def _audit_all_pages(
        self, *, profile: BuildProfile = "manual", limit: int = 500
    ) -> WikiAuditReport:
        """Run one complete, snapshot-consistent audit for a quality gate.

        ``audit_wiki`` is intentionally paginated for capability callers. Finalize
        is different: it must make a decision over every issue, not only the
        first page.  The audit cache keeps subsequent pages cheap while this
        helper makes the completeness requirement explicit and verifies that
        pagination never crosses an entity snapshot boundary.
        """
        # Prime the raw-chapter URI registry (list_chapters only — cheap,
        # no content reads) so raw-link audit checks pass consistently.
        for doc_id in self._library_doc_ids():
            self.list_chapters(doc_id)
        first = self.audit_wiki(profile=profile, offset=0, limit=limit)
        pages = list(first["issues"])
        snapshot_id = str(first.get("snapshot_id", ""))
        source_snapshot_id = str(first.get("source_snapshot_id", ""))
        next_offset = int(first.get("next_offset", -1))
        while next_offset != -1:
            page = self.audit_wiki(profile=profile, offset=next_offset, limit=limit)
            if str(page.get("snapshot_id", "")) != snapshot_id:
                raise ValueError(
                    "Wiki changed while audit pagination was in progress; rerun audit before finalize.",
                )
            if str(page.get("source_snapshot_id", "")) != source_snapshot_id:
                raise ValueError(
                    "Raw sources changed while audit pagination was in progress; rerun audit before finalize.",
                )
            pages.extend(page["issues"])
            next_offset = int(page.get("next_offset", -1))
        complete: WikiAuditReport = first.copy()
        complete["issues"] = pages
        complete["returned_issue_count"] = len(pages)
        complete["next_offset"] = -1
        return complete

    @staticmethod
    def _requires_formal_write_validation(content: str) -> bool:
        """Return whether a write claims to be a formal machine-valid page."""
        frontmatter = parse_frontmatter(content)
        return (
            entity_status(content) in {"confirmed", "deprecated"}
            or str(frontmatter.get("publication_state", "")).strip() == "published"
            or str(frontmatter.get("validation_state", "")).strip() == "machine_validated"
        )

    def _validate_formal_write(
        self,
        *,
        content: str,
        concept: str,
        class_name: str,
        object_name: str,
        skip_materialization: bool = False,
    ) -> None:
        """Enforce materialization policy and formal-page invariants."""
        if skip_materialization:
            return
        if self._requires_formal_write_validation(content):
            require_valid_entity(
                content=content,
                concept=concept,
                class_name=class_name,
                object_name=object_name,
                hooks=_FORMAL_WRITE_HOOKS,
            )

    def _reject_malformed_wiki_refs(self, content: str) -> None:
        """Reject pseudo-links before any draft or formal page is stored.

        ``{root_uri}/.../open_gap: ...`` is a common LLM failure mode: gap
        prose is accidentally wrapped in a URI-shaped Markdown link.  It is
        neither a resolvable entity nor an honest plain-text gap.  Rejecting
        it at the write boundary keeps the corpus convergent; the worker must
        either use a real URI or write ``open_gap`` as ordinary prose.
        """
        malformed = extract_malformed_wiki_uris(content)
        if malformed:
            shown = ", ".join(malformed[:5])
            suffix = "..." if len(malformed) > 5 else ""
            raise ValueError(
                "Entity write contains malformed wiki URI(s): "
                f"{shown}{suffix}. Use a real {self.store.root_uri}/<Concept>/<Class>/<Object> "
                "URI, or keep an unresolved gap as plain text.",
            )

    def _reject_wrong_raw_refs(self, content: str) -> None:
        """Reject chapter links rooted at the writable Wiki namespace.

        Raw chapter citations must remain addressable in the read-only raw
        namespace. A Wiki namespace URI containing ``/chapters/`` is always a
        stale or fabricated source link, even when its tail looks plausible.
        """
        if not self.store.root_uri.startswith("viking://"):
            return
        raw_prefix = self._raw_fs.root_uri + "/"
        tokens = re.findall(r"viking://resources/[^\s\"'<>\[\]]+", content)
        wrong = [
            token for token in tokens if "/chapters/" in token and not token.startswith(raw_prefix)
        ]
        if wrong:
            shown = ", ".join(sorted(set(wrong))[:3])
            raise ValueError(
                f"Entity write contains non-raw chapter URI(s): {shown}. Use the configured raw root {self._raw_fs.root_uri}/... .",
            )

    def _reject_phantom_body_refs(
        self,
        content: str,
        *,
        extra_known: set[str] | None = None,
    ) -> None:
        """Reject promote-from-draft content with unresolvable wiki hash URIs.

        A ``{root_uri}/<Concept>/<hash>`` reference is valid only if it
        resolves to a registered entity or belongs to the batch being
        promoted (``extra_known``, whose deterministic hashes are known).
        Any other 24-hex hash is a phantom — hand-written at extraction
        time — and must be fixed (via ``entity_uri``) or downgraded to
        plain text before promotion.  This closes the draft→formal gate
        so phantoms never reach the formal library.

        Raises:
            ValueError: listing the unresolvable hash URIs found.
        """
        known: set[str] = set(extra_known or ())
        phantoms: list[str] = []
        for uri in sorted(extract_wiki_uris(content)):
            stripped = uri.split("#")[0]
            if stripped in known:
                continue
            # Resolve and confirm the target actually exists on disk.
            if (
                self.store.lookup_by_uri(stripped) is not None
                and self.store.read_entity_by_uri(stripped) is not None
            ):
                known.add(stripped)
                continue
            phantoms.append(stripped)
        if phantoms:
            shown = ", ".join(phantoms[:10])
            suffix = "..." if len(phantoms) > 10 else ""
            raise ValueError(
                "Promotion blocked: content references unresolvable URIs "
                f"({len(phantoms)}): {shown}{suffix}. "
                f"Write readable URIs {self.store.root_uri}/<Concept>/<Class>/<Object> "
                "matching the target entity's identity, or downgrade the link "
                "to plain text if the target entity does not exist.",
            )

    def _reject_nonexistent_raw_sources(self, content: str) -> None:
        """Warn about source URIs that don't resolve to actual raw files.

        Downgraded from hard rejection to warning — a truncated or stale
        source URI in frontmatter ``sources`` is a reference-quality issue,
        not a data-integrity blocker.  The 4 core-path content checks
        (``confirmation_requirements``) are the real publish gate.
        """
        source_uris = extract_source_uris(content)
        nonexistent: list[str] = []
        for uri in source_uris:
            kind = classify_raw_source_uri(uri, raw_root_uri=self._raw_fs.root_uri)
            if kind is None or kind is RawSourceKind.EXTERNAL:
                continue
            result = self.read_raw_source(uri)
            if result.status is SourceReadStatus.NOT_FOUND:
                nonexistent.append(uri)
        if nonexistent:
            shown = ", ".join(nonexistent[:5])
            suffix = "..." if len(nonexistent) > 5 else ""
            logger.warning(
                "Entity write has unresolvable source URI(s): %s%s. "
                "Non-blocking — entity content is unaffected.",
                shown,
                suffix,
            )

    def _root_doc_name(self) -> str:
        """Return the library root's own name as the document id.

        Used when ``chapters`` sits directly under the library root (doc root
        == library root): the root name is the only stable identifier in both
        local (``file:///.../<name>``) and viking (``viking://.../<name>``)
        backends.
        """
        return self._raw_fs.root_uri.rstrip("/").rsplit("/", 1)[-1]

    def _build_doc_prefix_index(self) -> None:
        """Walk the raw root once and index every doc_id → path prefix.

        Standard ``.../<doc_id>/chapters/`` layout is preferred. For
        non-standard layouts, every directory segment in each ``.md`` path
        becomes a candidate doc_id mapped to the longest prefix ending with
        that segment. Result is cached in ``_doc_prefix_index`` and warmed
        into ``_raw_doc_prefixes`` so ``list_chapters`` skips blind DFS.
        """
        if self._doc_prefix_index is not None:
            return
        index: dict[str, str] = {}
        for key in self._raw_fs.list_dir("", recursive=True):
            if not key.endswith(".md"):
                continue
            parts = [p for p in key.split("/") if p]
            if len(parts) < 2:
                continue
            # ``chapters`` is a document-internal marker, not a root.  Match it
            # at ANY position, including a leading ``chapters/`` (the doc root
            # is the library root itself in that case).  Everything before it
            # names the document; non-``chapters`` layouts are documents of
            # one directory deep, located by drilling down with ``browse``.
            chapters_idx = -1
            for i, part in enumerate(parts):
                if part == "chapters":
                    chapters_idx = i
                    break
            if chapters_idx > 0:
                prefix = "/".join(parts[:chapters_idx])
                index[parts[0]] = prefix  # standard layout: doc root is the top segment
                continue
            if chapters_idx == 0 and len(parts) > 1:
                # Doc root == library root: the library root's own name is
                # the document id (``chapters`` is just its hierarchy).
                self._root_is_doc = True
                index[self._root_doc_name()] = ""
                continue
            # Non-standard layout (no ``chapters`` marker): the library is a
            # set of top-level documents.  ``browse`` drills down from the
            # root; only the first directory segment identifies a document —
            # intermediate segments are hierarchy, not document names.
            index[parts[0]] = parts[0]
        self._doc_prefix_index = index
        for doc_id, prefix in index.items():
            self._raw_doc_prefixes.setdefault(doc_id, prefix)

    def _resolve_doc_prefix(self, doc_id: str, max_depth: int = 8) -> str:
        """Find the shallowest directory whose name matches ``doc_id``.

        Iterative-deepening BFS over directories only, sibling expansion
        parallelised. Returns the relative path-key of the first match, or
        ``""`` if not found within ``max_depth``. Cheaper than
        :meth:`_build_doc_prefix_index` for single-doc lookups: skips the
        full-namespace walk when the caller only needs one doc.
        """
        expected = doc_id.casefold()
        backend_subdirs: Callable[[str], list[str]] | None = getattr(
            self._raw_fs, "list_subdirs", None
        )

        def list_subdirs(k: str) -> list[str]:
            if backend_subdirs is not None:
                return [str(e) for e in backend_subdirs(k)]
            return [
                e for e in self._raw_fs.list_entries(k, recursive=False) if self._raw_fs.is_dir(e)
            ]

        current_level: list[str] = [""]
        seen: set[str] = set()
        with ThreadPoolExecutor(max_workers=_io_worker_limit()) as pool:
            for _ in range(max_depth + 1):
                if not current_level:
                    break
                current_level = [d for d in current_level if d not in seen]
                for d in current_level:
                    seen.add(d)
                if not current_level:
                    break
                results = list(pool.map(list_subdirs, current_level))
                next_level: list[str] = []
                for child_dirs in results:
                    for entry in child_dirs:
                        if entry in seen:
                            continue
                        name = entry.rsplit("/", 1)[-1]
                        if name.casefold() == expected:
                            return entry
                        next_level.append(entry)
                current_level = next_level
        return ""

    def _library_doc_ids(self, *, refresh: bool = False) -> tuple[str, ...]:
        """Return all parsed document IDs available in the configured raw root.

        When ``WIKI_SOURCE_DOCS`` is set, the conductor drives document
        discovery via :meth:`browse` and passes full doc-id paths to
        ``list_chapters``, ``source_coverage_status``, etc.  No recursive
        scan is performed — returning an empty tuple here lets callers
        fall back to the conductor-supplied ``doc_ids``.
        """
        if self._source_doc_allowlist:
            return ()
        if self._library_doc_ids_cache is not None and not refresh:
            return self._library_doc_ids_cache
        # Batch path: one full walk populates both the doc list and the
        # prefix cache used by list_chapters. Single-doc callers don't go
        # through here — they use _resolve_doc_prefix instead.
        if refresh:
            self._doc_prefix_index = None
            self._raw_doc_prefixes.clear()
            self._chapters_cache.clear()
            self._library_doc_ids_cache = None
        self._build_doc_prefix_index()
        result = tuple(sorted(self._doc_prefix_index or {}))
        self._library_doc_ids_cache = result
        return result

    def get_schema(self, concept: str) -> dict[str, object]:
        """Return the authoritative schema and version for one Concept."""
        return {
            "version": get_schema_version(),
            "concept": concept,
            "schema": get_concept_schema(concept),
        }

    def auto_repair(self, entity_uris: list[str] | None = None) -> dict[str, object]:
        """Repair only the current build's changed entities by default."""
        if entity_uris is None:
            checkpoint = self.store.read_json("index/build_checkpoint.json")
            build_started_at = (
                str(checkpoint.get("started_at", "")) if isinstance(checkpoint, dict) else ""
            )
            changed = self.build_change_report(
                persist=False,
                include_op_flow=False,
            ).get("changed_entities", [])
            scoped: list[str] = []
            for row in changed if isinstance(changed, list) else []:
                if not isinstance(row, dict):
                    continue
                event_time = str(row.get("timestamp", ""))
                if build_started_at and (not event_time or event_time < build_started_at):
                    continue
                uri = str(row.get("uri", "")).strip()
                if uri:
                    scoped.append(uri)
            if not scoped and build_started_at and not self._log:
                try:
                    started = datetime.fromisoformat(build_started_at)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=UTC)
                    started_ns = int(started.timestamp() * 1_000_000_000)
                except ValueError:
                    started_ns = 0
                if started_ns:
                    for concept, class_name, object_name, uri in self.store.list_entities():
                        path = self.store.entity_path(concept, class_name or None, object_name)
                        modified_ns = self.store._fs.mtime_ns(self.store._key_of(path))
                        if modified_ns is not None and modified_ns >= started_ns:
                            scoped.append(uri)
            entity_uris = list(dict.fromkeys(scoped))
        report = batch_auto_repair(self, entity_uris=entity_uris)
        if report.entities_modified:
            self._invalidate_audit_cache()
        return report.to_dict()

    def input_snapshot_hash(self, doc_ids: tuple[str, ...] = (), *, refresh: bool = True) -> str:
        """Hash every exact source chapter in the selected raw documents."""
        snapshots = self.source_snapshot(doc_ids, refresh=refresh)
        records = [f"{uri}\x1f{content_hash}" for uri, content_hash in snapshots.items()]
        return sha256("\n".join(sorted(records)).encode()).hexdigest()
