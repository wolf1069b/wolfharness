"""Chapter reading, browsing, planning, and BOM enrichment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import time

from openviking_sdk.errors import OpenVikingError

from wolfharness.capabilities.wiki.io.chapter_scoring import (
    build_fingerprint,
    score_chapter_record,
    should_auto_register_no_entity,
    should_auto_register_no_entity_from_toc,
)
from wolfharness.capabilities.wiki.io.text_compact import compact_chapter
from wolfharness.capabilities.wiki.io.text_parsers import (
    _CHAPTER_PREFIX_RE,
    _dir_to_clean_title,
)
from wolfharness.capabilities.wiki.quality import (
    register_raw_chapter_uris,
)
from wolfharness.capabilities.wiki.section_constants import (
    SECTION_MECHANISM,
)


logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki._helpers import (
    _BOM_ENRICH_PLACEHOLDER_MARKERS,
    _chapter_idempotency_key,
    _entity_batch_limit,
    _io_worker_limit,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from wolfharness.capabilities.wiki.storage import FSBackend


_PAGE_RANGE_RE = re.compile(r"#p(\d+)-(\d+)")


def _strip_source_prefix(doc_id: str) -> str:
    """Strip the ``viking:`` / ``fixmaster:`` source prefix from ``doc_id``.

    The conductor prompt mandates ``doc_id`` format ``{source}:{identifier}``
    for checkpoint identity, but filesystem path resolution needs the bare
    document directory name.  This strips a single leading ``viking:`` or
    ``fixmaster:`` prefix; all other values pass through unchanged.
    """
    for prefix in ("viking:", "fixmaster:"):
        if doc_id.startswith(prefix):
            return doc_id[len(prefix) :]
    return doc_id


def _parse_page_range(uri: str) -> tuple[int, int] | None:
    """Extract (start, end) page numbers from a kb:// URI fragment."""
    m = _PAGE_RANGE_RE.search(uri)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _filter_leaf_manifest_entries(
    entries: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Remove manifest entries that fully contain another entry.

    When the conductor builds a ``chapter_manifest`` from a TOC tree, it may
    accidentally include both a parent section and its children.  This filters
    out parent entries so only leaf (most granular) segments remain, preventing
    duplicate extraction of the same pages.

    Containment is detected via page-range overlap (``#p{start}-{end}``) when
    both URIs carry it; otherwise URI string prefix containment is used.
    """
    if len(entries) <= 1:
        return entries
    parsed: list[tuple[int, tuple[int, int] | None, str]] = [
        (i, _parse_page_range(str(e.get("uri", ""))), str(e.get("uri", "")))
        for i, e in enumerate(entries)
    ]
    to_remove: set[int] = set()
    for i, (idx_a, range_a, uri_a) in enumerate(parsed):
        if idx_a in to_remove:
            continue
        for j, (idx_b, range_b, uri_b) in enumerate(parsed):
            if i == j or idx_b in to_remove:
                continue
            if range_a is not None and range_b is not None:
                # Page-range containment: a contains b → remove a
                if range_a[0] <= range_b[0] and range_a[1] >= range_b[1] and range_a != range_b:
                    to_remove.add(idx_a)
                    break
            elif uri_a and uri_b:
                # URI prefix containment fallback
                if uri_b.startswith(uri_a.rstrip("/")) and uri_a.rstrip("/") != uri_b.rstrip("/"):
                    to_remove.add(idx_a)
                    break
    if not to_remove:
        return entries
    removed_titles = [
        entries[idx].get("title", entries[idx].get("uri", "?")) for idx in sorted(to_remove)
    ]
    logger.warning(
        "Filtered %d parent manifest entries (contained children): %s",
        len(to_remove),
        removed_titles,
    )
    return [e for idx, e in enumerate(entries) if idx not in to_remove]


class ChapterMixin:
    """Chapter reading, browsing, planning, and BOM enrichment."""

    def load_manifest(self) -> list[dict]:
        """List parsed documents available on the raw backend.

        Prefers the local build manifests (``documents.json`` /
        ``run_info.json``) when present; falls back to discovering
        documents from the raw backend keys.
        """
        docs: list[dict] = []
        if self._library_root is not None and self._library_root.is_dir():
            path = self._library_root / "documents.json"
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                manifest = data.get("documents", data) if isinstance(data, dict) else data
                if manifest:
                    return manifest
                logger.warning(
                    "Document manifest %s contains no records; auto-discovering manuals instead",
                    path,
                )
            for child in sorted(self._library_root.iterdir()):
                if not child.is_dir():
                    continue
                run_info = child / "run_info.json"
                if not run_info.is_file():
                    continue
                info = json.loads(run_info.read_text(encoding="utf-8"))
                pdf_path = info.get("pdf_path", "")
                title = Path(pdf_path).stem if pdf_path else child.name
                model = self._extract_model_from_doc(child, pdf_path)
                docs.append({"output_id": child.name, "title": title, "model": model})
            if docs:
                logger.info("Auto-discovered %d docs from %s", len(docs), self._library_root_uri)
                return docs

        # Fallback: discover documents from the raw backend keys.
        for doc_id in self._library_doc_ids():
            model = self._extract_model_from_doc(Path(doc_id), "")
            docs.append({"output_id": doc_id, "title": doc_id, "model": model})
        logger.info("Auto-discovered %d docs from raw backend", len(docs))
        return docs

    def list_chapters(self, doc_id: str, *, refresh: bool = False) -> list[dict]:
        """List all leaf chapter directories for a parsed document.

        Reads the chapter tree from the raw backend (``{doc}/chapters``).
        Each entry has ``rootsection``, ``section``, ``subdir``, ``title``
        and ``md_path``.  ``subdir`` is the real path under ``chapters/``
        and is used directly in raw URIs returned by ``make_source_uri``.
        """
        if refresh:
            self._chapters_cache.pop(doc_id, None)
            self._raw_doc_prefixes.pop(doc_id, None)
        cached = self._chapters_cache.get(doc_id)
        if cached is not None and not refresh:
            return [dict(chapter) for chapter in cached]
        doc_name = _strip_source_prefix(doc_id)
        if doc_id not in self._raw_doc_prefixes and (self._doc_prefix_index is None or refresh):
            # If the full index is already built (batch path via
            # _library_doc_ids), a miss here means the doc genuinely isn't
            # in the namespace — skip the targeted search. Otherwise do a
            # cheap iterative-deepening lookup instead of a full walk.
            resolved = self._resolve_doc_prefix(doc_name)
            if resolved:
                self._raw_doc_prefixes[doc_id] = resolved
        if doc_id in self._raw_doc_prefixes:
            document_prefix = self._raw_doc_prefixes[doc_id] or ""
        else:
            document_prefix = doc_name
        keys = self._raw_fs.list_dir(document_prefix, recursive=True)
        if not keys and document_prefix == doc_name:
            document_prefix, keys = self._discover_nonstandard_doc_files(doc_name)
            if document_prefix:
                self._raw_doc_prefixes[doc_id] = document_prefix
        legacy_prefix = f"{document_prefix}/chapters" if document_prefix else "chapters"
        dirs: set[str] = set()
        for key in keys:
            if key.startswith(legacy_prefix + "/") and key.endswith("chapter.md"):
                dirs.add(key[: -len("chapter.md")].rstrip("/"))
        out: list[dict[str, str]] = []
        if dirs:
            leaf_dirs = sorted(
                d
                for d in dirs
                if not any(d != other and other.startswith(d + "/") for other in dirs)
            )
            for directory in leaf_dirs:
                rel = directory.removeprefix(legacy_prefix + "/").rstrip("/")
                if not rel:
                    continue
                parts = [part for part in rel.split("/") if part]
                out.append(
                    {
                        "rootsection": _CHAPTER_PREFIX_RE.sub("", parts[0]),
                        "section": _CHAPTER_PREFIX_RE.sub("", parts[-1]),
                        "subdir": rel,
                        "title": _dir_to_clean_title(parts[-1]),
                        "md_path": f"{legacy_prefix}/{rel}/chapter.md",
                    },
                )
        else:
            ignored_names = {"full.md", "toc.md", "readme.md"}
            doc_prefix = f"{document_prefix}/" if document_prefix else ""
            for key in sorted(keys):
                if not key.startswith(doc_prefix) or not key.endswith(".md"):
                    continue
                if key.rsplit("/", 1)[-1].lower() in ignored_names:
                    continue
                rel = key.removeprefix(doc_prefix)
                parts = [part for part in rel.split("/") if part]
                if parts and parts[0].casefold() == doc_name.casefold():
                    logical_parts = parts[1:]
                else:
                    logical_parts = parts
                if not logical_parts:
                    continue
                section_name = Path(logical_parts[-1]).stem
                root_name = logical_parts[0] if len(logical_parts) > 1 else section_name
                out.append(
                    {
                        "rootsection": _CHAPTER_PREFIX_RE.sub("", root_name),
                        "section": _CHAPTER_PREFIX_RE.sub("", section_name),
                        "subdir": rel,
                        "title": _dir_to_clean_title(section_name),
                        "md_path": key,
                    },
                )
        for chapter in out:
            subdir = str(chapter["subdir"])
            self._chapter_path_aliases[(doc_id, subdir)] = str(chapter["md_path"])
        register_raw_chapter_uris(
            [f"{self._raw_fs.root_uri}/{chapter['md_path']}" for chapter in out],
        )
        self._chapters_cache[doc_id] = [dict(chapter) for chapter in out]
        return [dict(chapter) for chapter in out]

    # ── Filesystem walking ────────────────────────────────────────────────

    def _chapter_read_candidates(self, doc_id: str, chapter_subdir: str) -> list[str]:
        """Return root-relative keys for both logical and catalog paths.

        Chapter planners may return either a path relative to the logical
        document or the authoritative catalog-relative ``md_path``.  The
        latter must not be prefixed with ``doc_id`` a second time.  Keeping
        this normalization at the read boundary also makes retries safe when
        the planner and worker run in separate service instances (where the
        in-memory chapter alias is unavailable).
        """
        raw_root = self._raw_fs.root_uri.rstrip("/") + "/"
        requested = chapter_subdir.strip("/")
        if requested.startswith(raw_root):
            requested = requested.removeprefix(raw_root).strip("/")
        path_parts = requested.split("/")
        if (
            not requested
            or any(part in {"", ".", ".."} for part in path_parts)
            or "\\" in requested
        ):
            raise ValueError(f"Invalid chapter path: {chapter_subdir!r}")

        candidates: list[str] = []
        alias = self._chapter_path_aliases.get((doc_id, chapter_subdir), "")
        if alias:
            candidates.append(alias.strip("/"))
        candidates.append(requested)

        if doc_id in self._raw_doc_prefixes:
            prefix = self._raw_doc_prefixes[doc_id] or ""
        else:
            prefix = self._chapter_window_root(doc_id)[0]
        prefix = prefix.strip("/")
        logical_doc = doc_id.strip("/")

        if requested.endswith(".md"):
            candidates.extend(
                f"{root}/{requested}"
                for root in (prefix, logical_doc)
                if root and not (requested == root or requested.startswith(root + "/"))
            )
        else:
            for root in (prefix, logical_doc):
                if root and not (requested == root or requested.startswith(root + "/")):
                    candidates.append(f"{root}/{requested}/chapter.md")
                    candidates.append(f"{root}/chapters/{requested}/chapter.md")
            candidates.append(f"chapters/{requested}/chapter.md")

        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def read_chapter(self, doc_id: str, chapter_subdir: str) -> str:
        """Read the full content of a chapter markdown file.

        ``chapter_subdir`` is the real path under ``{doc_id}/chapters/``.
        Content is read through the raw backend.
        """
        doc_parts = doc_id.split("/")
        if (
            not doc_id
            or doc_id.startswith("/")
            or "\\" in doc_id
            or any(part in {"", ".", ".."} for part in doc_parts)
        ):
            raise ValueError(f"Invalid document id: {doc_id!r}")
        extraction_started = time.perf_counter()
        content: str | None = None
        for candidate in self._chapter_read_candidates(doc_id, chapter_subdir):
            content = self._raw_fs.read_text(candidate)
            if content is not None:
                break
        if content is None:
            raise FileNotFoundError(f"Chapter not found: {doc_id}/{chapter_subdir}")
        # Compact at the read boundary (not in storage): drops OCR HTML
        # residue/template noise before the text reaches the model context.
        # Raw files and their hashes are untouched, so source references stay
        # valid.  ~58% fewer input tokens, zero extractable-fact loss.
        compacted = compact_chapter(content)
        self._record_phase_timing("extraction", extraction_started)
        return compacted

    def read_chapters_batch(self, doc_id: str, paths: list[str]) -> list[dict]:
        """Read multiple chapters in one call.

        ``paths`` is a list of chapter subdir paths (e.g.
        ``["01_故障诊断/01_发动机冒黑烟", "01_故障诊断/02_起动困难"]``).
        Returns ``[{"path": ..., "uri": ..., "content": ...}, ...]``.
        Failed reads are skipped silently — the caller should compare
        the result length against the request length to detect misses.
        """
        if not paths:
            return []

        # Viking imports may expose a logical document below a catalog path
        # and preserve one additional document-named directory below it. The
        # window API returns logical paths, while a worker may submit paths
        # relative to either level. Resolve the immediate directory aliases
        # once per batch so a miss does not turn into a worker-side browse/
        # retry loop.
        read_roots: list[str] = []
        for root in (doc_id.strip("/"), self._raw_doc_prefixes.get(doc_id, "").strip("/")):
            if root and root not in read_roots:
                read_roots.append(root)
        for root in tuple(read_roots):
            for entry, is_dir in self._raw_fs.list_entries_with_meta(root):
                if is_dir and entry not in read_roots:
                    read_roots.append(entry)

        def read_one(path: str) -> tuple[str, str, str] | None:
            try:
                key = self._chapter_path_aliases.get((doc_id, path), "")
                candidates = [key] if key else []
                normalized_path = path.strip("/")
                raw_prefix = self._raw_fs.root_uri.rstrip("/") + "/"
                if normalized_path.startswith(raw_prefix):
                    normalized_path = normalized_path.removeprefix(raw_prefix)
                # ``browse_chapters`` returns the resolved catalog-relative
                # md_path, while older workers may send a path relative to the
                # logical document. Try the resolved path directly before
                # applying document-relative roots; otherwise the document
                # prefix is duplicated and the batch looks falsely empty.
                candidates.append(normalized_path)
                for root in read_roots:
                    candidates.append(f"{root}/{normalized_path}")
                    if not normalized_path.endswith(".md"):
                        candidates.append(f"{root}/chapters/{normalized_path}/chapter.md")
                for candidate in dict.fromkeys(candidates):
                    content = self._raw_fs.read_text(candidate)
                    if content is not None:
                        return path, compact_chapter(content), candidate
            except (FileNotFoundError, OSError, OpenVikingError):
                logger.debug("read_chapters_batch skip %s/%s", doc_id, path)
                return None
            return None

        max_workers = min(_io_worker_limit(), max(1, len(paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            read_results = list(pool.map(read_one, paths))
        records: list[dict[str, object]] = []
        for item in read_results:
            if item is None:
                continue
            path, content, key = item
            self._chapter_path_aliases.setdefault((doc_id, path), key)
            records.append(
                {
                    "path": path,
                    "uri": self.make_source_uri(doc_id, path),
                    "content": content,
                },
            )
        return records

    def browse_chapters(self, doc_id: str, path: str = "") -> dict:
        """Browse one level of a document's chapter tree, per-level drill-down.

        Resolves the document directory via :meth:`_resolve_doc_prefix` (which
        handles catalog-nested layouts like ``menu/{model}/{doc}/{doc}/``)
        and lists a single directory level — no deep-tree traversal, so the
        output is just the direct children.

        ``browse_chapters(doc_id)`` → top-level rootsections;
        ``browse_chapters(doc_id, path)`` → subdirectories at *path*.

        For browsing an arbitrary URI/library (not bound to a specific
        document), use :meth:`browse`.

        Returns ``{type: "branch", path, children: [{title, path, has_children, uri?}]}``,
        ``{type: "leaf", title, uri}``, or ``{type: "not_found", error}``.
        """
        # A full raw root URI in path bypasses doc_id resolution entirely.
        prefix = self._raw_fs.root_uri.rstrip("/") + "/"
        if path.startswith(prefix):
            return self._browse_dir(path[len(prefix) :].strip("/"), self._raw_fs)
        # Strip source prefix (viking:/fixmaster:) and resolve the actual
        # document directory in the raw library.
        doc_name = _strip_source_prefix(doc_id)
        if doc_id in self._raw_doc_prefixes:
            document_prefix = self._raw_doc_prefixes[doc_id] or ""
        else:
            document_prefix = self._resolve_doc_prefix(doc_name)
            if document_prefix:
                self._raw_doc_prefixes[doc_id] = document_prefix
            else:
                document_prefix = doc_name
        rel = (
            f"{document_prefix}/{path.strip('/')}".strip("/")
            if path.strip("/")
            else document_prefix
        )
        return self._browse_dir(rel, self._raw_fs)

    def browse(self, uri: str = "") -> dict:
        """Browse one level under an arbitrary in-root URI/path.

        Generic drill-down over any library, document, case, wiki, BOM or
        other raw-root subtree — no assumption about a ``chapters`` layout.
        ``uri`` may be a full ``viking://resources/<ns>/...`` URI (or
        ``file://`` raw URI) within the raw manual library **or** the global
        BOM library (``_bom_fs``, e.g. ``viking://resources/730/bom/...``),
        or a path relative to the relevant root.  Empty lists the raw manual
        root itself.  Each call returns only the direct children of that
        location (lightweight non-recursive).

        Returns ``{type: "branch", path, children: [{title, path, has_children, uri?}]}``,
        ``{type: "leaf", title, uri}``, or ``{type: "not_found", error}``.
        """
        fs = self._raw_fs
        if self._bom_fs is not None and (
            uri == self._bom_fs.root_uri or uri.startswith(self._bom_fs.root_uri.rstrip("/") + "/")
        ):
            fs = self._bom_fs
        prefix = fs.root_uri.rstrip("/") + "/"
        if uri == fs.root_uri or uri == fs.root_uri.rstrip("/") + "/":
            rel = ""
        elif uri.startswith(prefix):
            rel = uri[len(prefix) :].strip("/")
        else:
            rel = uri.strip("/")
        return self._browse_dir(rel, fs)

    def _chapter_window_root(self, doc_id: str) -> tuple[str, str]:
        """Resolve a logical document to its root without building a leaf index."""
        if not doc_id.strip():
            raise ValueError("doc_id must not be empty")
        if doc_id in self._raw_doc_prefixes:
            # Empty prefix is legal: the library root itself is the document.
            document_prefix = self._raw_doc_prefixes[doc_id] or ""
        else:
            if self._doc_prefix_index is None:
                self._build_doc_prefix_index()
            if doc_id in self._raw_doc_prefixes:
                document_prefix = self._raw_doc_prefixes[doc_id] or ""
            else:
                document_prefix = (
                    doc_id if self._raw_fs.is_dir(doc_id) else self._resolve_doc_prefix(doc_id)
                )
            if not document_prefix and doc_id not in self._raw_doc_prefixes:
                raise FileNotFoundError(f"Document directory does not exist: {doc_id}")
        self._raw_doc_prefixes[doc_id] = document_prefix
        chapters_root = f"{document_prefix}/chapters" if document_prefix else "chapters"
        root = chapters_root if self._raw_fs.is_dir(chapters_root) else document_prefix
        return document_prefix, root

    def _browse_dir(self, rel: str, fs: FSBackend) -> dict:
        """Shared layered-browse helper: list one level under a root-relative path.

        ``fs`` is the resolved backend (raw manual library or global BOM
        library); ``rel`` is a path relative to ``fs.root_uri``.
        """
        pairs = fs.list_entries_with_meta(rel)
        if not pairs:
            if fs.exists(rel):
                return {
                    "type": "leaf",
                    "title": _dir_to_clean_title(rel.rsplit("/", 1)[-1]),
                    "uri": f"{fs.root_uri}/{rel}" if rel else fs.root_uri,
                }
            return {
                "type": "not_found",
                "error": f"Path not found: {rel or fs.root_uri}",
            }
        children: list[dict] = []
        for entry, is_dir in sorted(pairs):
            name = entry.rsplit("/", 1)[-1]
            if not is_dir and not name.endswith(".md"):
                continue
            node: dict[str, object] = {
                "title": _dir_to_clean_title(name),
                "path": entry,
                "has_children": is_dir,
            }
            if not is_dir:
                node["uri"] = f"{fs.root_uri}/{entry}"
            children.append(node)
        return {"type": "branch", "path": rel or fs.root_uri, "children": children}

    # ── Source URI (moved from old builder) ─────────────────────────────────

    def make_source_uri(self, doc_id: str, subdir: str) -> str:
        """Build a real-path raw chapter URI.

        Returns ``{root_uri}/{doc_id}/chapters/{subdir}/chapter.md`` where
        ``root_uri`` is the active raw backend root (``file://…`` locally,
        ``viking://resources/<raw_namespace>`` in viking mode).  ``subdir``
        is the real chapter path under ``{doc_id}/chapters/``.
        """
        raw_root = self._raw_fs.root_uri.rstrip("/") + "/"
        requested = subdir.strip("/")
        if requested.startswith(raw_root):
            return f"{raw_root}{requested.removeprefix(raw_root)}"
        alias = self._chapter_path_aliases.get((doc_id, subdir))
        if alias is not None:
            return f"{self._raw_fs.root_uri}/{alias}"
        if subdir.endswith(".md"):
            prefix = self._raw_doc_prefixes.get(doc_id, "").strip("/")
            logical_doc = doc_id.strip("/")
            if requested == prefix or (prefix and requested.startswith(prefix + "/")):
                return f"{self._raw_fs.root_uri}/{requested}"
            if requested == logical_doc or requested.startswith(logical_doc + "/"):
                return f"{self._raw_fs.root_uri}/{requested}"
            for root in (prefix, logical_doc):
                if root:
                    candidate = f"{root}/{requested}"
                    if self._raw_fs.exists(candidate):
                        return f"{self._raw_fs.root_uri}/{candidate}"
            return f"{self._raw_fs.root_uri}/{doc_id}/{subdir}"
        prefix = self._raw_doc_prefixes.get(doc_id, "")
        if doc_id in self._raw_doc_prefixes and not prefix.strip():
            return f"{self._raw_fs.root_uri}/chapters/{subdir}/chapter.md"
        return f"{self._raw_fs.root_uri}/{doc_id}/chapters/{subdir}/chapter.md"

    def inspect_wiki_state(self) -> dict[str, int | str]:
        """Return the deterministic build mode without mutating build state.

        Source packets are stable, build-scoped records and may legitimately
        exist before the first entity is materialized.  A state inspection
        must therefore never treat an empty entity library as permission to
        delete them.  ``record_source_packet`` already replaces packets from
        older build ids in place after validating the current source snapshot.
        """
        entity_count = len(self.store.list_entities())
        if entity_count == 0:
            formal_paths: set[Path] = set()
            for relative_root in self.store.CONCEPT_DIRS.values():
                keys = self.store.list_dir(relative_root, recursive=True)
                for key in keys:
                    rel_parts = Path(key).relative_to(Path(relative_root)).parts
                    if "tmp" in rel_parts or "profile" in rel_parts:
                        continue
                    formal_paths.add(Path(key))
            entity_count = len(formal_paths)
        mode = "empty" if entity_count == 0 else "incremental"
        draft_count = 0
        profile_draft_count = 0
        return {
            "mode": mode,
            "entity_count": entity_count,
            # Retained as a compatibility field for callers that still log
            # it.  State inspection is read-only, so the value is always 0.
            "purged_source_packets": 0,
            "draft_count": draft_count,
            "profile_draft_count": profile_draft_count,
            "storage_backend": "viking" if self.store.root_uri.startswith("viking://") else "local",
            "wiki_root": self.store.root_uri,
            "library_root": self._raw_fs.root_uri,
            "bom_root": self._bom_fs.root_uri if self._bom_fs is not None else "",
            "case_root": str(self._case_root) if self._case_root is not None else "",
            "faultannotated_root": str(self._faultannotated_root)
            if self._faultannotated_root is not None
            else "",
        }

    def inspect_build_checkpoint(
        self,
        *,
        doc_id: str = "",
        build_id: str = "",
    ) -> dict[str, object]:
        """Read the last durable build checkpoint, if one exists.

        When ``doc_id``/``build_id`` are provided they act as an identity
        filter: a checkpoint belonging to a *different* build is reported
        as nonexistent for the current build, so a new build on the same
        namespace does not falsely resume another build's stage.
        """
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        if checkpoint is None:
            return {"exists": False, "stage": ""}
        # Code-level source validation: detect source changes without relying
        # on LLM-passed doc_id/build_id.  The source_fingerprint combines
        # library_root + source_doc_allowlist + wiki_root — any of these
        # changing means the checkpoint belongs to a different build origin.
        stored_fp = str(checkpoint.get("source_fingerprint", ""))
        if stored_fp:
            current_fp = sha256(
                f"{self._raw_fs.root_uri}\x1f{','.join(getattr(self, '_source_doc_allowlist', ()))}\x1f{self.store.root_uri}".encode(),
            ).hexdigest()[:16]
            if stored_fp != current_fp:
                return {
                    "exists": False,
                    "stage": "",
                    "source_changed": True,
                    "stored_source_fingerprint": stored_fp,
                }
        if doc_id and str(checkpoint.get("doc_id", "")) != doc_id:
            return {
                "exists": False,
                "stage": "",
                "stored_doc_id": str(checkpoint.get("doc_id", "")),
            }
        if build_id and str(checkpoint.get("build_id", "")) != build_id:
            return {
                "exists": False,
                "stage": "",
                "stored_build_id": str(checkpoint.get("build_id", "")),
            }
        return {"exists": True, **checkpoint}

    def plan_chapter_work(
        self,
        doc_id: str,
        *,
        build_id: str,
        limit: int = 6,
        active_packet_ids: list[str] | None = None,
        audit_profile: str = "manual",
        refresh: bool = False,
        preview: bool = False,
        preview_offset: int = 0,
        preview_limit: int = 50,
        chapter_manifest: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        """Return the next receipt-missing chapters for dynamic dispatch.

        The service owns inventory, stable packet identities and completion
        detection.  The conductor owns scheduling: it chooses ``limit`` from
        the workers currently available and may change that value every call.
        No batch index or cursor participates in progress, so a restart or a
        different worker count cannot skip chapters.

        When ``chapter_manifest`` is provided, the planner skips local
        filesystem scanning (``list_chapters``) and uses the manifest entries
        directly.  Each entry must have ``uri`` and ``title``; ``section``
        and ``rootsection`` are optional.  Scoring is skipped (all chapters
        get ``score=1.0``, ``score_action="read"``) and the pre-filter gate
        is auto-satisfied — the conductor already excluded low-value
        chapters when building the manifest.
        """
        normalized_build_id = build_id.strip()
        if not normalized_build_id:
            raise ValueError("build_id is required for chapter work planning")
        if limit < 1 or limit > 32:
            raise ValueError("limit must be between 1 and 32")
        if audit_profile not in {"manual", "case"}:
            raise ValueError("audit_profile must be manual or case")
        bom_status = self.bom_enrichment_status()
        if str(bom_status.get("status", "")) == "pending":
            raw_pending_uris = bom_status.get("pending_uris", [])
            raw_missing_uris = bom_status.get("missing_uris", [])
            pending_uri_values = raw_pending_uris if isinstance(raw_pending_uris, list) else []
            missing_uri_values = raw_missing_uris if isinstance(raw_missing_uris, list) else []
            pending_bom_uris = [str(uri) for uri in pending_uri_values if str(uri)]
            missing_bom_uris = [str(uri) for uri in missing_uri_values if str(uri)]
            raise ValueError(
                "BOM identity is registered but semantic enrichment is incomplete; "
                "run plan_bom_enrichment() and complete guarded Component patches before "
                f"chapter planning. pending={pending_bom_uris[:10]}, missing={missing_bom_uris[:10]}",
            )

        plan_id = sha256(f"{normalized_build_id}\x1f{doc_id}".encode()).hexdigest()[:24]
        plan_key = f"index/chapter_plans/{plan_id}.json"
        persisted = self.store.read_json(plan_key)
        raw_chapters = persisted.get("chapters") if isinstance(persisted, dict) else None
        reusable_plan = (
            isinstance(persisted, dict)
            and persisted.get("version") == 4
            and persisted.get("doc_id") == doc_id
            and persisted.get("build_id") == normalized_build_id
            and persisted.get("audit_profile", "manual") == audit_profile
            and isinstance(raw_chapters, list)
            and all(
                isinstance(item, dict)
                and all(
                    str(item.get(field, "")).strip()
                    for field in ("uri", "packet_id", "shard_id", "task_description")
                )
                for item in raw_chapters
            )
        )
        if not reusable_plan:
            if chapter_manifest is not None:
                # Filter out parent entries that contain child entries — only
                # keep leaf (most granular) segments to prevent duplicate extraction.
                chapter_manifest = _filter_leaf_manifest_entries(chapter_manifest)
                # Build chapters from manifest — skip list_chapters, scoring, auto_no_entity
                chapters: list[dict[str, object]] = []
                for entry in chapter_manifest:
                    uri = str(entry["uri"])
                    title = str(entry["title"])
                    rootsection = str(entry.get("rootsection", ""))
                    section = str(entry.get("section", ""))
                    identity = sha256(f"{doc_id}\x1f{uri}".encode()).hexdigest()[:20]
                    packet_id = f"chapter_{identity}"
                    shard_id = f"chapter_{identity}"
                    idempotency_key = _chapter_idempotency_key(normalized_build_id, doc_id, uri)
                    chapters.append(
                        {
                            "uri": uri,
                            "rootsection": rootsection,
                            "section": section,
                            "subdir": "",
                            "title": title,
                            "md_path": "",
                            "packet_id": packet_id,
                            "shard_id": shard_id,
                            "idempotency_key": idempotency_key,
                            "score": 1.0,
                            "score_action": "read",
                            "auto_no_entity": False,
                            "auto_no_entity_reason": "",
                            "task_description": "\n".join(
                                [
                                    "worker_role: wiki_extraction_worker",
                                    "phase=1A_source_analysis",
                                    "chapter_count=1",
                                    f"build_id={normalized_build_id}",
                                    f"doc_id={doc_id}",
                                    f"audit_profile={audit_profile}",
                                    "depends_on_stage=bom_enriched",
                                    f"shard_id={shard_id}",
                                    f"chunk_id={shard_id}",
                                    "chunk_of=1",
                                    f"packet_id={packet_id}",
                                    f"idempotency_key={idempotency_key}",
                                    f"chapter_uri={uri}",
                                    "analysis_kinds=causal,procedure,assembly,specification,device",
                                    "heartbeat=task_update before first chapter read",
                                    "expected_artifacts=source_packet",
                                    (
                                        "Read this chapter only and call record_source_packet "
                                        "for the exact source URI."
                                    ),
                                ],
                            ),
                        },
                    )
                self.store.write_json(
                    plan_key,
                    {
                        "version": 4,
                        "plan_id": plan_id,
                        "doc_id": doc_id,
                        "build_id": normalized_build_id,
                        "audit_profile": audit_profile,
                        "chapter_count": len(chapters),
                        "chapters": chapters,
                    },
                    durable=True,
                )
                # Auto-satisfy pre-filter gate — conductor already filtered when building manifest
                from datetime import UTC, datetime

                self.store.write_json(
                    f"index/chapter_plans/{plan_id}.prefilter.json",
                    {
                        "source": "chapter_manifest",
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                    durable=True,
                )
            else:
                listed = self.list_chapters(doc_id, refresh=True)
                if not listed:
                    raise ValueError(f"Document has no readable leaf chapters: {doc_id}")
                paths = [str(chapter["md_path"]) for chapter in listed]
                with ThreadPoolExecutor(
                    max_workers=min(_io_worker_limit(), len(paths) or 1)
                ) as pool:
                    chapter_contents = list(pool.map(self._raw_fs.read_text, paths))
                toc_no_entity_decisions = [
                    should_auto_register_no_entity_from_toc(
                        rootsection=str(chapter["rootsection"]),
                        section=str(chapter["section"]),
                        title=str(chapter["title"]),
                    )
                    for chapter in listed
                ]
                fingerprint = build_fingerprint(
                    [
                        content
                        for content, toc_no_entity in zip(
                            chapter_contents,
                            toc_no_entity_decisions,
                            strict=True,
                        )
                        if isinstance(content, str) and not toc_no_entity
                    ],
                )
                chapters: list[dict[str, object]] = []
                for chapter, content, toc_auto_no_entity in zip(
                    listed,
                    chapter_contents,
                    toc_no_entity_decisions,
                    strict=True,
                ):
                    uri = self.make_source_uri(doc_id, str(chapter["subdir"]))
                    identity = sha256(f"{doc_id}\x1f{uri}".encode()).hexdigest()[:20]
                    packet_id = f"chapter_{identity}"
                    shard_id = f"chapter_{identity}"
                    idempotency_key = _chapter_idempotency_key(normalized_build_id, doc_id, uri)
                    score_record = (
                        score_chapter_record(content, fingerprint)
                        if isinstance(content, str)
                        else {
                            "score": 0.0,
                            "action": "read",
                            "signal_breakdown": {"unreadable": 1},
                        }
                    )
                    auto_no_entity = isinstance(content, str) and should_auto_register_no_entity(
                        content,
                        score_record,
                        directory_administrative=toc_auto_no_entity,
                    )
                    auto_no_entity_reason = ""
                    if auto_no_entity:
                        auto_no_entity_reason = (
                            "deterministic_directory_administrative"
                            if toc_auto_no_entity
                            else "deterministic_low_information"
                        )
                    chapters.append(
                        {
                            "uri": uri,
                            "rootsection": chapter["rootsection"],
                            "section": chapter["section"],
                            "subdir": chapter["subdir"],
                            "title": chapter["title"],
                            "md_path": chapter["md_path"],
                            "packet_id": packet_id,
                            "shard_id": shard_id,
                            "idempotency_key": idempotency_key,
                            "score": score_record["score"],
                            "score_action": score_record["action"],
                            "auto_no_entity": auto_no_entity,
                            "auto_no_entity_reason": auto_no_entity_reason,
                            "task_description": "\n".join(
                                [
                                    "worker_role: wiki_extraction_worker",
                                    "phase=1A_source_analysis",
                                    "chapter_count=1",
                                    f"build_id={normalized_build_id}",
                                    f"doc_id={doc_id}",
                                    f"audit_profile={audit_profile}",
                                    "depends_on_stage=bom_enriched",
                                    f"shard_id={shard_id}",
                                    f"chunk_id={shard_id}",
                                    "chunk_of=1",
                                    f"packet_id={packet_id}",
                                    f"idempotency_key={idempotency_key}",
                                    f"chapter_uri={uri}",
                                    "analysis_kinds=causal,procedure,assembly,specification,device",
                                    "heartbeat=task_update before first chapter read",
                                    "expected_artifacts=source_packet",
                                    "Read this chapter only and call record_source_packet for the exact source URI.",
                                ],
                            ),
                        },
                    )
                    if auto_no_entity:
                        assert isinstance(content, str)
                        self.record_source_packet(
                            packet_id=packet_id,
                            doc_id=doc_id,
                            source_uris=[uri],
                            status="complete",
                            evidence_count=0,
                            packet_body={
                                "kind": "no_entity",
                                "reason_code": auto_no_entity_reason,
                                "score": score_record["score"],
                                "signal_breakdown": score_record["signal_breakdown"],
                            },
                            source_contents={uri: content},
                            build_id=normalized_build_id,
                        )
                self.store.write_json(
                    plan_key,
                    {
                        "version": 4,
                        "plan_id": plan_id,
                        "doc_id": doc_id,
                        "build_id": normalized_build_id,
                        "audit_profile": audit_profile,
                        "chapter_count": len(chapters),
                        "chapters": chapters,
                    },
                    durable=True,
                )
        else:
            assert isinstance(raw_chapters, list)
            chapters = [dict(item) for item in raw_chapters]

        pending: list[dict[str, object]] = []
        completed_count = 0
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        ownership_checkpoint: Mapping[str, object] = (
            checkpoint
            if isinstance(checkpoint, dict)
            else {"doc_id": doc_id, "input_docs": [doc_id]}
        )
        # Map every source URI covered by a complete packet of this build to
        # that packet's key. Low-value registrations write ONE multi-URI packet
        # (e.g. '<build_id>_lowvalue'), while per-chapter completion only looks
        # at the chapter-scoped packet key; without this fallback those chapters
        # stay pending forever and get re-dispatched after pre-filtering.
        complete_uri_owners: dict[str, str] = {}
        for packet_key_chapter_scan in self.store.list_dir("source_packets", recursive=True):
            if not packet_key_chapter_scan.endswith(".json"):
                continue
            owning = self.store.read_json(packet_key_chapter_scan)
            if not isinstance(owning, dict) or str(owning.get("status", "")) != "complete":
                continue
            owning_build_id = str(owning.get("build_id", "")).strip()
            if owning_build_id and owning_build_id != normalized_build_id:
                continue
            raw_owning_uris = (
                owning.get("source_uris", []) if isinstance(owning.get("source_uris"), list) else []
            )
            for raw_uri in raw_owning_uris:
                if isinstance(raw_uri, str) and raw_uri not in complete_uri_owners:
                    complete_uri_owners[raw_uri] = packet_key_chapter_scan
        for chapter in chapters:
            packet_id = str(chapter.get("packet_id", ""))
            uri = str(chapter.get("uri", ""))
            packet_key = self._source_packet_key(packet_id) if packet_id else ""
            packet = self.store.read_json(packet_key) if packet_key else None
            completed = self._chapter_packet_complete_for_build(
                packet_key,
                packet if isinstance(packet, dict) else None,
                build_id=normalized_build_id,
                doc_id=doc_id,
                source_uri=uri,
                checkpoint=ownership_checkpoint,
            )
            if completed or uri in complete_uri_owners:
                completed_count += 1
            else:
                pending.append(chapter)
        active = {packet_id.strip() for packet_id in (active_packet_ids or []) if packet_id.strip()}
        dispatchable = [
            chapter for chapter in pending if str(chapter.get("packet_id", "")) not in active
        ]
        selected = dispatchable[:limit]
        auto_no_entity_count = sum(bool(chapter.get("auto_no_entity")) for chapter in chapters)
        pending_packet_ids = {str(ch.get("packet_id", "")) for ch in pending}

        def _build_preview_entry(ch: dict[str, object]) -> dict[str, object]:
            return {
                "title": str(ch.get("title", "")),
                "section": str(ch.get("section", "")),
                "rootsection": str(ch.get("rootsection", "")),
                "score": ch.get("score", 0.0),
                "score_action": ch.get("score_action", ""),
                "packet_id": str(ch.get("packet_id", "")),
                "uri": str(ch.get("uri", "")),
                "auto_no_entity": bool(ch.get("auto_no_entity")),
                "completed": str(ch.get("packet_id", "")) not in pending_packet_ids,
            }

        # ponytail: chapter_preview was returned on every call (~60K tokens for
        # 337 chapters). Now opt-in via preview=True with pagination so the
        # conductor pulls chapter metadata on demand instead of every response.
        preview_total = len(chapters)
        if preview:
            pv_end = min(preview_offset + preview_limit, preview_total)
            chapter_preview: list[dict[str, object]] = [
                _build_preview_entry(ch) for ch in chapters[preview_offset:pv_end]
            ]
            preview_has_more = pv_end < preview_total
        else:
            chapter_preview = []
            preview_has_more = preview_total > 0

        pre_filter_required = (
            self.store.read_json(
                f"index/chapter_plans/{plan_id}.prefilter.json",
            )
            is None
        )
        if pre_filter_required:
            selected = []
        return {
            "version": 4,
            "plan_id": plan_id,
            "doc_id": doc_id,
            "build_id": normalized_build_id,
            "chapter_count": len(chapters),
            "completed_count": completed_count,
            "auto_no_entity_count": auto_no_entity_count,
            "dispatchable_chapter_count": len(chapters) - completed_count,
            "pending_count": len(pending),
            "in_flight_count": len(pending) - len(dispatchable),
            "available_count": 0 if pre_filter_required else len(dispatchable),
            "chapters": selected,
            "work_items": selected,
            "count": len(selected),
            "chapter_preview": chapter_preview,
            "preview_total": preview_total,
            "preview_has_more": preview_has_more if preview else False,
            "preview_available": preview_total > 0,
            "pre_filter_required": pre_filter_required,
            "pre_filter_message": (
                "PRE_FILTER_REQUIRED: call plan_chapter_work with preview=True "
                "(and preview_offset/preview_limit for pagination) to retrieve "
                "chapter metadata for filtering. Then call "
                "register_no_entity_chapters(doc_id, build_id, "
                "packet_id='<build_id>_lowvalue', source_uris=[low-value URIs]) "
                "to skip administrative/preface chapters. work_items are withheld "
                "until pre-filtering is done."
            )
            if pre_filter_required
            else "",
            "done": not pending,
        }

    @staticmethod
    def _bom_mechanism_needs_enrichment(content: str | None) -> bool:
        """Return whether a Component page still has the registration stub."""
        if not content:
            return True
        marker = f"## {SECTION_MECHANISM}"
        start = content.find(marker)
        if start < 0:
            return True
        section = content[start + len(marker) :]
        next_heading = section.find("\n## ")
        if next_heading >= 0:
            section = section[:next_heading]
        body = section.strip()
        return not body or any(
            marker_text in body for marker_text in _BOM_ENRICH_PLACEHOLDER_MARKERS
        )

    def bom_enrichment_status(self, packet_id: str = "") -> dict[str, object]:
        """Inspect registered BOM Components without invoking a model.

        The identity packet is the durable input to this check.  A Component
        is pending until its page contains a non-placeholder ``工作机理``
        section, so the chapter preflight cannot silently bypass BOM enrich.
        """
        requested_packet_id = packet_id.strip()
        checkpoint = self.store.read_json("index/build_checkpoint.json")
        active_build_id = (
            str(checkpoint.get("build_id", "")).strip()
            if not requested_packet_id and isinstance(checkpoint, dict)
            else ""
        )
        packet_ids = (
            [requested_packet_id]
            if requested_packet_id
            else [
                Path(key).stem
                for key in self.store.list_dir("source_packets", recursive=False)
                if key.endswith(".json")
            ]
        )
        selected: list[tuple[str, dict[str, object]]] = []
        for candidate in packet_ids:
            packet = self.store.read_json(self._source_packet_key(candidate))
            if (
                active_build_id
                and isinstance(packet, dict)
                and str(packet.get("build_id", "")).strip() != active_build_id
            ):
                continue
            body = packet.get("packet") if isinstance(packet, dict) else None
            if not isinstance(body, dict) or body.get("kind") != "bom_identity_plan":
                continue
            selected.append((candidate, packet))
        if not selected:
            return {
                "status": "not_required",
                "packet_id": requested_packet_id,
                "packet_ids": [],
                "pending_uris": [],
                "missing_uris": [],
            }

        components_by_uri: dict[str, str] = {}
        for candidate, packet in selected:
            body = packet.get("packet")
            components = body.get("resolved_components", []) if isinstance(body, dict) else []
            for item in components:
                if not isinstance(item, dict):
                    continue
                uri = str(item.get("component_uri", "")).strip()
                if uri:
                    components_by_uri.setdefault(uri, candidate)
        component_uris = list(components_by_uri)
        pending_uris: list[str] = []
        missing_uris: list[str] = []
        for uri in component_uris:
            content = self.store.read_entity_by_uri(uri)
            if content is None:
                missing_uris.append(uri)
            elif self._bom_mechanism_needs_enrichment(content):
                pending_uris.append(uri)
        status = "pending" if pending_uris or missing_uris else "enriched"
        return {
            "status": status,
            "packet_id": requested_packet_id
            if requested_packet_id
            else (selected[0][0] if len(selected) == 1 else ""),
            "packet_ids": [candidate for candidate, _packet in selected],
            "component_count": len(component_uris),
            "component_uris": component_uris,
            "pending_uris": pending_uris,
            "missing_uris": missing_uris,
        }

    def plan_bom_enrichment(
        self,
        packet_id: str = "",
        *,
        max_parallel_shards: int = 1,
    ) -> dict[str, object]:
        """Create deterministic, URI-disjoint BOM enrich work descriptions.

        ``max_parallel_shards`` is supplied from the conductor's live worker
        capacity.  The service partitions the current pending URI set into at
        most that many balanced shards, while retaining the generic entity
        batch limit as a per-task safety bound.  When more work remains than
        one wave can safely carry, a later planner call returns the next wave.
        """
        if max_parallel_shards < 1:
            raise ValueError("max_parallel_shards must be positive")
        status = self.bom_enrichment_status(packet_id)
        selected_packet_id = str(status.get("packet_id", ""))
        if str(status.get("status", "")) == "not_required":
            return {
                "status": "not_required",
                "shards": [],
                "candidate_count": 0,
                "dispatch_count": 0,
                "remaining_count": 0,
                "shard_count": 0,
            }
        pending = {str(uri) for uri in status.get("pending_uris", []) if str(uri)}
        missing = {str(uri) for uri in status.get("missing_uris", []) if str(uri)}
        items: list[dict[str, str]] = []
        packet_ids = [str(value) for value in status.get("packet_ids", []) if str(value).strip()]
        for current_packet_id in packet_ids or [selected_packet_id]:
            packet = self.store.read_json(self._source_packet_key(current_packet_id))
            body = packet.get("packet") if isinstance(packet, dict) else None
            target_model = (
                str(body.get("target_model", "")).strip() if isinstance(body, dict) else ""
            )
            components = body.get("resolved_components", []) if isinstance(body, dict) else []
            for item in components:
                if not isinstance(item, dict):
                    continue
                uri = str(item.get("component_uri", "")).strip()
                if uri not in pending | missing:
                    continue
                items.append(
                    {
                        "component_uri": uri,
                        "class_name": str(item.get("class_name", "")),
                        "object_name": str(item.get("object_name", "")),
                        "bom_path": str(item.get("bom_path", "")),
                        "packet_id": current_packet_id,
                        "target_model": target_model,
                    },
                )
        unique_items = {str(item["component_uri"]): item for item in items}
        items = sorted(unique_items.values(), key=lambda item: str(item["component_uri"]))
        parallel_shards = min(max_parallel_shards, len(items)) if items else 0
        chunk_size = (
            min(
                _entity_batch_limit(),
                (len(items) + parallel_shards - 1) // parallel_shards,
            )
            if parallel_shards
            else 0
        )
        dispatch_items = items[: chunk_size * parallel_shards] if chunk_size else []
        shards: list[dict[str, object]] = []
        for index, start in enumerate(range(0, len(dispatch_items), chunk_size), 1):
            chunk = dispatch_items[start : start + chunk_size]
            uris = [str(item["component_uri"]) for item in chunk]
            shard_id = f"bom_enrich_{index}"
            shard_packet_ids = sorted({str(item["packet_id"]) for item in chunk})
            shard_packet_id = shard_packet_ids[0] if shard_packet_ids else selected_packet_id
            target_models = sorted({
                str(item["target_model"]) for item in chunk if str(item["target_model"]).strip()
            })
            target_model = target_models[0] if target_models else "unknown"
            shards.append(
                {
                    "shard_id": shard_id,
                    "component_uris": uris,
                    "write_set": uris,
                    "write_scope": f"bom_enrich:{shard_id}",
                    "task_description": "\n".join(
                        [
                            "worker_role: wiki_extraction_worker",
                            "phase=phase0",
                            "phase0_operation=bom_enrich",
                            "audit_profile=manual",
                            f"packet_id={shard_packet_id}",
                            f"packet_ids={json.dumps(shard_packet_ids, ensure_ascii=False, separators=(',', ':'))}",
                            f"target_model={target_model}",
                            "expected_artifacts=bom_enriched",
                            "depends_on_stage=bom_registered",
                            f"shard_id={shard_id}",
                            f"write_scope=bom_enrich:{shard_id}",
                            f"write_set={json.dumps(uris, ensure_ascii=False, separators=(',', ':'))}",
                            "bom_enrichment_items="
                            + json.dumps(chunk, ensure_ascii=False, separators=(",", ":")),
                            "Use diff_entity then one guarded patch_entity per Component.",
                        ],
                    ),
                },
            )
        return {
            "status": "pending" if shards else "enriched",
            "packet_id": selected_packet_id,
            "packet_ids": packet_ids,
            "shards": shards,
            "candidate_count": len(items),
            "dispatch_count": len(dispatch_items),
            "remaining_count": len(items) - len(dispatch_items),
            "shard_count": len(shards),
            "missing_uris": sorted(missing),
        }
