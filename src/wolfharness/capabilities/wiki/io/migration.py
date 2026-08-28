"""URI migration, source-URI sync, and read paths."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
from pathlib import Path
import re

from wolfharness.capabilities.wiki.io.text_parsers import (
    _dir_to_clean_title,
)
from wolfharness.capabilities.wiki.quality import (
    RawSourceKind,
    classify_raw_source_uri,
)
from wolfharness.capabilities.wiki.storage import (
    viking_read,
)


logger = logging.getLogger(__name__)

from wolfharness.capabilities.wiki._helpers import _CHAPTER_COMPONENT_RE


class MigrationMixin:
    """URI migration, source-URI sync, and read paths."""

    def create_subdir(self, concept: str, subdir_path: str) -> bool:
        """Create a subdirectory under a concept directory.

        ``subdir_path`` is relative to the concept root, e.g. ``"发动机/SC9D"``
        creates ``Component/发动机/SC9D/``.
        """
        if concept not in self.store.CONCEPT_DIRS or concept in {"OPA", "OPS", "OPL"}:
            raise ValueError(f"Cannot create entity subdirectories for concept: {concept}")
        relative = Path(subdir_path)
        if relative.is_absolute() or not subdir_path.strip():
            raise ValueError("subdir_path must be a non-empty relative path")
        base = self.store.CONCEPT_DIRS[concept]
        target = str(Path(base) / relative)
        if not target.startswith(base + "/") and target != base:
            raise ValueError("subdir_path escapes the concept directory")
        self.store.mkdir_p(target)
        logger.info("Subdir created: %s", target)
        return True

    def _rewrite_persisted_uri(self, old_uri: str, new_uri: str) -> list[str]:
        """Rewrite a moved/deleted URI in pages and published indexes.

        Returns the URIs of the pages whose content was rewritten (the
        ``from`` side of the moved link) so the caller can re-sync native
        graph edges for exactly those pages.
        """
        affected: list[str] = []
        # ponytail: rewrite only pages that link old_uri, resolved via the
        # persisted backlink index (one index read). When no index exists
        # (unbuilt library) fall back to a full scan so referrers are not lost.
        referrers = self.store.get_backlinks(old_uri)
        if not referrers:
            referrers = [
                record["uri"]
                for record in self._formal_entity_snapshot_records()
                if old_uri in record["content"]
            ]
        for uri in referrers:
            content = self.read_resource(uri)
            if content is None or old_uri not in content:
                continue
            updated = content.replace(old_uri, new_uri)
            expected_sha256 = sha256(content.encode("utf-8")).hexdigest()
            profile_identity = self.store.split_symptom_profile_uri(uri)
            if profile_identity is not None:
                parent_uri, profile_id = profile_identity
                self.write_symptom_profile(
                    parent_uri,
                    profile_id,
                    updated,
                    expected_sha256=expected_sha256,
                    skip_materialization=True,
                )
                affected.append(uri)
                continue
            identity = self.store.lookup_by_uri(uri)
            if identity is None:
                continue
            concept, class_name, object_name = identity
            self.merge_entity(
                concept,
                class_name or "",
                object_name,
                updated,
                expected_sha256=expected_sha256,
                skip_materialization=True,
            )
            affected.append(uri)
        for relative_path in ("resource.json", "index/concepts_index.json"):
            raw = self.store.read_text(relative_path)
            if raw is None:
                continue
            updated = raw.replace(old_uri, new_uri)
            if updated != raw:
                self.store.write_text(relative_path, updated)
        return affected

    def migrate_source_uri_references(
        self,
        old_uri: str,
        new_uri: str,
        *,
        entity_uris: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        """Patch Device references when a manual chapter receives a new URI.

        Only exact references to ``old_uri`` are replaced. Each affected Device
        is read and patched independently with its current SHA-256, so a
        concurrent or expert-owned page cannot be overwritten. Repeating the
        same migration is idempotent: pages no longer containing ``old_uri``
        are reported as skipped.

        Args:
            old_uri: Previous readable raw manual URI.
            new_uri: New readable raw manual URI.
            entity_uris: Optional explicit Device URIs. Defaults to all Devices.
            dry_run: Return the planned pages without writing them.

        Returns:
            Counts and per-page results. Expert-authority or optimistic-lock
            conflicts are isolated in ``blocked`` and do not stop other pages.
        """
        previous_uri = old_uri.strip()
        replacement_uri = new_uri.strip()
        if not previous_uri or not replacement_uri:
            raise ValueError("old_uri and new_uri are required")
        if previous_uri == replacement_uri:
            raise ValueError("old_uri and new_uri must be different")
        if classify_raw_source_uri(previous_uri, raw_root_uri=self._raw_fs.root_uri) is None:
            raise ValueError(f"old_uri is not a supported raw source URI: {previous_uri}")
        if classify_raw_source_uri(replacement_uri, raw_root_uri=self._raw_fs.root_uri) is None:
            raise ValueError(f"new_uri is not a supported raw source URI: {replacement_uri}")
        if self.read_raw_resource(replacement_uri) is None:
            raise ValueError(f"new_uri is not readable: {replacement_uri}")

        selected: list[tuple[str, str, str]] = []
        if entity_uris is None:
            selected = [
                (class_name or "", object_name, uri)
                for _concept, class_name, object_name, uri in self.store.list_entities("Device")
            ]
        else:
            for requested_uri in dict.fromkeys(uri.strip() for uri in entity_uris if uri.strip()):
                identity = self.store.lookup_by_uri(requested_uri)
                if identity is None or identity[0] != "Device":
                    raise ValueError(f"entity_uris must identify Device pages: {requested_uri}")
                selected.append((identity[1] or "", identity[2], identity[3]))

        planned: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        updated: list[dict[str, object]] = []
        blocked: list[dict[str, object]] = []
        for class_name, object_name, entity_uri in selected:
            content = self.store.read_entity_by_uri(entity_uri)
            if content is None:
                blocked.append({"uri": entity_uri, "reason": "entity_not_readable"})
                continue
            if previous_uri not in content:
                skipped.append({"uri": entity_uri, "reason": "old_uri_not_present"})
                continue
            current_sha256 = sha256(content.encode("utf-8")).hexdigest()
            lines = content.splitlines(keepends=True)
            operations = [
                {
                    "op": "line_replace",
                    "start": line_number,
                    "end": line_number,
                    "content": line.replace(previous_uri, replacement_uri),
                }
                for line_number, line in enumerate(lines, 1)
                if previous_uri in line
            ]
            item: dict[str, object] = {
                "uri": entity_uri,
                "class_name": class_name,
                "object_name": object_name,
                "expected_sha256": current_sha256,
                "replacement_count": sum(line.count(previous_uri) for line in lines),
            }
            if dry_run:
                planned.append(item)
                continue
            try:
                self.patch_entity(
                    "Device",
                    class_name,
                    object_name,
                    operations,
                    expected_sha256=current_sha256,
                    reference_replacements=[(previous_uri, replacement_uri)],
                )
            except (FileNotFoundError, OSError, ValueError) as error:
                blocked.append({**item, "reason": str(error)})
            else:
                updated.append(item)

        return {
            "old_uri": previous_uri,
            "new_uri": replacement_uri,
            "dry_run": dry_run,
            "scanned_count": len(selected),
            "planned_count": len(planned),
            "updated_count": len(updated),
            "skipped_count": len(skipped),
            "blocked_count": len(blocked),
            "planned": planned,
            "updated": updated,
            "skipped": skipped,
            "blocked": blocked,
            "idempotent": True,
        }

    def sync_device_system_chapters(
        self,
        doc_id: str,
        device_id: str,
    ) -> dict[str, object]:
        """Synchronize a Device's raw chapter navigation from the source tree.

        Legacy/incremental builds can create the Device skeleton before the
        chapter planner has run. The later chapter packets then provide real
        source evidence, but nothing used to backfill the existing Device
        page. This operation updates only the ``system_chapters`` frontmatter
        field; all other Device content, including expert-owned sections, is
        left untouched.

        Each entry carries the chapter number, readable title and exact raw
        URI. Re-running the operation is a no-op when the chapter inventory has
        not changed. An empty local chapter inventory is not treated as a
        successful sync because external-MCP builds must supply their own
        citations from the source service.
        """
        normalized_doc_id = doc_id.strip()
        normalized_device_id = device_id.strip()
        if not normalized_doc_id or not normalized_device_id:
            raise ValueError("doc_id and device_id must not be empty")

        current = self.store.read_entity("Device", None, normalized_device_id)
        if current is None:
            # ponytail: fuzzy fallback — BOM may register Device under a different object_name
            for _concept, _cls, obj_name, _uri in self.store.list_entities("Device"):
                if obj_name.startswith(normalized_device_id) or normalized_device_id.startswith(
                    obj_name
                ):
                    current = self.store.read_entity("Device", None, obj_name)
                    if current is not None:
                        normalized_device_id = obj_name
                        break
        if current is None:
            return {
                "status": "skipped",
                "reason": "device_not_found",
                "doc_id": normalized_doc_id,
                "device_id": normalized_device_id,
                "chapter_count": 0,
                "updated": False,
            }

        chapters = self.list_chapters(normalized_doc_id, refresh=True)
        if not chapters:
            return {
                "status": "skipped",
                "reason": "no_local_chapters",
                "doc_id": normalized_doc_id,
                "device_id": normalized_device_id,
                "chapter_count": 0,
                "updated": False,
            }

        chapter_rows: list[tuple[str, str]] = []
        for chapter in chapters:
            subdir = str(chapter.get("subdir", "")).strip()
            if not subdir:
                continue
            uri = self.make_source_uri(normalized_doc_id, subdir)
            parts = [part for part in subdir.split("/") if part]
            leaf = parts[-1] if parts else ""
            if leaf.endswith(".md"):
                leaf = leaf[:-3]
            number_match = _CHAPTER_COMPONENT_RE.match(leaf)
            chapter_number = number_match.group(0).replace("_", ".") if number_match else ""
            title = str(chapter.get("title", "")).strip() or _dir_to_clean_title(leaf)
            title = re.sub(r"_\d+more_[0-9a-f]+(?:_\d+)?$", "", title).replace("_", " ").strip()
            label = (
                " ".join(value for value in (chapter_number, title) if value).strip()
                or uri.rsplit("/", 1)[-1]
            )
            chapter_rows.append((label, uri))

        if not chapter_rows:
            return {
                "status": "skipped",
                "reason": "chapters_have_no_readable_uri",
                "doc_id": normalized_doc_id,
                "device_id": normalized_device_id,
                "chapter_count": 0,
                "updated": False,
            }

        chapter_rows = list(dict.fromkeys(chapter_rows))
        chapter_uris = [uri for _label, uri in chapter_rows]
        current_hash = sha256(current.encode("utf-8")).hexdigest()
        operations: list[dict[str, object]] = [
            {"op": "fm_set_list", "field": "system_chapters", "values": chapter_uris},
        ]
        before = current
        self.patch_entity(
            "Device",
            "",
            normalized_device_id,
            operations,
            expected_sha256=current_hash,
        )
        after = self.store.read_entity("Device", None, normalized_device_id) or before
        return {
            "status": "updated" if after != before else "unchanged",
            "reason": "chapter_inventory_synced",
            "doc_id": normalized_doc_id,
            "device_id": normalized_device_id,
            "chapter_count": len(chapter_rows),
            "updated": after != before,
            "idempotent": True,
            "source_uris": chapter_uris,
        }

    def read_resource(self, uri: str, line_numbers: bool = False) -> str | None:
        """Resolve a ``{root_uri}/<Concept>/<hash>`` URI to file content.

        When *line_numbers* is True, each line is prefixed with its 1-indexed
        line number formatted as ``NN: <content>`` for precise patch
        positioning.

        Delegates to ``read_entity_by_uri`` (hash-based lookup); falls back
        to ``_uri_to_path`` for backward compatibility.
        """
        resource_uri, separator, fragment = uri.partition("#")
        # Follow persisted redirects (old URI -> moved location) before any
        # backend read.  Directories reorganized after materialization leave
        # stale canonical URIs in OPA/OPS evidence and entity bodies; without
        # this step every read of a redirected URI returns None even though
        # register_redirect/move_entity correctly persist the mapping.
        resolved_uri = self.store.resolve_redirect(resource_uri)
        if resolved_uri != resource_uri:
            resource_uri = resolved_uri
        content: str | None = None
        if self.store.is_wiki_uri(resource_uri):
            # Wiki namespace: OP records keep their real backend URI; entity
            # canonical URIs (no `.md`) must go through the hash-entity mapping
            # layer, otherwise the canonical→physical-file resolution is
            # bypassed and every read returns None (dangling audit).
            relative = resource_uri.removeprefix(self.store.root_uri + "/")
            if relative.startswith("OP/"):
                content = self.store.read_text(relative)
            elif relative.startswith("source_packets/") and relative.endswith(".json"):
                raw = self.store.read_json(relative)
                content = json.dumps(raw, ensure_ascii=False, indent=2) if raw is not None else None
            else:
                content = self.store.read_entity_by_uri(resource_uri)
                if content is None and resource_uri.endswith(".md"):
                    content = self.store.read_entity_by_uri(resource_uri.removesuffix(".md"))
                if content is None:
                    path = self._uri_to_path(resource_uri)
                    if path is not None:
                        content = self.store.read_text(self.store._key_of(path))
        elif resource_uri.startswith("viking://resources/"):
            # Cross-namespace OpenViking read — arbitrary remote URI (raw
            # manuals, cases, repairmenus) regardless of which namespace it
            # lives in.  Only reached for URIs outside the active wiki store.
            content = viking_read(resource_uri)
        elif self._bom_fs is not None and resource_uri.startswith(self._bom_fs.root_uri + "/"):
            content = self._bom_fs.read_text(
                resource_uri.removeprefix(self._bom_fs.root_uri + "/"),
            )
        elif resource_uri.startswith(self._raw_fs.root_uri + "/"):
            content = self.read_raw_resource(resource_uri)
        elif (
            classify_raw_source_uri(resource_uri, raw_root_uri=self._raw_fs.root_uri)
            is RawSourceKind.EXTERNAL
        ):
            content = ""
        else:
            result = self.store.read_entity_by_uri(resource_uri)
            if result is not None:
                content = result
            else:
                path = self._uri_to_path(resource_uri)
                if path is not None:
                    content = self.store.read_text(self.store._key_of(path))

        if content is not None and separator:
            content = self._read_markdown_fragment(content, fragment)

        if content is not None and line_numbers:
            lines = content.splitlines()
            max_width = len(str(len(lines)))
            content = "\n".join(f"{i + 1:>{max_width}}: {line}" for i, line in enumerate(lines))
        return content

    def read_raw_resource(self, uri: str) -> str | None:
        """Resolve a real-path raw chapter URI to chapter markdown content.

        Accepts URIs shaped ``{root_uri}/{doc_id}/chapters/<subdir>/chapter.md``
        where ``root_uri`` is the active raw backend root.  Also supports
        title-based lookup via ``read_chapter_map`` when the URI's tail
        matches a chapter title.

        Returns ``None`` for invalid or unresolvable URIs.
        """
        cached = self._raw_resource_cache.get(uri)
        if cached is not None or uri in self._raw_resource_cache:
            return cached
        root_prefix = self._raw_fs.root_uri + "/"
        if not uri.startswith(root_prefix):
            logger.warning("Not a raw chapter URI: %s", uri)
            return None
        remainder = uri.removeprefix(root_prefix)
        content = self._raw_fs.read_uri(uri)
        if content is not None:
            self._raw_resource_cache[uri] = content
            return content
        parts = remainder.split("/chapters/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            logger.warning("Malformed raw chapter URI: %s", uri)
            self._raw_resource_cache[uri] = None
            return None
        doc_id, subdir_tail = parts
        if not subdir_tail.endswith("/chapter.md"):
            logger.warning("Raw chapter URI does not point to chapter.md: %s", uri)
            self._raw_resource_cache[uri] = None
            return None
        subdir = subdir_tail[: -len("/chapter.md")]
        if not subdir:
            logger.warning("Empty chapter subdir in URI: %s", uri)
            self._raw_resource_cache[uri] = None
            return None
        try:
            content = self.read_chapter(doc_id, subdir)
        except (FileNotFoundError, ValueError):
            pass
        else:
            self._raw_resource_cache[uri] = content
            return content
        try:
            mapping = self.read_chapter_map(doc_id)
            if subdir in mapping:
                target_uri = mapping[subdir]
                target_subdir = target_uri.split("/chapters/", 1)[1].removesuffix("/chapter.md")
                content = self.read_chapter(doc_id, target_subdir)
                self._raw_resource_cache[uri] = content
                return content
            subdir_lower = subdir.lower().strip()
            for key, key_uri in mapping.items():
                if key.lower().strip() == subdir_lower:
                    target_subdir = key_uri.split("/chapters/", 1)[1].removesuffix("/chapter.md")
                    content = self.read_chapter(doc_id, target_subdir)
                    self._raw_resource_cache[uri] = content
                    return content
        except (FileNotFoundError, ValueError, IndexError):
            pass
        logger.warning("Cannot resolve raw resource %s", uri)
        self._raw_resource_cache[uri] = None
        return None

    def _uri_to_path(self, uri: str) -> Path | None:
        """Convert a ``{root_uri}/...`` URI to a filesystem path.

        Note: with flat hash URIs (``{root_uri}/Concept/<hash>``), this
        method only works when ``<hash>`` looks like a filename.  New code
        should prefer ``read_entity_by_uri`` instead.
        """
        if self.store.is_wiki_uri(uri):
            parts = uri[len(self.store.root_uri) + 1 :].split("/")
        else:
            return None
        if not parts:
            return None
        concept = parts[0]
        dir_name = self.store.CONCEPT_DIRS.get(concept, concept)
        base = self.store.root / dir_name
        if len(parts) == 1:
            return None  # No object name
        # Last part is object_name, middle parts are class path.
        object_name = parts[-1]
        class_parts = parts[1:-1]
        for cp in class_parts:
            base = base / cp
        if concept in self.store.DIRECTORY_CONCEPTS:
            return base / object_name / "index.md"
        return base / f"{object_name}.md"

    # ── Semantic retrieval ─────────────────────────────────────────────────

    def find_wiki(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        deep: bool = False,
    ) -> list[dict]:
        """Use OpenViking's native bounded retrieval over wiki or raw scope.

        The caller may scope retrieval to the configured Wiki root or raw
        manual root.  No local path scan is used for semantic discovery.
        """
        target = target_uri.rstrip("/")
        allowed = (self.store.root_uri, self._raw_fs.root_uri)
        if target and not any(target == root or target.startswith(root + "/") for root in allowed):
            raise ValueError("target_uri must belong to the configured wiki or raw OpenViking root")
        return self.store._fs.find(query, target_uri=target, limit=limit, deep=deep)

    def search_wiki(self, query: str, *, limit: int = 10) -> list[dict]:
        """Compatibility alias for native OpenViking ``find`` retrieval."""
        return self.find_wiki(query, limit=limit)
