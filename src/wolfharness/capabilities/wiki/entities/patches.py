"""Patch, diff, merge, move, and classification operations."""

from __future__ import annotations

from difflib import unified_diff
from hashlib import sha256
import logging
from pathlib import Path

from wolfharness.capabilities.wiki.auto_repair import materialize_body_links
from wolfharness.capabilities.wiki.io.text_parsers import (
    _ENTITY_URI_HEADING_RE,
    _parse_forward_links,
)
from wolfharness.capabilities.wiki.quality import (
    entity_status,
    parse_frontmatter,
)
from wolfharness.capabilities.wiki.validation import (
    require_valid_entity,
)


logger = logging.getLogger(__name__)

from wolfharness.capabilities.wiki._helpers import _FORMAL_WRITE_HOOKS, _entity_batch_limit


class PatchMixin:
    """Patch, diff, merge, move, and classification operations."""

    def patch_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        operations: list[dict],
        *,
        expected_sha256: str = "",
        reference_replacements: list[tuple[str, str]] | None = None,
    ) -> str:
        """Apply patch operations to an existing entity in-place.

        Supports three operation families:

        - **Line-based** (1-indexed, relative to entire file):
          - ``{"op": "line_replace", "start": int, "end": int, "content": str}``
          - ``{"op": "line_insert", "at": int, "content": str}``
          - ``{"op": "line_delete", "start": int, "end": int}``

        - **Section-based** (identified by ``## `` heading text):
          - ``{"op": "section_replace", "heading": str, "content": str}``
          - ``{"op": "section_insert_after", "heading": str, "heading_new": str, "content": str}``

        - **Frontmatter** (YAML fields between ``---`` markers):
          - ``{"op": "fm_append", "field": str, "values": list[str]}``
          - ``{"op": "fm_set", "field": str, "value": str}``
          - ``{"op": "fm_set_list", "field": str, "values": list[str]}``

        ``expected_sha256`` is the hash of the page read before preparing the
        operations.  It is mandatory for an existing page; a stale hash is
        rejected so the caller must re-read and re-diff instead of overwriting
        a concurrent update.

        Operations are applied sequentially.  After all patches, the same
        post-processing as ``write_entity`` runs (resolve_body_refs,
        repair_frontmatter, dedup_citations).

        Returns the canonical ``{root_uri}/...`` URI.
        """
        self._invalidate_audit_cache()
        clz = class_name or None
        content = self.store.read_entity(concept, clz, object_name)
        if content is None:
            raise FileNotFoundError(
                f"Entity not found: {self.store.entity_uri(concept, clz, object_name)}",
            )
        current_sha256 = sha256(content.encode("utf-8")).hexdigest()
        if not expected_sha256:
            raise ValueError(
                "Existing entity patches require expected_sha256 from diff_entity/read_resource before patch_entity",
            )
        if expected_sha256 != current_sha256:
            raise ValueError(
                f"Entity changed before patch; rerun diff_entity before patch (expected={expected_sha256}, actual={current_sha256}).",
            )

        # Capture original model prefix for cross-model contamination guard.
        _orig_class_name = ""
        if concept == "DTC":
            _orig_cn = parse_frontmatter(content).get("class_name")
            if isinstance(_orig_cn, str):
                _orig_class_name = _orig_cn.strip()

        for op in operations:
            op_type = op.get("op", "")
            if op_type in ("line_replace", "line_insert", "line_delete"):
                content = self._apply_line_op(content, op)
            elif op_type in ("section_replace", "section_insert_after"):
                content = self._apply_section_op(content, op)
            elif op_type in ("fm_append", "fm_set", "fm_set_list"):
                content = self._apply_fm_op(content, op)
            else:
                raise ValueError(f"Unknown patch op: {op_type!r}")

        # Post-process — same pipeline as write_entity.
        uri = self.store.entity_uri(concept, clz, object_name)
        content = _ENTITY_URI_HEADING_RE.sub(f"# {uri}\n", content, count=1)
        content = self.store.repair_frontmatter(content, None)
        content = self.store.resolve_body_refs(content, None)
        content = materialize_body_links(content, concept, self)
        self._reject_wrong_raw_refs(content)
        self._reject_malformed_wiki_refs(content)
        content = self.store.dedup_citations(content)
        content = self._dedupe_h2_sections(content)
        content = self._preserve_expert_sections(
            target_uri=uri,
            current=self.store.read_entity(concept, clz, object_name) or "",
            candidate=content,
        )
        if entity_status(content) == "confirmed":
            if concept == "Symptom" and not self.list_symptom_profiles(uri):
                raise ValueError(
                    "Symptom cannot become confirmed without at least one Profile",
                )
            require_valid_entity(
                content=content,
                concept=concept,
                class_name=class_name,
                object_name=object_name,
                hooks=_FORMAL_WRITE_HOOKS,
            )
        else:
            self._validate_formal_write(
                content=content,
                concept=concept,
                class_name=class_name,
                object_name=object_name,
            )
        # Guard: block cross-model DTC patch — different model prefix means a
        # different entity that must be created via write_entity, not merged.
        if concept == "DTC" and _orig_class_name:
            _new_cn = parse_frontmatter(content).get("class_name")
            _new_class_name = _new_cn.strip() if isinstance(_new_cn, str) else ""
            _orig_prefix = _orig_class_name.split("_", 1)[0]
            _new_prefix = _new_class_name.split("_", 1)[0]
            if _orig_prefix and _new_prefix and _orig_prefix != _new_prefix:
                raise ValueError(
                    f"Cross-model DTC patch blocked: class_name model prefix "
                    f"changed {_orig_prefix!r} → {_new_prefix!r}. "
                    f"Use write_entity to create a separate DTC for {_new_prefix}.",
                )
        if content == self.store.read_entity(concept, clz, object_name):
            logger.info("Entity patch skipped (no change): %s", uri)
            return uri
        self.merge_entity(
            concept,
            class_name,
            object_name,
            content,
            expected_sha256=current_sha256,
            reference_replacements=reference_replacements,
        )
        logger.info("Entity patched: %s (%d ops, %d chars)", uri, len(operations), len(content))
        if concept == "Component":
            self._sync_component_narrative_links(uri)
        elif concept in {"Fault", "Procedure"}:
            relation_field = "affected_components" if concept == "Fault" else "target_components"
            title = parse_frontmatter(content).get("title")
            label = title.strip() if isinstance(title, str) and title.strip() else object_name
            for component_uri in self._frontmatter_uri_values(
                self.store.root_uri, parse_frontmatter(content), relation_field
            ):
                canonical_component_uri = component_uri.split("#", 1)[0]
                if concept == "Fault":
                    self._sync_component_narrative_links(
                        canonical_component_uri,
                        fault_links=[(label, uri)],
                    )
                else:
                    self._sync_component_narrative_links(
                        canonical_component_uri,
                        procedure_links=[(label, uri)],
                    )
        return uri

    def patch_entities_batch(
        self,
        patches: list[dict[str, object]],
        *,
        sync_component_links: bool = True,
        preloaded_contents: dict[str, str] | None = None,
    ) -> dict[str, object]:
        """Apply guarded patches to independent existing entities in one commit.

        Relation closure is the main caller: it already groups all relation
        updates by source URI, so sending one remote write per page only adds
        transport latency.  Every item still carries its own optimistic hash;
        the batch is prepared and validated before any page is committed.
        """
        if not patches:
            raise ValueError("patches must not be empty")
        batch_limit = _entity_batch_limit()
        if len(patches) > batch_limit:
            raise ValueError(f"patch_entities_batch accepts at most {batch_limit} patches")

        prepared: list[tuple[str, str, str, str, str, int]] = []
        sync_targets: set[str] = set()
        known_uris = {record[3] for record in self.store.list_entities()}
        identities: set[tuple[str, str, str]] = set()
        for index, item in enumerate(patches):
            concept = item.get("concept")
            class_name = item.get("class_name", "")
            object_name = item.get("object_name")
            operations = item.get("operations")
            expected_sha256 = item.get("expected_sha256")
            if not all(
                isinstance(value, str)
                for value in (concept, class_name, object_name, expected_sha256)
            ):
                raise TypeError(f"patches[{index}] identity/hash fields must be strings")
            if not isinstance(operations, list) or any(
                not isinstance(op, dict) for op in operations
            ):
                raise TypeError(f"patches[{index}].operations must be a list of objects")
            assert isinstance(concept, str)
            assert isinstance(class_name, str)
            assert isinstance(object_name, str)
            assert isinstance(expected_sha256, str)
            identity = (concept, class_name, object_name)
            if identity in identities:
                raise ValueError(f"duplicate entity identity in batch: {identity}")
            identities.add(identity)
            uri = self.store.entity_uri(concept, class_name or None, object_name)
            if preloaded_contents is not None and uri in preloaded_contents:
                current = preloaded_contents[uri]
            else:
                current = self.store.read_entity(concept, class_name or None, object_name)
            if current is None:
                raise FileNotFoundError(
                    f"Entity not found: {uri}",
                )
            current_sha256 = sha256(current.encode("utf-8")).hexdigest()
            if not expected_sha256:
                raise ValueError(
                    f"Existing batch entity requires expected_sha256: {identity}",
                )
            if expected_sha256 != current_sha256:
                raise ValueError(f"Batch entity changed since diff: {identity}")

            original_class_name = ""
            if concept == "DTC":
                original_class_value = parse_frontmatter(current).get("class_name")
                if isinstance(original_class_value, str):
                    original_class_name = original_class_value.strip()

            content = self._apply_operations(current, operations)
            uri = self.store.entity_uri(concept, class_name or None, object_name)
            content = _ENTITY_URI_HEADING_RE.sub(f"# {uri}\n", content, count=1)
            content = self.store.repair_frontmatter(content, None)
            content = self.store.resolve_body_refs(content, None)
            content = materialize_body_links(content, concept, self, known_uris=known_uris)
            self._reject_wrong_raw_refs(content)
            self._reject_malformed_wiki_refs(content)
            content = self.store.dedup_citations(content)
            content = self._dedupe_h2_sections(content)
            content = self._preserve_expert_sections(
                target_uri=uri,
                current=current,
                candidate=content,
            )
            if entity_status(content) == "confirmed":
                if concept == "Symptom" and not self.list_symptom_profiles(uri):
                    raise ValueError(
                        "Symptom cannot become confirmed without at least one Profile",
                    )
                require_valid_entity(
                    content=content,
                    concept=concept,
                    class_name=class_name,
                    object_name=object_name,
                    hooks=_FORMAL_WRITE_HOOKS,
                )
            else:
                self._validate_formal_write(
                    content=content,
                    concept=concept,
                    class_name=class_name,
                    object_name=object_name,
                )
            if concept == "DTC" and original_class_name:
                new_class_value = parse_frontmatter(content).get("class_name")
                new_class_name = new_class_value.strip() if isinstance(new_class_value, str) else ""
                original_prefix = original_class_name.split("_", 1)[0]
                new_prefix = new_class_name.split("_", 1)[0]
                if original_prefix and new_prefix and original_prefix != new_prefix:
                    raise ValueError(
                        "Cross-model DTC patch blocked: class_name model prefix "
                        f"changed {original_prefix!r} → {new_prefix!r}. "
                        f"Use write_entity to create a separate DTC for {new_prefix}.",
                    )
            if content != current:
                content = self._mark_merge_conflict(concept, uri, current, content)
            frontmatter = parse_frontmatter(content)
            if concept == "Component":
                sync_targets.add(uri)
            elif concept in {"Fault", "Procedure"}:
                relation_field = (
                    "affected_components" if concept == "Fault" else "target_components"
                )
                sync_targets.update(
                    value.split("#", 1)[0]
                    for value in self._frontmatter_uri_values(
                        self.store.root_uri,
                        frontmatter,
                        relation_field,
                    )
                )
            prepared.append((
                concept,
                class_name,
                object_name,
                content,
                expected_sha256,
                len(current),
            ))

        self._invalidate_audit_cache()
        if self._log:
            for concept, class_name, object_name, _content, _expected, _before in prepared:
                self._log.mutation_attempt(
                    self.store.entity_uri(concept, class_name or None, object_name),
                    "patch_entities_batch",
                )
        with self._relation_sync_lock:
            committed: list[tuple[str, str | None, str, str]] = []
            for concept, class_name, object_name, content, expected, _before in prepared:
                latest = self.store.read_entity(concept, class_name or None, object_name)
                latest_sha256 = sha256((latest or "").encode("utf-8")).hexdigest()
                if latest is None or latest_sha256 != expected:
                    raise ValueError(
                        f"Batch entity changed before commit; rerun diff_entity before patch ({concept}/{class_name}/{object_name}).",
                    )
                # Re-merge expert-owned sections against the fresh commit-time
                # read; the merged content must actually reach the write.
                merged = self._preserve_expert_sections(
                    target_uri=self.store.entity_uri(concept, class_name or None, object_name),
                    current=latest,
                    candidate=content,
                )
                committed.append((concept, class_name or None, object_name, merged))
            self.store.write_entities(committed)
        uris: list[str] = []
        for concept, class_name, object_name, content, _expected, char_before in prepared:
            uri = self.store.entity_uri(concept, class_name or None, object_name)
            self.store.register_natural_key(concept, class_name or None, object_name, uri)
            uris.append(uri)
            if self._log:
                self._log.mutation_applied(uri, "patch_entities_batch")
                self._log.entity_updated(
                    uri,
                    concept,
                    object_name,
                    len(content),
                    char_before,
                    class_name=class_name or None,
                    reason="batch_patch",
                )
        if sync_component_links:
            for component_uri in sorted(sync_targets):
                self._sync_component_narrative_links(component_uri, force=True)
        return {"patched_count": len(uris), "uris": uris, "idempotent": True}

    def _apply_operations(self, content: str, operations: list[dict]) -> str:
        """Apply the shared line/section/frontmatter patch vocabulary."""
        for op in operations:
            op_type = op.get("op", "")
            if op_type in ("line_replace", "line_insert", "line_delete"):
                content = self._apply_line_op(content, op)
            elif op_type in ("section_replace", "section_insert_after"):
                content = self._apply_section_op(content, op)
            elif op_type in ("fm_append", "fm_set", "fm_set_list"):
                content = self._apply_fm_op(content, op)
            else:
                raise ValueError(f"Unknown patch op: {op_type!r}")
        return content

    def diff_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        candidate_content: str,
    ) -> dict[str, object]:
        """Return a deterministic diff and content hashes before a merge."""
        current = self.store.read_entity(concept, class_name or None, object_name)
        current_hash = sha256((current or "").encode("utf-8")).hexdigest()
        candidate_hash = sha256(candidate_content.encode("utf-8")).hexdigest()
        diff = "".join(
            unified_diff(
                (current or "").splitlines(keepends=True),
                candidate_content.splitlines(keepends=True),
                fromfile="current",
                tofile="candidate",
            ),
        )
        return {
            "uri": self.store.entity_uri(concept, class_name or None, object_name),
            "exists": current is not None,
            "changed": current != candidate_content,
            "current_sha256": current_hash,
            "candidate_sha256": candidate_hash,
            "diff": diff,
        }

    def merge_entity(
        self,
        concept: str,
        class_name: str,
        object_name: str,
        content: str,
        *,
        expected_sha256: str = "",
        conflict_policy: str = "detect",
        skip_materialization: bool = False,
        reference_replacements: list[tuple[str, str]] | None = None,
    ) -> str:
        """Materialize an explicit candidate with an optimistic-lock check."""
        current = self.store.read_entity(concept, class_name or None, object_name)
        current_hash = sha256((current or "").encode("utf-8")).hexdigest()
        if expected_sha256 and expected_sha256 != current_hash:
            raise ValueError(
                f"Entity changed since diff; rerun diff_entity before merge (expected={expected_sha256}, actual={current_hash}).",
            )
        if current is not None and not expected_sha256:
            raise ValueError(
                "Existing entity merges require expected_sha256 from diff_entity",
            )
        return self.write_entity(
            concept,
            class_name,
            object_name,
            content,
            expected_sha256=expected_sha256,
            conflict_policy=conflict_policy,
            skip_materialization=skip_materialization,
            reference_replacements=reference_replacements,
        )

    def delete_entity(self, concept: str, class_name: str, object_name: str) -> bool:
        """Delete an entity and its persistent identity-index entries."""
        self._invalidate_audit_cache()
        clz = class_name or None
        uri = self.store.lookup_natural_key(concept, clz, object_name)
        if uri is not None:
            self.rebuild_all_backlinks()
            backlinks = [source for source in self.get_backlinks(uri) if source != uri]
            if backlinks:
                raise ValueError(
                    f"Delete blocked: entity is still referenced by {len(backlinks)} page(s). Repair those references first: {backlinks[:5]}",
                )
        deleted = False
        path = self.store.entity_path(concept, clz, object_name)
        key = self.store._key_of(path)
        if self.store.exists(key):
            if concept in self.store.DIRECTORY_CONCEPTS:
                prefix = self.store._key_of(path.parent)
                for child_key in self.store.list_dir(prefix, recursive=True):
                    self.store.delete(child_key)
                if self.store.exists(key):
                    raise OSError(f"Failed to remove entity directory: {path.parent}")
                logger.info("Entity deleted (directory): %s", path.parent)
            else:
                self.store.delete(key)
                logger.info("Entity deleted: %s", path)
            deleted = True
        legacy = self.store.legacy_entity_path(concept, clz, object_name)
        if not deleted and legacy is not None and self.store.exists(self.store._key_of(legacy)):
            self.store.delete(self.store._key_of(legacy))
            logger.info("Entity deleted (legacy): %s", legacy)
            deleted = True
        if uri is not None:
            self.store.unregister_uri(uri)
            self.rebuild_all_backlinks()
            deleted = True
        if deleted:
            self.store._invalidate_entity_content_cache()
        return deleted

    def move_entity(
        self,
        src_concept: str,
        src_class: str,
        src_name: str,
        dst_concept: str,
        dst_class: str,
        dst_name: str,
    ) -> str:
        """Move an entity file to a new location and update the natural key.

        For DIRECTORY_CONCEPTS (Symptom), moves the entire entity directory
        (including profile/).  Falls back to legacy .md path if the new
        index.md path does not exist.

        Returns the new ``{root_uri}/...`` URI.
        """
        self._invalidate_audit_cache()
        old_uri = self.store.entity_uri(src_concept, src_class or None, src_name)
        src_path = self.store.entity_path(src_concept, src_class or None, src_name)
        if not self.store.exists(self.store._key_of(src_path)):
            legacy = self.store.legacy_entity_path(src_concept, src_class or None, src_name)
            if legacy is not None and self.store.exists(self.store._key_of(legacy)):
                src_path = legacy
            else:
                raise FileNotFoundError(f"Source entity not found: {src_path}")
        dst_path = self.store.entity_path(dst_concept, dst_class or None, dst_name)
        if self.store.read_entity(dst_concept, dst_class or None, dst_name) is not None:
            raise FileExistsError(f"Destination entity already exists: {dst_path}")
        src_key = self.store._key_of(src_path)
        dst_key = self.store._key_of(dst_path)
        if src_concept in self.store.DIRECTORY_CONCEPTS:
            src_dir = src_path.parent
            dst_dir = dst_path.parent
            if src_dir == dst_dir:
                return self.store.entity_uri(dst_concept, dst_class or None, dst_name)
            # Move every .md under the directory concept's folder.
            src_prefix = self.store._key_of(src_dir)
            dst_prefix = self.store._key_of(dst_dir)
            moved_any = False
            for key in self.store.list_dir(src_prefix, recursive=True):
                self.store.move(key, f"{dst_prefix}/{Path(key).name}")
                moved_any = True
            if not moved_any:
                raise FileNotFoundError(f"Source entity not found: {src_path}")
        else:
            self.store.move(src_key, dst_key)
            src_parent_key = "/".join(src_key.split("/")[:-1])
            if src_parent_key != "/".join(dst_key.split("/")[:-1]):
                self.store.remove_empty_dir(src_parent_key)
        new_uri = self.store.entity_uri(dst_concept, dst_class or None, dst_name)
        self.store.unregister_uri(old_uri)
        self.store.register_natural_key(dst_concept, dst_class or None, dst_name, new_uri)
        self.store.register_redirect(old_uri, new_uri)
        affected = self._rewrite_persisted_uri(old_uri, new_uri)
        # ponytail: re-sync OpenViking native edges only for the pages whose
        # content changed, via SDK link/unlink. No full-library remote JSON
        # backlink rebuild — a move touches a handful of referrers.
        pairs = [
            (uri, [u for u in _parse_forward_links(self.read_resource(uri) or "") if u != uri])
            for uri in affected
        ]
        self.store.sync_native_relations(pairs)
        if self._log:
            self._log.entity_moved(src_key, dst_key, new_uri)
        logger.info("Entity moved: %s → %s (%s)", src_path, dst_path, new_uri)
        return new_uri

    def plan_entity_move(
        self,
        src_concept: str,
        src_class_name: str,
        src_object_name: str,
        dst_class_name: str,
        dst_object_name: str = "",
    ) -> dict[str, object]:
        """Read-only preview of a storage-side entity move (no writes).

        Mirrors ``move_entity`` semantics so an expert can review the
        before/after layout — entity paths and URIs, directory creation
        and pruning, redirect registration, affected backlinks — before
        confirming an OPS that carries ``candidate_operations`` with
        ``{"op": "move_entity", ...}``.
        """
        if src_concept not in self.store.CONCEPT_DIRS:
            raise ValueError(f"Unknown Wiki concept: {src_concept}")
        dst = dst_object_name.strip() or src_object_name
        old_uri = self.store.entity_uri(src_concept, src_class_name or None, src_object_name)
        src_path = self.store.entity_path(src_concept, src_class_name or None, src_object_name)
        if not self.store.exists(self.store._key_of(src_path)):
            legacy = self.store.legacy_entity_path(
                src_concept, src_class_name or None, src_object_name
            )
            if legacy is not None and self.store.exists(self.store._key_of(legacy)):
                src_path = legacy
            else:
                return {
                    "action": "error",
                    "reason": f"Source entity not found: {old_uri}",
                    "source": {"uri": old_uri, "path": str(src_path)},
                }
        new_uri = self.store.entity_uri(src_concept, dst_class_name or None, dst)
        dst_path = self.store.entity_path(src_concept, dst_class_name or None, dst)
        same_dir = src_path.parent == dst_path.parent
        dst_exists = same_dir or self.store.read_entity_by_uri(new_uri) is not None
        # ponytail: preview must be cheap — read the persisted backlink index
        # (one file read) instead of scanning the whole library.
        affected_backlinks = self.store.get_backlinks(old_uri)
        return {
            "action": "noop" if same_dir and dst == src_object_name else "move",
            "reason": "destination is the current location"
            if same_dir and dst == src_object_name
            else (
                "destination identity already exists; plan a guarded merge instead"
                if dst_exists and not same_dir
                else "storage relocation is safe to apply"
            ),
            "directory_concept": src_concept in self.store.DIRECTORY_CONCEPTS,
            "source": {
                "uri": old_uri,
                "path": str(src_path),
                "class_name": src_class_name,
                "object_name": src_object_name,
            },
            "destination": {
                "uri": new_uri,
                "path": str(dst_path),
                "class_name": dst_class_name,
                "object_name": dst,
            },
            "directory": {
                "src_dir": str(src_path.parent),
                "dst_dir": str(dst_path.parent),
                "same_dir": same_dir,
                "dst_dir_created_by_execute": not dst_path.parent.is_dir()
                if hasattr(dst_path.parent, "is_dir")
                else True,
                "src_dir_pruned_after_move": not same_dir,
            },
            "redirect": {"old_uri": old_uri, "new_uri": new_uri} if not same_dir else {},
            "affected_backlinks": affected_backlinks,
            "rollback": {"source_uri": new_uri, "target_uri": old_uri},
            "execute_required": bool(not same_dir or dst != src_object_name),
        }

    def plan_component_classification(
        self,
        component_uri: str,
        candidate_class_name: str,
        evidence_uris: list[str] | None = None,
    ) -> dict[str, object]:
        """Return a deterministic BOM-backed class move plan without writing.

        The planner only classifies a nested path change.  A cross-branch
        change is deliberately ``ambiguous`` and must become an OPA before a
        caller may invoke ``move_entity``.
        """
        identity = self.store.lookup_by_uri(component_uri)
        if identity is None or identity[0] != "Component":
            raise ValueError(f"Unknown Component URI: {component_uri}")
        _concept, current_class, object_name = identity
        current_class_name = current_class or ""
        candidate = self.store.normalize_class_name("Component", candidate_class_name.strip()) or ""
        if not candidate:
            raise ValueError("candidate_class_name must not be empty")
        evidence = list(dict.fromkeys(uri.strip() for uri in (evidence_uris or []) if uri.strip()))
        if any(not uri.startswith(self._raw_fs.root_uri + "/") for uri in evidence):
            raise ValueError("evidence_uris must contain raw chapter URIs")
        bom_record = self.get_bom_taxonomy().get(component_uri)
        bom_path = str(bom_record.get("bom_path", "")) if bom_record is not None else ""

        if candidate == current_class_name:
            action = "noop"
            reason = "candidate class is already canonical"
        elif candidate.startswith(current_class_name + "/"):
            action = "move_down"
            reason = "candidate is a deeper BOM assembly path"
        elif current_class_name.startswith(candidate + "/"):
            action = "move_up"
            reason = "candidate is an ancestor BOM assembly path"
        else:
            action = "ambiguous"
            reason = (
                "candidate crosses a BOM branch; require OPA or explicit classification decision"
            )

        target_uri = self.store.entity_uri("Component", candidate, object_name)
        destination = self.store.entity_path("Component", candidate, object_name)
        destination_exists = self.store.read_entity_by_uri(target_uri) is not None
        if action != "noop" and not evidence:
            action = "ambiguous"
            reason = "non-noop classification requires at least one BOM/raw evidence URI"
        if bom_path and candidate != bom_path and not candidate.startswith(bom_path + "/"):
            action = "ambiguous"
            reason = "candidate is outside the registered global BOM branch; require OPA"
        if destination_exists and target_uri != component_uri:
            action = "ambiguous"
            reason = "destination identity already exists; plan a guarded merge instead of a move"

        # ponytail: preview must be cheap — read the persisted backlink index.
        affected_backlinks = self.store.get_backlinks(component_uri)
        return {
            "action": action,
            "reason": reason,
            "component_uri": component_uri,
            "target_uri": target_uri,
            "current_class_name": current_class_name,
            "candidate_class_name": candidate,
            "object_name": object_name,
            "destination_exists": destination_exists,
            "directory_created_by_execute": not destination.parent.is_dir(),
            "evidence_uris": evidence,
            "bom_record": bom_record or {},
            "bom_path": bom_path,
            "affected_backlinks": affected_backlinks,
            "rollback": {"source_uri": target_uri, "target_uri": component_uri},
            "execute_required": action != "noop",
        }
