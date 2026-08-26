"""design_717 filesystem storage for wiki entities.

Backend-neutral knowledge graph store:
- Concept → Class → Object directory layout under ``wiki/``
- Atomic markdown writes via temp-file rename
- ``index/backlinks_index.json`` for reverse-link queries
- Chinese filesystem paths (``Concept/Class/Object.md``) for readability
- Readable, self-describing URIs for new entities; legacy hash URIs remain readable
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING

from wolfharness.capabilities.wiki.quality import extract_source_uris, is_source_uri_scheme

from .backend import FSBackend
from .indexes import WikiIndex
from .local_fs import LocalFS


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class WikiStore:
    """Filesystem store for design_717 wiki entities.

    One instance owns one ``root`` directory. All paths are resolved against it.
    """

    CONCEPT_DIRS: dict[str, str] = {
        "Device": "Device",
        "Component": "Component",
        "DTC": "DTC",
        "Symptom": "Symptom",
        "Fault": "Fault",
        "Procedure": "Procedure",
        "OPA": "OP/OpA",
        "OPS": "OP/OpS",
        "OPL": "OP/OpL",
    }

    # Concepts that use a directory + index.md layout instead of a flat .md file.
    # Symptom uses Symptom/<object>/index.md with a profile/ subdirectory
    # per design_717.md §2.1.
    DIRECTORY_CONCEPTS: frozenset[str] = frozenset({"Symptom"})

    # Legacy 关重件/普通件 prefixes, accepted-and-stripped for backward compat
    # only (never produced on write). Component folders now follow the logical
    # class_name path alone.
    _LEGACY_TIERS: tuple[str, ...] = ("关重件/", "普通件/")

    # Parenthetical annotations (full-width ／half-width) stripped from
    # class_name before path/hash so the same controller always merges
    # into one directory regardless of LLM uncertainty notes.
    _PAREN_RE: re.Pattern[str] = re.compile(r"[（(][^）)]*[）)]")

    _MAX_OBJECT_NAME_BYTES = 180

    @staticmethod
    def canonical_object_name(object_name: str) -> str:
        """Keep model-internal slashes out of filesystem path components.

        Manual model names often contain a slash (for example
        ``HP3V80/AV10RC``).  A slash is data in that name, not a nested wiki
        directory.  Store it as a full-width solidus so the display remains
        recognizable while path traversal protection stays strict.

        Also clips the name to a safe byte length so the resulting filename
        stays within OS limits (macOS ``NAME_MAX`` = 255 bytes, but
        ``LocalVikingFS`` enforces 200 bytes at upload). The 180-byte cap
        leaves room for the ``.md`` extension and path components.
        """
        trans_table: dict[str, str | int | None] = {"/": "／", "\\": "＼"}
        sanitized = object_name.translate(str.maketrans(trans_table))
        encoded = sanitized.encode("utf-8")
        if len(encoded) <= WikiStore._MAX_OBJECT_NAME_BYTES:
            return sanitized
        # ponytail: byte-clip from the end, trim incomplete UTF-8 tail
        return encoded[: WikiStore._MAX_OBJECT_NAME_BYTES].decode("utf-8", errors="ignore")

    def __init__(self, root: Path | FSBackend) -> None:
        """Create a store over a local directory or an ``FSBackend``.

        ``root`` may be a local ``Path`` (wrapped in :class:`LocalFS`, the
        historical-lib fallback) or an ``FSBackend`` instance directly
        (typically :class:`~wiki.storage.viking_fs.VikingFS` for remote
        OpenViking storage).
        """
        self._fs: FSBackend = root if isinstance(root, FSBackend) else LocalFS(Path(root))
        if isinstance(self._fs, LocalFS):
            self.root = self._fs.root
        else:
            # Synthetic root for URI/path computation only; real I/O routes
            # through the backend keyed by root-relative paths.
            self.root = Path(f"viking://resources/{self._fs.namespace}")  # type: ignore[attr-defined]
        self._indexes = WikiIndex(self._fs)
        self._ensure_layout()
        self._hash_scan_cache: dict[str, tuple[str, str | None, str]] | None = None
        self._physical_entity_cache: dict[
            str,
            list[tuple[str, str | None, str, str]],
        ] = {}
        self._physical_entity_cache_at: dict[str, float] = {}
        # Audit and graph traversal repeatedly resolve the same readable URI.
        # Cache the immutable content for the lifetime of this service
        # instance; every write invalidates it below.  The fingerprint also
        # detects atomic writes made by another MCP/service process, so a
        # long-lived reader cannot serve stale forward relations.
        self._entity_content_cache: dict[str, str | None] = {}
        self._entity_content_cache_fingerprints: dict[str, tuple[int, int] | None] = {}
        self._redirects_cache_content: str | None = None
        self._native_relation_sync_result: dict[str, object] | None = None

    @property
    def root_uri(self) -> str:
        """Canonical URI prefix for this store's backend."""
        return self._fs.root_uri

    def is_wiki_uri(self, uri: str) -> bool:
        """Return ``True`` if *uri* belongs to this store's backend."""
        return uri.startswith(self.root_uri + "/")

    def _invalidate_entity_content_cache(self) -> None:
        """Drop cached entity bodies after a write or migration."""
        self._entity_content_cache.clear()
        self._entity_content_cache_fingerprints.clear()

    def _invalidate_entity_discovery_cache(self) -> None:
        """Drop cached path listings after local mutations."""
        self._physical_entity_cache.clear()
        self._physical_entity_cache_at.clear()

    @staticmethod
    def _entity_discovery_cache_seconds() -> float:
        """Return the bounded cross-worker discovery-cache lifetime."""
        try:
            return max(
                0.0,
                min(60.0, float(os.environ.get("WIKI_ENTITY_DISCOVERY_CACHE_SECONDS", "1"))),
            )
        except ValueError:
            return 1.0

    def _ensure_layout(self) -> None:
        """Create required top-level directories (idempotent)."""
        dirs = [
            "Device",
            "Component",
            "DTC",
            "Symptom",
            "Fault",
            "Procedure",
            "OP/OpA",
            "OP/OpS",
            "OP/OpL",
            "index",
            ".staging",
        ]
        for d in dirs:
            self._fs.mkdir_p(d)

    # ── Entity hash (deterministic, replaces Chinese chars in URI paths) ───────

    @classmethod
    def normalize_class_name(cls, concept: str, class_name: str | None) -> str | None:
        """Return the canonical (untiered) Component class name.

        The 关重件/普通件 physical-tier routing was removed; a Component's folder
        is its logical class_name path. Legacy prefixed inputs are stripped so
        old references still resolve to the flat identity.
        """
        if not class_name:
            return class_name
        # Strip parenthetical annotations for ALL concepts so the same
        # entity always merges into one directory/URI.
        cleaned = cls._PAREN_RE.sub("", class_name).strip()
        if cleaned != class_name:
            class_name = cleaned
        if concept != "Component":
            return class_name
        for prefix in cls._LEGACY_TIERS:
            if class_name.startswith(prefix):
                return class_name.removeprefix(prefix)
        return class_name

    @classmethod
    def logical_class_name(cls, concept: str, class_name: str | None) -> str | None:
        """Return the logical Component class name for display.

        Strips a legacy ``关重件/``/``普通件/`` prefix so historical
        tiered index entries still display as their bare logical class.
        New entries are already flat (``normalize_class_name`` strips on
        write), so this is a no-op for them.
        """
        if concept == "Component" and class_name is not None:
            for prefix in cls._LEGACY_TIERS:
                if class_name.startswith(prefix):
                    return class_name.removeprefix(prefix)
        return class_name

    @classmethod
    def entity_hash(cls, concept: str, class_name: str | None, object_name: str) -> str:
        """Deterministic SHA256 hash for entity URIs.

        Always hashes ``concept + class + object`` into a 24-char hex string.
        Used in URIs (``{root_uri}/<Concept>/<hash>``) to avoid Chinese
        characters in references.  Filesystem paths remain Chinese
        (``<Concept>/<Class>/<Object>.md``) for readability and dedup.
        """
        normalized_class = cls.normalize_class_name(concept, class_name)
        canonical_object = cls.canonical_object_name(object_name)
        raw = "\x1f".join([concept, normalized_class or "", canonical_object]).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    # ── Path resolution ────────────────────────────────────────────────────

    def entity_path(self, concept: str, class_name: str | None, object_name: str) -> Path:
        """Return the filesystem path for an entity markdown file.

        Human-readable layout: ``<root>/<Concept>/<Class>/<Object>.md``.
        When ``class_name`` is None: ``<root>/<Concept>/<Object>.md``.

        Symptom uses a directory layout per design_717.md:
        ``<root>/Symptom/<Class>/<Object>/index.md``.
        """
        class_name = self.normalize_class_name(concept, class_name)
        object_name = self.canonical_object_name(object_name)
        if concept not in self.CONCEPT_DIRS:
            raise ValueError(f"Unknown Wiki concept: {concept}")
        self._safe_component(object_name, "object_name")
        if class_name is not None:
            self._safe_class(class_name)
        base = self.root / self.CONCEPT_DIRS[concept]
        if concept in self.DIRECTORY_CONCEPTS:
            if class_name:
                path = base / class_name / object_name / "index.md"
            else:
                path = base / object_name / "index.md"
        elif class_name:
            path = base / class_name / f"{object_name}.md"
        else:
            path = base / f"{object_name}.md"
        if isinstance(self._fs, LocalFS):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError("Wiki entity path escapes the configured root")
            return resolved
        return path

    def legacy_entity_path(
        self, concept: str, class_name: str | None, object_name: str
    ) -> Path | None:
        """Return the pre-directory-layout .md path for backward compat.

        Only DIRECTORY_CONCEPTS had a legacy flat .md layout.  Other concepts
        always used .md files, so their legacy path is the same as entity_path.
        """
        class_name = self.normalize_class_name(concept, class_name)
        object_name = self.canonical_object_name(object_name)
        if concept not in self.DIRECTORY_CONCEPTS:
            return None
        base = self.root / self.CONCEPT_DIRS[concept]
        return base / class_name / f"{object_name}.md" if class_name else base / f"{object_name}.md"

    @staticmethod
    def _safe_component(value: str, label: str) -> None:
        if (
            not value
            or value in {".", ".."}
            or Path(value).name != value
            or any(separator in value for separator in ("/", "\\"))
        ):
            raise ValueError(f"Invalid {label}: {value!r}")

    @classmethod
    def _safe_profile_id(cls, value: str) -> None:
        """Reject encoded path semantics while leaving entity names intact."""
        cls._safe_component(value, "profile_id")
        segments = re.split(r"[／＼]", value)
        reserved = {"profile", "index.md", ".", ".."}
        if (
            any(not segment for segment in segments)
            or any(segment.casefold() in reserved for segment in segments)
            or len(segments) > 1
        ):
            raise ValueError(f"Invalid profile_id path shape: {value!r}")

    @staticmethod
    def _safe_class(value: str) -> None:
        """Validate class_name — allows ``/`` for nested directories (e.g. ``发动机/电控发动机/EGR``)."""
        if not value or "\\" in value:
            raise ValueError(f"Invalid class_name: {value!r}")
        for seg in value.split("/"):
            if not seg or seg in {".", ".."}:
                raise ValueError(f"Invalid class_name segment: {seg!r} in {value!r}")

    def entity_uri(self, concept: str, class_name: str | None, object_name: str) -> str:
        """Build the canonical readable ``{root_uri}/<Concept>/<Class>/<Object>`` URI.

        Readable URIs encode the entity's own class/object identity, so a
        reference is self-describing — it cannot be a "phantom hash" that
        fails to resolve.  Resolution back to a file path is direct (see
        ``_parse_readable_uri`` + ``entity_path``); no reverse index needed.

        The class segment is the LOGICAL class name (Component storage tiers
        are hidden); ``entity_path`` re-applies tier routing on resolution.
        """
        clz = self.normalize_class_name(concept, class_name)
        clz = self.logical_class_name(concept, clz) if clz else None
        object_name = self.canonical_object_name(object_name)
        self._safe_component(object_name, "object_name")
        if clz is not None:
            self._safe_class(clz)
        if clz:
            return f"{self.root_uri}/{concept}/{clz}/{object_name}"
        return f"{self.root_uri}/{concept}/{object_name}"

    def symptom_profile_uri(self, symptom_uri: str, profile_id: str) -> str:
        """Build a stable URI for a profile belonging to a Symptom entity."""
        self.symptom_profile_path(symptom_uri, profile_id)
        return f"{symptom_uri}/profile/{profile_id}"

    def symptom_profile_path(self, symptom_uri: str, profile_id: str) -> Path:
        """Return the profile path below an existing canonical Symptom."""
        self._safe_profile_id(profile_id)
        info = self.lookup_by_uri(symptom_uri)
        if info is None or info[0] != "Symptom":
            raise ValueError(f"Unknown canonical Symptom URI: {symptom_uri}")
        concept, class_name, object_name = info
        parent = self.entity_path(concept, class_name, object_name).parent
        path = parent / "profile" / f"{profile_id}.md"
        if isinstance(self._fs, LocalFS):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError("Symptom profile path escapes the configured root")
            return resolved
        return path

    def split_symptom_profile_uri(self, uri: str) -> tuple[str, str] | None:
        """Split ``<symptom_uri>/profile/<profile_id>`` into its parts.

        Supports both legacy hash parents and readable parents.
        """
        marker = "/profile/"
        if marker not in uri:
            return None
        symptom_uri, profile_id = uri.rsplit(marker, 1)
        if not symptom_uri.startswith(f"{self.root_uri}/Symptom/"):
            return None
        try:
            self._safe_profile_id(profile_id)
        except ValueError:
            return None
        parent = self.lookup_by_uri(symptom_uri)
        if parent is None or parent[0] != "Symptom":
            return None
        return symptom_uri, profile_id

    # ── Atomic read / write ────────────────────────────────────────────────

    def _key_of(self, path: Path) -> str:
        """Convert a root-anchored path (or key) to a backend-relative key."""
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            rel = Path(str(path))
        parts = [p for p in rel.parts if p not in {"", "."}]
        return "/".join(parts)

    def read_text(self, key: str) -> str | None:
        """Read a root-relative key through the active backend."""
        return self._fs.read_text(key)

    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        """Write a root-relative key through the active backend."""
        self._fs.write_text(key, content, overwrite=overwrite)

    def write_text_durable(self, key: str, content: str) -> None:
        """Write and poll until read-after-write is visible (Viking eventual consistency)."""
        self._fs.write_text_durable(key, content)

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        """Atomically publish related root-relative text records."""
        self._fs.commit_many(writes)
        self._invalidate_entity_discovery_cache()
        self._invalidate_entity_content_cache()

    def exists(self, key: str) -> bool:
        """Return whether a root-relative key exists as a file."""
        return self._fs.exists(key)

    def is_dir(self, key: str) -> bool:
        """Return whether a root-relative key exists as a directory."""
        return self._fs.is_dir(key)

    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        """List root-relative file keys under a directory key."""
        return self._fs.list_dir(key, recursive=recursive)

    def scan_frontmatter(
        self,
        key: str,
        *,
        keys: tuple[str, ...] | None = None,
        node_limit: int = 20000,
    ) -> list[tuple[str, str]]:
        """Scan single-line frontmatter fields under ``key`` in one round-trip.

        Delegates to the backend; remote (Viking) backends use an SDK ``grep``
        fast path, local backends return ``[]`` (callers fall back to the
        sequential read path).
        """
        scan = getattr(self._fs, "scan_frontmatter", None)
        if scan is None:
            return []
        result = scan(key, keys=keys, node_limit=node_limit)
        return list(result) if isinstance(result, list) else []

    def fingerprint(self, key: str) -> tuple[int | None, int | None]:
        """Return a cheap backend-native freshness marker for one file key."""
        return self._fs.fingerprint(key)

    def mkdir_p(self, key: str) -> None:
        """Ensure a root-relative directory key exists."""
        self._fs.mkdir_p(key)

    def delete(self, key: str) -> None:
        """Delete a root-relative file key."""
        self._fs.delete(key)
        self._invalidate_entity_discovery_cache()
        self._invalidate_entity_content_cache()

    def move(self, src: str, dst: str) -> None:
        """Move a root-relative file key."""
        self._fs.move(src, dst)
        self._invalidate_entity_discovery_cache()
        self._invalidate_entity_content_cache()

    def remove_empty_dir(self, key: str) -> bool:
        """Remove an empty directory key; returns ``True`` if removed."""
        removed = bool(self._fs.remove_empty_dir(key))
        if removed:
            self._invalidate_entity_discovery_cache()
            self._invalidate_entity_content_cache()
        return removed

    def read_json(self, key: str) -> dict[str, object] | None:
        """Read and parse a root-relative JSON key; ``None`` if absent/broken."""
        raw = self._fs.read_text(key)
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning("Ignoring malformed wiki JSON: %s", key)
            return None
        return value if isinstance(value, dict) else None

    def write_json(
        self,
        key: str,
        data: dict[str, object],
        *,
        durable: bool = False,
    ) -> None:
        """Serialize and write a root-relative JSON key atomically.

        ``durable=True`` is reserved for control-plane receipts whose caller
        immediately makes a scheduling decision from a separate read.
        """
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        if durable:
            self._fs.write_text_durable(key, content)
        else:
            self._fs.write_text(key, content)

    def write_text_atomic(self, path: Path, content: str) -> Path:
        """Write UTF-8 text atomically through the active backend."""
        self._fs.write_text(self._key_of(path), content)
        return path

    def write_entity(
        self, concept: str, class_name: str | None, object_name: str, content: str
    ) -> Path:
        """Atomically write an entity markdown file.

        For DIRECTORY_CONCEPTS, also removes the legacy flat .md file if it
        exists (migration from old layout to index.md layout).
        """
        path = self.entity_path(concept, class_name, object_name)
        self.write_text_atomic(path, content)
        self._invalidate_entity_content_cache()
        self._hash_scan_cache = None
        self._remember_entity(concept, class_name, object_name)
        legacy = self.legacy_entity_path(concept, class_name, object_name)
        if legacy is not None and self._fs.exists(self._key_of(legacy)):
            self._fs.delete(self._key_of(legacy))
        return path

    def write_entities(self, entities: list[tuple[str, str | None, str, str]]) -> list[Path]:
        """Write validated entity bodies through one backend batch.

        Identity and validation remain the service layer's responsibility;
        this method only turns readable entity identities into independent
        file writes and maintains local discovery caches.
        """
        if not entities:
            return []
        paths = [
            self.entity_path(concept, class_name, object_name)
            for concept, class_name, object_name, _content in entities
        ]
        self._fs.write_many(
            [
                (self._key_of(path), content)
                for path, (_concept, _class_name, _object_name, content) in zip(
                    paths, entities, strict=True
                )
            ],
        )
        self._invalidate_entity_content_cache()
        self._hash_scan_cache = None
        for concept, class_name, object_name, _content in entities:
            self._remember_entity(concept, class_name, object_name)
            legacy = self.legacy_entity_path(concept, class_name, object_name)
            if legacy is not None and self._fs.exists(self._key_of(legacy)):
                self._fs.delete(self._key_of(legacy))
        return paths

    def _remember_entity(self, concept: str, class_name: str | None, object_name: str) -> None:
        """Update an already-populated discovery cache without a tree rescan."""
        if not self._physical_entity_cache:
            return
        normalized_class = self.normalize_class_name(concept, class_name)
        canonical_object = self.canonical_object_name(object_name)
        identity = (concept, normalized_class, canonical_object)
        entity = (*identity, self.entity_uri(concept, normalized_class, canonical_object))
        for cache_key in ("*", concept):
            cached = self._physical_entity_cache.get(cache_key)
            if cached is None:
                continue
            self._physical_entity_cache[cache_key] = [
                item for item in cached if item[:3] != identity
            ]
            self._physical_entity_cache[cache_key].append(entity)
            self._physical_entity_cache_at[cache_key] = time.monotonic()

    def read_entity(self, concept: str, class_name: str | None, object_name: str) -> str | None:
        """Read an entity's content; return ``None`` if absent.

        For DIRECTORY_CONCEPTS, falls back to the legacy flat .md path if
        the new index.md path does not exist (backward compat). For Component,
        falls back to legacy 关重件/普通件 storage tiers if the flat path
        doesn't exist (removed tier routing, transient pre-migration data).
        """
        path = self.entity_path(concept, class_name, object_name)
        key = self._key_of(path)
        content = self._fs.read_text(key)
        if content is not None:
            return content
        legacy = self.legacy_entity_path(concept, class_name, object_name)
        if legacy is not None:
            content = self._fs.read_text(self._key_of(legacy))
            if content is not None:
                return content
        if concept == "Component" and class_name:
            for prefix in self._LEGACY_TIERS:
                tiered = (
                    self.root
                    / self.CONCEPT_DIRS["Component"]
                    / f"{prefix}{class_name}"
                    / f"{object_name}.md"
                )
                content = self._fs.read_text(self._key_of(tiered))
                if content is not None:
                    return content
        return None

    def read_entity_by_uri(self, uri: str) -> str | None:
        """Read a readable entity URI directly or resolve a legacy hash URI."""
        profile = self.split_symptom_profile_uri(uri)
        if profile is not None:
            symptom_uri, profile_id = profile
            try:
                path = self.symptom_profile_path(symptom_uri, profile_id)
            except ValueError:
                return None
        else:
            info = self.lookup_by_uri(uri)
            if info is None:
                self._entity_content_cache[uri] = None
                self._entity_content_cache_fingerprints[uri] = None
                return None
            concept, class_name, object_name = info
            path = self.entity_path(concept, class_name, object_name)
            key = self._key_of(path)
            if self._fs.read_text(key) is None:
                legacy = self.legacy_entity_path(concept, class_name, object_name)
                if legacy is not None and self._fs.read_text(self._key_of(legacy)) is not None:
                    path = legacy
                elif concept == "Component" and class_name:
                    for prefix in self._LEGACY_TIERS:
                        tiered = (
                            self.root
                            / self.CONCEPT_DIRS["Component"]
                            / f"{prefix}{class_name}"
                            / f"{object_name}.md"
                        )
                        if self._fs.read_text(self._key_of(tiered)) is not None:
                            path = tiered
                            break

        key = self._key_of(path)
        fingerprint: tuple[int, int] | None
        mtime = self._fs.mtime_ns(key)
        size = self._fs.size(key)
        fingerprint = (mtime, size) if mtime is not None and size is not None else None
        if (
            uri in self._entity_content_cache
            and self._entity_content_cache_fingerprints.get(uri) == fingerprint
        ):
            return self._entity_content_cache[uri]
        content = self._fs.read_text(key)
        self._entity_content_cache[uri] = content
        self._entity_content_cache_fingerprints[uri] = fingerprint
        return content

    def write_symptom_profile(self, symptom_uri: str, profile_id: str, content: str) -> Path:
        """Atomically write a Symptom Profile below its canonical Symptom."""
        path = self.symptom_profile_path(symptom_uri, profile_id)
        written = self.write_text_atomic(path, content)
        self._invalidate_entity_content_cache()
        return written

    def list_symptom_profiles(self, symptom_uri: str) -> list[tuple[str, str]]:
        """Return ``[(profile_id, profile_uri), ...]`` for a canonical Symptom."""
        info = self.lookup_by_uri(symptom_uri)
        if info is None or info[0] != "Symptom":
            return []
        concept, class_name, object_name = info
        profile_dir = self.entity_path(concept, class_name, object_name).parent / "profile"
        keys = self._fs.list_dir(self._key_of(profile_dir))
        return [
            (Path(key).stem, f"{symptom_uri}/profile/{Path(key).stem}")
            for key in keys
            if key.endswith(".md") and Path(key).name != "index.md"
        ]

    def lookup_by_uri(self, uri: str) -> tuple[str, str | None, str] | None:
        """Resolve a wiki URI to ``(concept, class_name, object_name)``.

        Accepts two forms:
        - Readable (canonical): ``{root_uri}/<Concept>/<Class>/<Object>`` —
          parsed positionally, no index involved.
        - Legacy hash form: ``{root_uri}/<Concept>/<hash>`` — resolved through
          a read-only historical mapping or a one-time page scan.
        """
        uri = self.resolve_redirect(uri)
        readable = self._parse_readable_uri(uri)
        if readable is not None:
            return readable
        info = self._indexes.lookup_uri(uri)
        if info is not None:
            return info
        return self._scan_for_hash_uri(uri)

    def _parse_readable_uri(
        self,
        uri: str,
    ) -> tuple[str, str | None, str] | None:
        """Parse a readable ``{root_uri}/<Concept>/<Class>/<Object>`` URI.

        Returns ``None`` when the URI is not a readable-form wiki URI
        (e.g. a hash URI, a malformed path, or a non-wiki scheme).

        Args:
            uri: The URI string to parse.

        Returns:
            ``(concept, class_name, object_name)`` when parseable.
        """
        prefix = self.root_uri + "/"
        if not uri.startswith(prefix):
            return None
        parts = uri[len(prefix) :].split("/")
        if not parts or parts[0] not in self.CONCEPT_DIRS:
            return None
        concept = parts[0]
        object_name = parts[-1].removesuffix(".md") if parts[-1].endswith(".md") else parts[-1]
        class_name = "/".join(parts[1:-1]) or None
        # A hash-form tail (single 24-hex segment) is legacy, not readable.
        if class_name is None and re.fullmatch(r"[0-9a-f]{24}", object_name):
            return None
        # Component class is normalized (untiered) — its folder is the logical
        # class_name path; legacy 关重件/普通件 prefixes are stripped.
        return concept, self.normalize_class_name(concept, class_name), object_name

    def _scan_for_hash_uri(self, uri: str) -> tuple[str, str | None, str] | None:
        """File-system fallback: scan .md files for a matching first-line hash URI.

        Reads only the first line of each ``.md`` file (``# {root_uri}/...``)
        and compares against *uri*. Triggered only for a legacy hash URI that
        was absent from the historical mapping.

        The scan is executed at most once per process; the resulting
        ``hash -> identity`` map is cached so repeated misses are O(1)
        instead of re-walking every ``.md`` file (which made full-library
        audits quadratic in the number of phantom references).
        """
        if not self.is_wiki_uri(uri):
            return None
        if self._hash_scan_cache is None:
            cache: dict[str, tuple[str, str | None, str]] = {}
            for concept, subdir in self.CONCEPT_DIRS.items():
                keys = self._fs.list_dir(subdir, recursive=True)
                for key in keys:
                    if not key.endswith(".md"):
                        continue
                    try:
                        content = self._fs.read_text(key)
                    except (OSError, UnicodeDecodeError):
                        continue
                    if content is None:
                        continue
                    first_line = content.splitlines()[0].strip() if content else ""
                    if not self.is_wiki_uri(first_line):
                        continue
                    rel = Path(key).relative_to(Path(subdir))
                    parts = rel.parts
                    if Path(key).name == "index.md" and concept in self.DIRECTORY_CONCEPTS:
                        object_name = parts[-2]
                        class_name = "/".join(parts[:-2]) or None
                    else:
                        object_name = Path(key).stem
                        class_name = "/".join(parts[:-1]) or None
                    cache[first_line] = (concept, class_name, object_name)
            self._hash_scan_cache = cache
        return self._hash_scan_cache.get(uri)

    # ── Natural-key registry ───────────────────────────────────────────────

    def ensure_natural_key_index(self) -> None:
        """Compatibility no-op; readable entity paths are the identity map."""

    def register_case_uri(self, display_uri: str) -> str:
        """Return the validated readable case URI without a mutable alias map."""
        return display_uri

    def resolve_case_uri(self, uri: str) -> str | None:
        """Return a readable case URI or resolve a historical hash alias."""
        prefix = self.root_uri + "/case/"
        if uri.startswith(prefix):
            return uri
        return self._indexes.resolve_case_uri(uri)

    def register_natural_key(
        self, concept: str, class_name: str | None, natural_key: str, uri: str
    ) -> None:
        """Validate identity without writing a global natural-key mapping."""
        class_name = self.normalize_class_name(concept, class_name)
        natural_key = self.canonical_object_name(natural_key)
        expected = self.entity_uri(concept, class_name, natural_key)
        if uri != expected:
            raise ValueError(f"Entity URI does not match its readable identity: {uri}")

    def update_natural_key(
        self, concept: str, class_name: str | None, natural_key: str, uri: str
    ) -> None:
        """Update (overwrite) a natural-key → URI mapping.

        New readable URIs need no mutable reverse index. Moves remain
        addressable through the redirect layer.
        """
        class_name = self.normalize_class_name(concept, class_name)
        natural_key = self.canonical_object_name(natural_key)
        expected = self.entity_uri(concept, class_name, natural_key)
        if uri != expected:
            raise ValueError(f"Entity URI does not match its readable identity: {uri}")

    def unregister_uri(self, uri: str) -> bool:
        """Invalidate discovery state; readable identities have no registry row."""
        existed = self.lookup_by_uri(uri) is not None
        self._invalidate_entity_discovery_cache()
        return existed

    def lookup_natural_key(
        self, concept: str, class_name: str | None, natural_key: str
    ) -> str | None:
        """Resolve a natural key directly through the entity's readable path."""
        class_name = self.normalize_class_name(concept, class_name)
        natural_key = self.canonical_object_name(natural_key)
        if self.read_entity(concept, class_name, natural_key) is None:
            return None
        return self.entity_uri(concept, class_name, natural_key)

    def fuzzy_lookup_natural_key(
        self, concept: str, class_name: str | None, object_name: str
    ) -> str | None:
        """Fuzzy-match discovered entities without a global JSON map."""
        class_name = self.normalize_class_name(concept, class_name)
        object_name = self.canonical_object_name(object_name)
        candidates = self.list_entities(concept)
        if concept == "Component":
            for _concept, candidate_class, candidate_name, uri in candidates:
                if (
                    class_name
                    and candidate_class
                    and not (
                        candidate_class == class_name or candidate_class.startswith(class_name)
                    )
                ):
                    continue
                if (
                    (candidate_name.endswith(object_name) or object_name.endswith(candidate_name))
                    and len(candidate_name) >= 3
                    and len(object_name) >= 3
                ):
                    return uri
        elif concept == "Device":
            for _concept, _candidate_class, candidate_name, uri in candidates:
                if candidate_name.startswith(object_name) or object_name.startswith(candidate_name):
                    return uri
        elif concept == "DTC":

            def normalize_code(code: str) -> str:
                match = re.match(r"^([A-Z])-?(\d+)", code)
                return f"{match.group(1)}{int(match.group(2)):03d}" if match else code

            normalized = normalize_code(object_name)
            for _concept, candidate_class, candidate_name, uri in candidates:
                if class_name and candidate_class != class_name:
                    continue
                candidate_normalized = normalize_code(candidate_name)
                if candidate_normalized.startswith(normalized) or normalized.startswith(
                    candidate_normalized
                ):
                    return uri
        return None

    def list_entities(
        self,
        concept: str | None = None,
    ) -> list[tuple[str, str | None, str, str]]:
        """Enumerate entities from their physical self-describing paths."""
        return self.physical_entities(concept)

    def physical_entities(
        self,
        concept: str | None = None,
    ) -> list[tuple[str, str | None, str, str]]:
        """Enumerate the self-describing entity files from backend storage."""
        cache_key = concept or "*"
        cached = self._physical_entity_cache.get(cache_key)
        cached_at = self._physical_entity_cache_at.get(cache_key, 0.0)
        if (
            cached is not None
            and time.monotonic() - cached_at <= self._entity_discovery_cache_seconds()
        ):
            return list(cached)
        concepts = (
            [concept]
            if concept
            else [name for name in self.CONCEPT_DIRS if name not in {"OPA", "OPS", "OPL"}]
        )
        found: list[tuple[str, str | None, str, str]] = []
        for name in concepts:
            if name not in self.CONCEPT_DIRS:
                continue
            subdir = self.CONCEPT_DIRS[name]
            keys = self._fs.list_dir(subdir, recursive=True)
            for key in keys:
                if not key.endswith(".md"):
                    continue
                rel = Path(key).relative_to(Path(subdir))
                parts = rel.parts
                if name == "Symptom":
                    if Path(key).name != "index.md" or len(parts) < 2:
                        continue
                    object_name = parts[-2]
                    class_name = "/".join(parts[:-2]) or None
                else:
                    object_name = Path(key).stem
                    class_name = "/".join(parts[:-1]) or None
                uri = self.entity_uri(name, class_name, object_name)
                found.append((name, class_name, object_name, uri))
        self._physical_entity_cache[cache_key] = found
        self._physical_entity_cache_at[cache_key] = time.monotonic()
        if concept is None:
            for concept_name in concepts:
                self._physical_entity_cache[concept_name] = [
                    entity for entity in found if entity[0] == concept_name
                ]
                self._physical_entity_cache_at[concept_name] = time.monotonic()
        return found

    # ── Class-level hash URI (browse without Chinese in URI) ──────────────

    def class_uri(self, concept: str, class_name: str) -> str:
        """Build a readable class URI directly from the directory identity."""
        class_name = self.normalize_class_name(concept, class_name) or class_name
        if class_name != "(flat)":
            self._safe_class(class_name)
        suffix = "" if class_name == "(flat)" else f"/{class_name}"
        return f"{self.root_uri}/{concept}/@class{suffix}"

    def lookup_class_by_uri(self, uri: str) -> tuple[str, str] | None:
        """Resolve a readable class URI or a historical class hash URI.

        Returns ``None`` if no entity currently establishes that class.
        """
        prefix = self.root_uri + "/"
        if not uri.startswith(prefix):
            return None
        parts = uri[len(prefix) :].strip("/").split("/")
        if len(parts) < 2:
            return None
        concept = parts[0]
        classes = {
            class_name or "(flat)"
            for entity_concept, class_name, _object_name, _entity_uri in self.list_entities(concept)
            if entity_concept == concept
        }
        if parts[1] == "@class":
            readable_class = "/".join(parts[2:]) or "(flat)"
            if readable_class in classes:
                return concept, readable_class
            return None
        # Rolling compatibility for old two-segment class hash URIs.
        return self._indexes.lookup_class_uri(uri)

    # ── Body reference resolution ──────────────────────────────────────────

    def resolve_body_refs(
        self,
        content: str,
        is_existing_external_uri: Callable[[str], bool] | None = None,
    ) -> str:
        """Convert human-readable ``{root_uri}/...`` URIs to hash URIs when possible.

        LLM-generated content contains references like
        ``{root_uri}/Component/散热器/散热器总成``.  When the target entity is
        already registered, the reference is converted to its canonical hash
        URI. When the target is **not yet registered**, the natural-key URI is
        preserved as a deliberate build-time dangling forward reference.
        """

        def _resolve(uri: str) -> str:
            if is_existing_external_uri is not None and is_existing_external_uri(uri):
                return uri
            prefix = self.root_uri + "/"
            if not uri.startswith(prefix):
                return uri
            parts = uri.removeprefix(prefix).split("/")
            if len(parts) < 2:
                return uri  # malformed
            concept = parts[0]
            # OP sidecars (OPA/OPS/OPL) are review artifacts, not entity
            # concepts. Their URIs are already canonical and must not be sent
            # through the entity natural-key resolver.
            if concept not in {"Device", "Component", "DTC", "Symptom", "Fault", "Procedure"}:
                return uri
            object_or_hash = parts[-1]
            if (
                len(parts) == 2
                and len(object_or_hash) == 24
                and all(character in "0123456789abcdef" for character in object_or_hash)
            ):
                return uri
            object_name = parts[-1].removesuffix(".md") if parts[-1].endswith(".md") else parts[-1]
            class_name = "/".join(parts[1:-1]) or None
            resolved = self.lookup_natural_key(concept, class_name, object_name)
            return resolved if resolved else uri

        for uri in sorted(
            (
                candidate
                for candidate in extract_source_uris(content)
                if self.is_wiki_uri(candidate)
            ),
            key=len,
            reverse=True,
        ):
            content = content.replace(uri, _resolve(uri))
        return content

    _INLINE_CITATION_RE = re.compile(r"\[([a-z][a-z0-9+.\-]*://[^\]\s]+)\]")
    _SECTION_HEADER_RE = re.compile(r"^(#{2,3}\s+.+)$", re.MULTILINE)

    # YAML frontmatter repair patterns
    # Detect a source URI concatenated with the next YAML field:
    #   "  - viking://.../ch0285title: ..." → "  - viking://.../ch0285\ntitle: ..."
    _FM_CONCAT_RE = re.compile(
        r"^(\s+- [a-z][a-z0-9+.\-]*://\S+?/ch\d+)([a-z_]+:.*)$",
        re.MULTILINE,
    )
    # Detect [简述][source-uri] format in source lines — extract the URI
    _FM_DESC_SOURCE_RE = re.compile(
        r"^  - \[[^\]]+\]\[([a-z][a-z0-9+.\-]*://[^\]]+)\]$", re.MULTILINE
    )
    # YAML list fields that contain source URIs needing hash resolution
    _FM_URI_FIELDS = frozenset(
        {
            "forward_links",
            "related_devices",
            "related_faults",
            "related_symptoms",
            "related_dtc",
            "related_components",
            "affected_components",
            "critical_components",
            "symptom_refs",
            "device_refs",
            "direct_component_uri",
            "applicable_models",
            "assembly_parts",
            "verification_procedures",
            "repair_procedures",
            "target_components",
            "specification_refs",
            "controller_component",
            "possible_faults",
        },
    )

    def dedup_citations(self, content: str) -> str:
        """Remove duplicate ``[source-uri]`` citations within each section.

        Keeps only the first occurrence of each URI inside each ``##`` or
        ``###`` section.  Frontmatter (between ``---`` fences) is untouched.
        """
        fm_end = content.find(
            "\n---\n", content.find("---\n") + 4 if content.startswith("---\n") else 0
        )
        if fm_end == -1:
            fm_part = ""
            body = content
        else:
            fm_end += 4
            fm_part = content[:fm_end]
            body = content[fm_end:]

        sections: list[str] = []
        current_header = ""
        current_lines: list[str] = []

        for line in body.splitlines(keepends=True):
            if self._SECTION_HEADER_RE.match(line):
                if current_header or current_lines:
                    sections.append(current_header + "".join(current_lines))
                current_header = line
                current_lines = []
            else:
                current_lines.append(line)
        if current_header or current_lines:
            sections.append(current_header + "".join(current_lines))

        result = fm_part
        for section in sections:
            seen: set[str] = set()

            def _dedup(m: re.Match[str], seen_uris: set[str] = seen) -> str:
                uri = m.group(1)
                if uri in seen_uris:
                    return ""
                seen_uris.add(uri)
                return m.group(0)

            result += self._INLINE_CITATION_RE.sub(_dedup, section)

        return result

    def repair_frontmatter(
        self,
        content: str,
        is_existing_external_uri: Callable[[str], bool] | None = None,
    ) -> str:
        r"""Repair LLM-generated YAML frontmatter.

        Fixes three systematic issues:
        1. **Concatenation bug**: last source URI merged with next YAML field
           (``ch0285title: ...`` → ``ch0285\\ntitle: ...``)
        2. **[简述] prefix**: ``[描述][source-uri]`` → bare ``source-uri``
        3. **Duplicate sources**: same URI appears multiple times
        4. **Path-based URIs**: resolves ``{root_uri}/Component/...`` to hash URIs
           in frontmatter list fields (``forward_links``, ``related_*``).
        """
        fm_start = content.find("---\n")
        if fm_start == -1:
            return content
        fm_end = content.find("\n---\n", fm_start + 4)
        if fm_end == -1:
            return content

        fm_end += 4
        fm_part = content[:fm_end]
        body = content[fm_end:]

        # 1. Fix concatenation: split source URI from next YAML field
        fm_part = self._FM_CONCAT_RE.sub(r"\1\n\2", fm_part)

        # 2. Strip [简述] prefix: [描述][source-uri] → source-uri
        fm_part = self._FM_DESC_SOURCE_RE.sub(r"  - \1", fm_part)

        # 3. Deduplicate sources
        fm_part = self._dedup_sources(fm_part)

        # 4. Resolve path-based URIs in frontmatter list fields
        fm_part = self._resolve_fm_uris(fm_part, is_existing_external_uri)

        return fm_part + body

    def _dedup_sources(self, fm: str) -> str:
        """Remove duplicate URIs from the ``sources:`` list in frontmatter."""
        lines = fm.splitlines(keepends=True)
        result: list[str] = []
        in_sources = False
        seen_uris: set[str] = set()

        for line in lines:
            stripped = line.rstrip("\n")
            if stripped == "sources:":
                in_sources = True
                result.append(line)
                seen_uris = set()
                continue
            if in_sources:
                if stripped.startswith("  - ") and is_source_uri_scheme(stripped[4:].strip()):
                    uri = stripped[4:].strip()
                    if uri in seen_uris:
                        continue
                    seen_uris.add(uri)
                    result.append(line)
                    continue
                in_sources = False
            result.append(line)

        return "".join(result)

    def _resolve_fm_uris(
        self,
        fm: str,
        is_existing_external_uri: Callable[[str], bool] | None,
    ) -> str:
        """Resolve path-based wiki URIs to canonical URIs in frontmatter."""

        def _resolve_uri(uri: str) -> str:
            if is_existing_external_uri is not None and is_existing_external_uri(uri):
                return uri
            prefix = self.root_uri + "/"
            if not uri.startswith(prefix):
                return uri
            parts = uri.removeprefix(prefix).split("/")
            if len(parts) < 2:
                return uri
            object_or_hash = parts[-1]
            if (
                len(parts) == 2
                and len(object_or_hash) == 24
                and all(c in "0123456789abcdef" for c in object_or_hash)
            ):
                return uri
            concept = parts[0]
            object_name = parts[-1].removesuffix(".md") if parts[-1].endswith(".md") else parts[-1]
            class_name = "/".join(parts[1:-1]) or None
            resolved = self.lookup_natural_key(concept, class_name, object_name)
            if resolved is None:
                resolved = self.fuzzy_lookup_natural_key(concept, class_name, object_name)
            return resolved if resolved else uri

        lines = fm.splitlines(keepends=True)
        result: list[str] = []
        current_field = ""

        for source_line in lines:
            stripped = source_line.rstrip("\n")
            field_match = re.match(r"^(\w+):", stripped)
            if field_match:
                current_field = field_match.group(1)

            if current_field in self._FM_URI_FIELDS:
                rewritten_line = source_line
                for uri in sorted(
                    (
                        candidate
                        for candidate in extract_source_uris(source_line)
                        if self.is_wiki_uri(candidate)
                    ),
                    key=len,
                    reverse=True,
                ):
                    rewritten_line = rewritten_line.replace(uri, _resolve_uri(uri))
                result.append(rewritten_line)
            else:
                result.append(source_line)

        return "".join(result)

    # ── Backlinks index ────────────────────────────────────────────────────

    def _backlinks_key(self) -> str:
        return "index/backlinks_index.json"

    def _redirects_key(self) -> str:
        return "index/redirects.json"

    def register_redirect(self, old_uri: str, new_uri: str) -> None:
        """Persist an old URI redirect so moves do not break callers."""
        if not old_uri or not new_uri or old_uri == new_uri:
            return
        key = self._redirects_key()
        redirects: dict[str, str] = {}
        raw = self._fs.read_text(key)
        if raw is not None:
            data = json.loads(raw)
            if isinstance(data, dict):
                redirects = {str(k): str(v) for k, v in data.items()}
        redirects[old_uri] = new_uri
        self._redirects_cached = False
        self.write_text_atomic(
            Path(key), json.dumps(redirects, ensure_ascii=False, indent=2) + "\n"
        )

    def _redirects_cache(self) -> str | None:
        """Read the redirects table once and cache it for the instance.

        Redirect writes are rare (URI moves) and invalidate the cache
        explicitly; a per-instance cache turns the O(OPA-count) remote
        reads inside ``resolve_redirect`` into one read.
        """
        if not getattr(self, "_redirects_cached", False):
            self._redirects_cache_content = self._fs.read_text(self._redirects_key())
            self._redirects_cached = True
        return self._redirects_cache_content

    def resolve_redirect(self, uri: str) -> str:
        """Follow persisted redirects with a cycle limit."""
        raw = self._redirects_cache()
        if raw is None:
            return uri
        data = json.loads(raw)
        redirects = data if isinstance(data, dict) else {}
        current = uri
        seen: set[str] = set()
        for _ in range(16):
            if current in seen:
                raise ValueError(f"URI redirect cycle detected at {current}")
            seen.add(current)
            target = redirects.get(current)
            if not isinstance(target, str) or not target:
                return current
            current = target
        raise ValueError(f"URI redirect chain is too long for {uri}")

    def rebuild_backlinks(self, entity_uris_with_links: list[tuple[str, list[str]]]) -> None:
        """Rebuild ``backlinks_index.json`` from scratch per build pass.

        Also reconciles backend-native relation edges (e.g. OpenViking
        ``link``) to the same authoritative mapping.  Returns nothing;
        native sync outcomes are reported via backend logs.
        """
        backlinks: dict[str, list[str]] = {}
        for entity_uri, linked_uris in entity_uris_with_links:
            backlinks.setdefault(entity_uri, [])
            for linked_uri in linked_uris:
                refs = backlinks.setdefault(linked_uri, [])
                if entity_uri not in refs:
                    refs.append(entity_uri)
        self._fs.write_text_durable(
            self._backlinks_key(),
            json.dumps(backlinks, ensure_ascii=False, indent=2) + "\n",
        )
        self._native_relation_sync_result = self._fs.sync_native_relations(entity_uris_with_links)

    def sync_native_relations(
        self, entity_uris_with_links: list[tuple[str, list[str]]]
    ) -> dict[str, object]:
        """Reconcile backend-native edges (OpenViking ``link``/``unlink``) only.

        Unlike ``rebuild_backlinks`` this skips the ``backlinks_index.json``
        artifact and syncs just the native graph for the given pages.
        """
        self._native_relation_sync_result = self._fs.sync_native_relations(entity_uris_with_links)
        return self._native_relation_sync_result

    @property
    def native_relation_sync_result(self) -> dict[str, object] | None:
        """Return the latest native relation sync result for finalize gating."""
        return self._native_relation_sync_result

    def get_backlinks(self, uri: str) -> list[str]:
        """Return URIs that link to *uri* (empty if none / no index)."""
        raw = self._fs.read_text(self._backlinks_key())
        if raw is None:
            return []
        index: dict[str, list[str]] = json.loads(raw)
        return index.get(uri, [])
