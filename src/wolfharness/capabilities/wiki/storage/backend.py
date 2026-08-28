"""Storage backends for :class:`~wiki.storage.storage.WikiStore`.

A backend abstracts every filesystem primitive that ``WikiStore`` / ``WikiIndex``
depend on behind one interface, so the same knowledge-graph store can live on a
local directory (``LocalFS``, read-only fallback for historical libs) or on a
remote OpenViking instance (``VikingFS``, the primary read/write path).

Path keys are backend-relative and use ``/`` separators only. The entity layout
is identical in both backends (``Concept/Class/Object.md``); operational
metadata may use independent JSON records, but entity identity does not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re


logger = logging.getLogger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_control_chars(text: str) -> str:
    r"""Remove JSON-illegal control characters (preserving \\n, \\r, \\t)."""
    return _CONTROL_CHAR_RE.sub("", text)


class FSBackend(ABC):
    """Minimal filesystem surface used by ``WikiStore`` / ``WikiIndex``.

    All methods take path-keys (relative, POSIX separators, no leading ``/``).
    Write semantics are atomic/replace: a write leaves the content fully visible
    to a subsequent read, even to another process.
    """

    @property
    @abstractmethod
    def root_uri(self) -> str:
        """Canonical URI prefix for this backend (e.g. ``viking://resources/ns``)."""

    @abstractmethod
    def read_text(self, key: str) -> str | None:
        """Return file content as text, or ``None`` if absent."""

    def read_uri(self, uri: str) -> str | None:
        """Read a canonical URI without converting it through a local Path."""
        prefix = self.root_uri.rstrip("/") + "/"
        if not uri.startswith(prefix):
            return None
        return self.read_text(uri.removeprefix(prefix))

    @abstractmethod
    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        """Atomically write text.  ``overwrite=False`` does not clobber an
        existing file (first-write-wins).
        """

    def write_text_durable(self, key: str, content: str) -> None:
        """Write text with a durability guarantee (fsync or remote ack).

        Default implementation delegates to :meth:`write_text`; backends with
        a stronger durability primitive (e.g. remote Viking with server-side
        ack) should override.
        """
        self.write_text(key, content)

    def write_many(self, writes: list[tuple[str, str]]) -> None:
        """Write a small related set; remote backends may commit it as a batch."""
        for key, content in writes:
            self.write_text(key, content)

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        """Commit a publication set without exposing a partially written set.

        Backends must override this method with an all-or-nothing primitive.
        Publication must never silently degrade to sequential ``write_text``
        calls because readers could then observe a mixed build.
        """
        raise RuntimeError(f"{type(self).__name__} does not support atomic publication commits")

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether the path-key exists as a file."""

    @abstractmethod
    def is_dir(self, key: str) -> bool:
        """Return whether the path-key exists as a directory."""

    @abstractmethod
    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        """Return descendant file path-keys beneath ``key`` (files only)."""

    def list_entries(self, key: str, *, recursive: bool = False) -> list[str]:
        """Return immediate descendant path-keys beneath ``key``, **including
        directories**.

        Unlike :meth:`list_dir` (files only), this exposes directory nodes so a
        caller can do layered, per-level drill-down over an arbitrary in-root
        path without a full recursive walk.  Returned keys are relative to the
        backend root.  Default behaviour reuses :meth:`list_dir`.
        """
        return self.list_dir(key, recursive=recursive)

    def list_entries_with_meta(self, key: str) -> list[tuple[str, bool]]:
        """Return ``(path_key, is_dir)`` pairs for immediate children.

        Single-call alternative to :meth:`list_entries` + per-entry
        :meth:`is_dir` — avoids N+1 network round-trips on remote backends.
        Default impl falls back to the two-call pattern (fine for local FS).
        """
        entries = self.list_entries(key, recursive=False)
        return [(e, self.is_dir(e)) for e in entries]

    @abstractmethod
    def mtime_ns(self, key: str) -> int | None:
        """Return a monotonic file modification timestamp (ns), ``None`` if absent."""

    @abstractmethod
    def size(self, key: str) -> int | None:
        """Return the file size in bytes, ``None`` if absent or unknown."""

    def fingerprint(self, key: str) -> tuple[int | None, int | None]:
        """Return modification time and size for cache validation."""
        return self.mtime_ns(key), self.size(key)

    @abstractmethod
    def mkdir_p(self, key: str) -> None:
        """Ensure a directory path-key exists (idempotent)."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove a file path-key (no-op if absent)."""

    def remove_empty_dir(self, key: str) -> bool:
        """Remove an empty directory path-key; return ``True`` if removed.

        Default returns ``False``; backends with real filesystems override.
        """
        return False

    @abstractmethod
    def move(self, src: str, dst: str) -> None:
        """Move/rename a file path-key."""

    # ── semantic retrieval (optional, VikingFS only) ─────────────────────

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        """Semantic search within this backend's root.  Default: unsupported."""
        return []

    def find(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        deep: bool = False,
    ) -> list[dict[str, object]]:
        """Retrieve ranked resources through the backend's native search API.

        ``find`` is the OpenViking fast semantic primitive.  Local storage has
        no semantic index, so the default remains an empty result rather than
        pretending that a filesystem scan is equivalent retrieval.
        """
        return []

    def grep(
        self, pattern: str, *, limit: int = 256, target_uri: str = ""
    ) -> list[dict[str, object]]:
        """Regex text search within this backend's root.  Default: unsupported."""
        return []

    def relations(self, uri: str) -> list[dict[str, object]]:
        """Return backend-native relations for one resource, if supported."""
        return []

    def sync_native_relations(self, pairs: list[tuple[str, list[str]]]) -> dict[str, object]:
        """Sync entity relations to the backend's native graph API.

        Local backends have no remote graph — relations live in entity
        frontmatter only, so this is a no-op.  ``pairs`` is
        ``(entity_uri, [linked_uris])``.
        """
        return {"linked": 0, "unlinked": 0, "errors": []}
