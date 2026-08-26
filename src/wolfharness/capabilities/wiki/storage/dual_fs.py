"""Dual-backend: mirror every write to local disk *and* remote OpenViking.

``DualFS`` wraps a :class:`LocalFS` (authoritative reads, zero-HTTP during a
build) and a :class:`VikingFS` (remote mirror).  Every mutation is applied to
both backends so the wiki is immediately available both locally (synchronous,
never blocked by network) and remotely (best-effort, failures are logged, never
raise).  Reading is served from the local mirror only — the local copy is the
source of truth for the current process.

This is the storage mode an operator wants when a capability (e.g. the wiki
ticket tools) must produce state that outsiders can consume from OpenViking
right away while the running service keeps a local ``output/`` copy for
debugging and offline work.
"""

from __future__ import annotations

import logging

from .backend import FSBackend


logger = logging.getLogger(__name__)


class DualFS(FSBackend):
    """Mirror writes to a local backend and a remote Viking backend."""

    def __init__(self, local: FSBackend, remote: FSBackend) -> None:
        self.local = local
        self.remote = remote
        # WikiStore inspects ``backend.namespace`` to derive its viking root.
        self.namespace = str(getattr(remote, "namespace", ""))

    @property
    def root_uri(self) -> str:
        return self.remote.root_uri

    # ── reads: local mirror is authoritative ────────────────────────────

    def read_text(self, key: str) -> str | None:
        return self.local.read_text(key)

    def exists(self, key: str) -> bool:
        return self.local.exists(key)

    def is_dir(self, key: str) -> bool:
        return self.local.is_dir(key)

    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        return self.local.list_dir(key, recursive=recursive)

    def list_entries(self, key: str, *, recursive: bool = False) -> list[str]:
        return self.local.list_entries(key, recursive=recursive)

    def list_entries_with_meta(self, key: str) -> list[tuple[str, bool]]:
        return self.local.list_entries_with_meta(key)

    def mtime_ns(self, key: str) -> int | None:
        return self.local.mtime_ns(key)

    def size(self, key: str) -> int | None:
        return self.local.size(key)

    def fingerprint(self, key: str) -> tuple[int | None, int | None]:
        return self.local.fingerprint(key)

    def find(
        self, query: str, *, target_uri: str = "", limit: int = 10, deep: bool = False
    ) -> list[dict[str, object]]:
        """Semantic retrieval only exists on the remote backend — delegate."""
        return self.remote.find(query, target_uri=target_uri, limit=limit, deep=deep)

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        return self.remote.search(query, limit=limit)

    def grep(
        self, pattern: str, *, limit: int = 256, target_uri: str = ""
    ) -> list[dict[str, object]]:
        return self.remote.grep(pattern, limit=limit, target_uri=target_uri)

    def relations(self, uri: str) -> list[dict[str, object]]:
        return self.remote.relations(uri)

    # ── writes: mirror to both backends, remote is best-effort ──────────

    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        self.local.write_text(key, content, overwrite=overwrite)
        self._mirror("write_text", key, content=content, overwrite=overwrite)

    def write_text_durable(self, key: str, content: str) -> None:
        self.local.write_text_durable(key, content)
        self._mirror("write_text_durable", key, content=content)

    def write_many(self, writes: list[tuple[str, str]]) -> None:
        self.local.write_many(writes)
        self._mirror("write_many", None, writes=writes)

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        self.local.commit_many(writes)
        self._mirror("write_many", None, writes=writes)

    def mkdir_p(self, key: str) -> None:
        self.local.mkdir_p(key)
        self._mirror("mkdir_p", key)

    def delete(self, key: str) -> None:
        self.local.delete(key)
        self._mirror("delete", key)

    def move(self, src: str, dst: str) -> None:
        self.local.move(src, dst)
        self._mirror("move", None, src=src, dst=dst)

    # ── internals ────────────────────────────────────────────────────────

    def _mirror(self, method: str, key: str | None, **kwargs: object) -> None:
        """Apply one mutation to the remote backend, swallowing failures.

        The local write already happened; the remote mirror must never take the
        caller down with it.  Failures are logged so an operator can detect a
        silently-desynced remote.
        """
        try:
            if key is not None:
                getattr(self.remote, method)(key, **kwargs)
            else:
                getattr(self.remote, method)(**kwargs)
        except Exception as exc:
            logger.warning("DualFS remote mirror %s failed (key=%s): %s", method, key, exc)
