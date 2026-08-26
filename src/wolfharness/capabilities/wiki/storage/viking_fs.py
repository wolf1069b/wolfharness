"""Remote OpenViking backend for :class:`~wiki.storage.storage.WikiStore`.

Maps the wiki path-key layout (Concept/Class/Object.md + index/*.json) onto an
OpenViking resource tree rooted at ``viking://resources/{namespace}``.  Writes
are atomic server-side (``write(mode="replace")``), which lets us drop the local
temp-file+rename machinery.

The same namespace also doubles as the primary retrieval surface: because the
whole wiki lives in Viking, the consumer can go straight at ``client.find`` /
``client.search`` / ``client.grep`` / ``client.relations`` instead of rebuilding
a local index.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
import hashlib
import logging
import os
import threading
import time
from typing import Protocol

from httpx import HTTPError
from openviking_sdk.errors import (  # type: ignore[import-untyped]
    AlreadyExistsError,
    NotFoundError,
    OpenVikingError,
)

from .backend import FSBackend, _strip_control_chars


logger = logging.getLogger(__name__)

class VikingClient(Protocol):
    """Structural interface for the OpenViking SDK client duck-type."""

    def read(self, uri: str) -> str: ...
    def write(
        self,
        uri: str,
        content: str,
        *,
        mode: str,
        wait: bool,
        timeout: int = ...,
        processing_mode: str = "semantic_and_vectors",
    ) -> None: ...
    def stat(self, uri: str) -> dict[str, object]: ...
    def ls(self, uri: str, *, recursive: bool, simple: bool) -> list[dict[str, object]]: ...
    def rm(self, uri: str) -> None: ...
    def mkdir(self, uri: str) -> None: ...
    def mv(self, src: str, dst: str) -> None: ...
    def find(self, query: str, *, target_uri: str, limit: int) -> dict[str, object]: ...
    def search(self, query: str, *, target_uri: str, limit: int) -> dict[str, object]: ...
    def grep(self, target_uri: str, pattern: str, *, node_limit: int) -> dict[str, object]: ...
    def relations(self, uri: str) -> list[dict[str, object]]: ...
    def batch_write(
        self, root_uri: str, operations: list[dict[str, object]], *, wait: bool
    ) -> None: ...
    def read_raw(self, uri: str) -> str: ...
    def tree(self, uri: str) -> list[dict[str, object]]: ...
    def add_resource(self, *, path: str, to: str, processing_mode: str, wait: bool) -> object: ...
    def initialize(self) -> None: ...


# seconds vs ns threshold: timestamps below are seconds, above already ns
_NS_TIMESTAMP_THRESHOLD = 1e12


class VikingFS(FSBackend):
    """OpenViking-backed filesystem, one ``viking://resources/<ns>`` root."""

    # Coordinate conflicting writes without serializing unrelated entity URIs.
    # Stripes are shared by backend instances for the same namespace.
    _lock_registry_guard = threading.Lock()
    _lock_registry: dict[str, tuple[threading.RLock, ...]] = {}

    def __init__(self, namespace: str, client: VikingClient) -> None:
        """Wrap a ``openviking_sdk.SyncHTTPClient`` rooted at ``namespace``.

        ``client`` is duck-typed to keep this module import-light and
        testable without the SDK installed.
        """
        self.namespace = namespace
        self._root_uri = f"viking://resources/{namespace}"
        self._client = client
        with self._lock_registry_guard:
            self._write_locks = self._lock_registry.setdefault(
                namespace,
                tuple(threading.RLock() for _ in range(self._write_lock_stripes())),
            )

    @property
    def root_uri(self) -> str:
        return self._root_uri

    # ── key ↔ uri ────────────────────────────────────────────────────────

    def _uri(self, key: str) -> str:
        return f"{self.root_uri}/{key}"

    @staticmethod
    def _write_lock_stripes() -> int:
        """Return configurable bounded lock-striping width."""
        try:
            return max(8, min(1024, int(os.environ.get("VIKING_WRITE_LOCK_STRIPES", "64"))))
        except ValueError:
            return 64

    def _lock_index(self, uri: str) -> int:
        digest = hashlib.sha256(uri.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self._write_locks)

    def _write_lock_for(self, uri: str) -> threading.RLock:
        return self._write_locks[self._lock_index(uri)]

    # ── FSBackend ─────────────────────────────────────────────────────────

    def read_text(self, key: str) -> str | None:
        result = self._read_remote(self._uri(key))
        if result is not None:
            result = _strip_control_chars(result)
        return result

    def write_text(self, key: str, content: str, *, overwrite: bool = True) -> None:
        """Write with bounded busy-lock retries and no create/replace amplification.

        OpenViking's ``replace`` mode is the update primitive.  A failed
        ``create`` is not evidence that replacement is safe: a 409 may mean a
        live path lock.  Reading first also makes a retry idempotent when the
        server committed before a response was lost.
        """
        uri = self._uri(key)
        retry_limit = self._write_retry_limit()
        with self._write_lock_for(uri):
            mode = "create"
            for attempt in range(retry_limit + 1):
                existing = self._read_for_write(uri)
                if existing is not None:
                    # ``read`` strips a stored trailing newline; accept either
                    # form so an identical write is a genuine no-op.
                    if existing == content or existing == content.rstrip("\n") or not overwrite:
                        return
                    mode = "replace"
                try:
                    self._client.write(uri, content, mode=mode, wait=False)
                except AlreadyExistsError:
                    if not overwrite:
                        return
                    # A different writer won the create race.  Re-read before
                    # replacing it on the next iteration.
                    mode = "replace"
                except NotFoundError:
                    # The target disappeared between read and replace.
                    mode = "create"
                except OpenVikingError as exc:
                    if not self._is_busy(exc):
                        raise
                    if attempt >= retry_limit:
                        logger.exception("OpenViking write remained busy: %s", uri)
                        raise
                    delay = self._write_backoff(attempt)
                    logger.warning(
                        "OpenViking write busy; retrying uri=%s attempt=%d delay=%.2fs",
                        uri,
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    return

    def write_text_durable(self, key: str, content: str) -> None:
        """Write text and poll until read-after-write is visible.

        Remote Viking is eventually consistent: a write is accepted before
        it's visible to reads.  Control-plane receipts (prefilter markers,
        backlinks) must be visible before the caller makes a scheduling
        decision, so this method polls ``read_text`` until the content
        matches or the timeout expires.
        """
        self.write_text(key, content)
        timeout = float(os.environ.get("VIKING_DURABLE_WRITE_TIMEOUT_SECONDS", "30"))
        poll_interval = float(os.environ.get("VIKING_DURABLE_WRITE_POLL_SECONDS", "0.5"))
        deadline = time.monotonic() + timeout
        expected = content.rstrip("\n")
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            result = self.read_text(key)
            if result is not None and result.rstrip("\n") == expected:
                return
        logger.warning("Durable write to %s not visible after %.1fs", key, timeout)

    def write_many(self, writes: list[tuple[str, str]]) -> None:
        """Commit related files with OpenViking's preconditioned batch API.

        The preconditions are captured once.  A busy lock may retry the exact
        same request safely; a content conflict is returned to the caller so it
        can re-read and merge instead of silently overwriting another worker.
        """
        if not writes:
            return
        batch_write = getattr(self._client, "batch_write", None)
        if not callable(batch_write):
            for key, content in writes:
                self.write_text(key, content)
            return

        lock_indexes = sorted({self._lock_index(self._uri(key)) for key, _content in writes})
        with ExitStack() as stack:
            for lock_index in lock_indexes:
                stack.enter_context(self._write_locks[lock_index])
            operations: list[dict[str, object]] = []
            for key, content in writes:
                uri = self._uri(key)
                current = self._read_for_write(uri)
                if current is not None and (current == content or current == content.rstrip("\n")):
                    continue
                if current is None:
                    precondition: dict[str, str] = {"kind": "create_if_absent"}
                else:
                    raw_reader = getattr(self._client, "read_raw", None)
                    if not callable(raw_reader):
                        raise TypeError(
                            "OpenViking read_raw is required for exact batch preconditions",
                        )
                    stored_content = raw_reader(uri)
                    if not isinstance(stored_content, str):
                        raise TypeError("OpenViking read_raw must return text content")
                    digest = hashlib.sha256(stored_content.encode("utf-8")).hexdigest()
                    precondition = {
                        "kind": "replace_if_hash",
                        "base_hash": f"sha256:{digest}",
                    }
                operations.append(
                    {
                        "uri": uri,
                        "content": content,
                        "precondition": precondition,
                    },
                )
            if not operations:
                return

            retry_limit = self._write_retry_limit()
            for attempt in range(retry_limit + 1):
                try:
                    batch_write(self.root_uri, operations, wait=False)
                except OpenVikingError as exc:
                    if not self._is_busy(exc):
                        raise
                    if attempt >= retry_limit:
                        raise
                    delay = self._write_backoff(attempt)
                    logger.warning(
                        "OpenViking batch write busy; retrying root=%s attempt=%d delay=%.2fs",
                        self.root_uri,
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    return

    def commit_many(self, writes: list[tuple[str, str]]) -> None:
        """Atomically expose one finalized build through OpenViking batch write."""
        if not writes:
            return
        if not callable(getattr(self._client, "batch_write", None)):
            raise TypeError("OpenViking batch_write is required for atomic finalize")
        self.write_many(writes)

    def _read_for_write(self, uri: str) -> str | None:
        """Read a target while preserving non-not-found failures."""
        return self._read_remote(uri)

    def _read_remote(self, uri: str) -> str | None:
        """Read with bounded transport retries, preserving auth/data errors."""
        retry_limit = self._read_retry_limit()
        for attempt in range(retry_limit + 1):
            try:
                return str(self._client.read(uri))
            except OpenVikingError as exc:
                if self._is_not_found(exc):
                    return None
                if not self._is_transient(exc) or attempt >= retry_limit:
                    raise
            except HTTPError:
                if attempt >= retry_limit:
                    raise
            time.sleep(self._read_backoff(attempt))
        return None

    @staticmethod
    def _is_transient(exc: OpenVikingError) -> bool:
        """Return whether a server response is safe to retry as a read."""
        code = str(getattr(exc, "code", "")).upper()
        text = str(exc).lower()
        return code in {"408", "429", "500", "502", "503", "504"} or any(
            marker in text
            for marker in ("http 408", "http 429", "http 5", "temporarily unavailable", "timeout")
        )

    @staticmethod
    def _read_retry_limit() -> int:
        try:
            return max(0, min(5, int(os.environ.get("VIKING_READ_RETRIES", "3"))))
        except ValueError:
            return 3

    @staticmethod
    def _read_backoff(attempt: int) -> float:
        try:
            base = max(0.05, min(5.0, float(os.environ.get("VIKING_READ_BACKOFF_SECONDS", "0.25"))))
        except ValueError:
            base = 0.25
        return float(min(5.0, base * (2**attempt)))

    @staticmethod
    def _is_not_found(exc: OpenVikingError) -> bool:
        return str(getattr(exc, "code", "")).upper() == "NOT_FOUND" or isinstance(
            exc, NotFoundError
        )

    @staticmethod
    def _is_busy(exc: OpenVikingError) -> bool:
        code = str(getattr(exc, "code", "")).upper()
        details = getattr(exc, "details", {})
        detail_text = (
            " ".join(str(value) for value in details.values()) if isinstance(details, dict) else ""
        )
        text = f"{type(exc).__name__} {exc} {detail_text}".lower()
        return code in {"CONFLICT", "ABORTED"} and any(
            marker in text
            for marker in ("resourcebusy", "resource busy", "lock", "busy", "acquisition")
        )

    @staticmethod
    def _write_retry_limit() -> int:
        try:
            return max(0, min(8, int(os.environ.get("VIKING_WRITE_RETRIES", "4"))))
        except ValueError:
            return 4

    @staticmethod
    def _write_backoff(attempt: int) -> float:
        try:
            base = max(
                0.05,
                min(10.0, float(os.environ.get("VIKING_WRITE_BACKOFF_SECONDS", "0.75"))),
            )
        except ValueError:
            base = 0.75
        return float(min(30.0, base * (2**attempt)))

    def exists(self, key: str) -> bool:
        try:
            info = self._client.stat(self._uri(key))
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return False
            raise
        return bool(info and not info.get("isDir"))

    def is_dir(self, key: str) -> bool:
        if not key:
            # The empty key is the namespace root, which always exists.
            return True
        try:
            info = self._client.stat(self._uri(key))
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return False
            raise
        return bool(info and info.get("isDir"))

    def list_dir(self, key: str, *, recursive: bool = False) -> list[str]:
        prefix = self._uri(key)
        if recursive:
            # Parallel BFS descent using non-recursive ``ls`` per directory.
            # Avoids server-side ``glob("**")`` whose result requires an O(n²)
            # directory-prefix filter on the client. Each wave of sibling
            # directories is expanded concurrently.
            return self._walk_files_parallel(prefix)
        try:
            nodes = self._client.ls(prefix, recursive=False, simple=False)
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return []
            raise
        out: list[str] = []
        for node in nodes or []:
            uri = self._node_uri(node)
            if not uri or not uri.startswith(self.root_uri):
                continue
            rel = uri[len(self.root_uri) + 1 :]
            if self._node_is_dir(node):
                continue
            out.append(rel)
        return sorted(out)

    def _expand_dir(self, uri: str) -> tuple[list[str], list[str]]:
        """One ``ls`` call; returns (file_rels, subdir_uris)."""
        files: list[str] = []
        subdirs: list[str] = []
        try:
            nodes = self._client.ls(uri, recursive=False, simple=False)
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return files, subdirs
            raise
        for node in nodes or []:
            node_uri = self._node_uri(node)
            if not node_uri or not node_uri.startswith(self.root_uri):
                continue
            rel = node_uri[len(self.root_uri) + 1 :]
            if not rel:
                continue
            if self._node_is_dir(node):
                subdirs.append(node_uri)
            else:
                files.append(rel)
        return files, subdirs

    def _walk_files_parallel(self, prefix: str, max_workers: int = 16) -> list[str]:
        """BFS descent expanding each wave of sibling dirs concurrently."""
        out: list[str] = []
        seen: set[str] = set()
        pending = [prefix]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending:
                wave = [d for d in pending if d not in seen]
                for d in wave:
                    seen.add(d)
                pending = []
                if not wave:
                    break
                results = pool.map(self._expand_dir, wave)
                for files, subdirs in results:
                    out.extend(files)
                    pending.extend(subdirs)
        return sorted(out)

    def list_entries(self, key: str, *, recursive: bool = False) -> list[str]:
        """List immediate child path-keys (files *and* directories) in one level.

        Uses a non-recursive ``ls`` so browsing an arbitrary in-root path is a
        single lightweight request — no deep-tree ``glob``.  Directory nodes are
        retained (unlike :meth:`list_dir` which filters them out).
        """
        return [path for path, _ in self.list_entries_with_meta(key)]

    def list_entries_with_meta(self, key: str) -> list[tuple[str, bool]]:
        """Single ``ls`` returning ``(path, is_dir)`` — no per-entry ``stat``."""
        prefix = self._uri(key)
        try:
            nodes = self._client.ls(prefix, recursive=False, simple=False)
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return []
            raise
        out: list[tuple[str, bool]] = []
        for node in nodes or []:
            uri = self._node_uri(node)
            if not uri or not uri.startswith(self.root_uri):
                continue
            rel = uri[len(self.root_uri) + 1 :]
            if rel:
                out.append((rel, self._node_is_dir(node)))
        return sorted(out)

    def list_subdirs(self, key: str) -> list[str]:
        """Immediate subdirectory relative keys via one ``ls`` (no per-entry stat)."""
        prefix = self._uri(key)
        try:
            nodes = self._client.ls(prefix, recursive=False, simple=False)
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return []
            raise
        out: list[str] = []
        for node in nodes or []:
            uri = self._node_uri(node)
            if not uri or not uri.startswith(self.root_uri):
                continue
            if self._node_is_dir(node):
                rel = uri[len(self.root_uri) + 1 :]
                if rel:
                    out.append(rel)
        return sorted(out)

    def mtime_ns(self, key: str) -> int | None:
        try:
            info = self._client.stat(self._uri(key))
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return None
            raise
        if not info or info.get("isDir"):
            return None
        # Prefer an explicit timestamp; fall back to a version counter so the
        # cache-fingerprint comparison still invalidates on write.
        updated = (
            info.get("modTime")
            or info.get("updated_at")
            or info.get("version")
            or info.get("mtime")
        )
        return self._timestamp_to_ns(updated)

    def size(self, key: str) -> int | None:
        try:
            info = self._client.stat(self._uri(key))
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return None
            raise
        if not info or info.get("isDir"):
            return None
        size = info.get("size") or info.get("byte_size") or info.get("content_length")
        return int(size) if isinstance(size, (int, float, str)) and str(size).isdigit() else None

    def fingerprint(self, key: str) -> tuple[int | None, int | None]:
        """Return Viking modification time and size with one remote stat."""
        try:
            info = self._client.stat(self._uri(key))
        except OpenVikingError as exc:
            if self._is_not_found(exc):
                return None, None
            raise
        if not info or info.get("isDir"):
            return None, None
        updated = (
            info.get("modTime")
            or info.get("updated_at")
            or info.get("version")
            or info.get("mtime")
        )
        mtime_ns = self._timestamp_to_ns(updated)
        size = info.get("size") or info.get("byte_size") or info.get("content_length")
        size_value = (
            int(size) if isinstance(size, (int, float, str)) and str(size).isdigit() else None
        )
        return mtime_ns, size_value

    def mkdir_p(self, key: str) -> None:
        # Viking creates intermediate nodes implicitly on write; nothing to do.
        return None

    def delete(self, key: str) -> None:
        try:
            self._client.rm(self._uri(key))
        except OpenVikingError as exc:
            if not self._is_not_found(exc):
                raise

    def move(self, src: str, dst: str) -> None:
        self._client.mv(self._uri(src), self._uri(dst))

    # ── semantic retrieval ───────────────────────────────────────────────

    def find(
        self,
        query: str,
        *,
        target_uri: str = "",
        limit: int = 10,
        deep: bool = False,
    ) -> list[dict[str, object]]:
        """Use OpenViking's native ``find``/``search`` retrieval primitive.

        ``find`` is deliberately the default because it is bounded, stateless,
        and does not accumulate a session context during a long build.  The
        optional deep mode is reserved for workers such as OPS that need the
        server's intent-aware search.  URI scoping is delegated to OpenViking;
        the caller must still pass a configured wiki or raw namespace.

        Viking semantic processing generates ``.overview.md`` / ``.abstract.md``
        navigation files that dominate vector rankings.  We over-fetch by 3×
        and strip those metadata files so callers only see real entity pages.
        """
        target = target_uri.rstrip("/") or self.root_uri
        method = self._client.search if deep else self._client.find
        fetch_limit = max(limit * 3, limit + 20)
        try:
            result = method(query, target_uri=target, limit=fetch_limit)
        except OpenVikingError:  # pragma: no cover - optional retrieval surface
            logger.warning("viking %s failed", "search" if deep else "find", exc_info=True)
            return []
        hits = self._extract_hits(result, target_root=target)
        return hits[:limit]

    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        return self.find(query, limit=limit, deep=True)

    def grep(
        self, pattern: str, *, limit: int = 256, target_uri: str = ""
    ) -> list[dict[str, object]]:
        try:
            result = self._client.grep(
                target_uri.rstrip("/") or self.root_uri,
                pattern,
                node_limit=limit,
            )
        except OpenVikingError:  # pragma: no cover - optional retrieval surface
            logger.warning("viking grep failed", exc_info=True)
            return []
        return self._extract_hits(result, target_root=target_uri.rstrip("/") or self.root_uri)

    def relations(self, uri: str) -> list[dict[str, object]]:
        """Read OpenViking's native relation edges for *uri*."""
        try:
            result = self._client.relations(uri)
        except OpenVikingError:  # pragma: no cover - optional graph surface
            logger.warning("viking relations failed: %s", uri, exc_info=True)
            return []
        return [item for item in result or [] if isinstance(item, dict)]

    _METADATA_FILENAMES = frozenset({".overview.md", ".abstract.md", "entities.json"})

    def _extract_hits(
        self, result: dict[str, object], *, target_root: str = ""
    ) -> list[dict[str, object]]:
        hits: list[dict[str, object]] = []
        scope = target_root.rstrip("/") or self.root_uri
        for bucket in ("resources", "memories", "skills"):
            bucket_items = result.get(bucket, [])
            if not isinstance(bucket_items, list):
                continue
            for item in bucket_items:
                if not isinstance(item, dict):
                    continue
                uri = str(item.get("uri", ""))
                if not (uri == scope or uri.startswith(scope + "/")):
                    continue
                leaf = uri.rsplit("/", 1)[-1] if "/" in uri else uri
                if leaf in self._METADATA_FILENAMES:
                    continue
                key = uri[len(scope) + 1 :] if len(uri) > len(scope) else uri
                hits.append(
                    {
                        "key": key,
                        "uri": uri,
                        "score": item.get("score"),
                        "abstract": item.get("abstract"),
                        "context_type": item.get("context_type"),
                    },
                )
        return hits

    # ── node inspection (structure-coupling isolated here) ───────────────

    @staticmethod
    def _node_uri(node: object) -> str | None:
        if isinstance(node, dict):
            return str(node.get("uri") or node.get("path") or node.get("name") or "")
        return str(node) if node else ""

    @staticmethod
    def _node_is_dir(node: object) -> bool:
        if isinstance(node, dict):
            if node.get("isDir") is True:
                return True
            ntype = str(node.get("type") or node.get("kind") or "")
            return ntype in {"directory", "dir", "folder"}
        return False

    @staticmethod
    def _timestamp_to_ns(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            # seconds → ns; tolerate already-ns values.
            return int(value * 1e9) if abs(value) < 1e12 else int(value)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return int(dt.timestamp() * 1e9)
            except ValueError:
                return None
        return None
