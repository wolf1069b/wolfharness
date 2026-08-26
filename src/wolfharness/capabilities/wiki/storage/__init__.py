"""Storage backends for :class:`~wiki.storage.storage.WikiStore`.

The same knowledge-graph store can live on a local directory (``LocalFS``,
read-only fallback for historical libs) or on a remote OpenViking instance
(``VikingFS``, the primary read/write path).  The active backend is chosen
per-process from the environment:

- ``WIKI_STORAGE_BACKEND``: ``viking`` (default) or ``local``
- ``VIKING_BASE_URL``: OpenViking server root (default ``http://viking.ai.rootcloud.info/``)
- ``VIKING_API_KEY``: API key / JWT for the OpenViking server
- ``VIKING_NAMESPACE``: ``viking://resources/<namespace>`` root to store the wiki under

``create_wiki_store`` is the single entry point used by ``WikiBuildTools``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from openviking_sdk.errors import NotFoundError, OpenVikingError  # type: ignore[import-untyped]
from wolfharness.capabilities.wiki.namespaces import raw_namespace, wiki_namespace

from .backend import FSBackend, _strip_control_chars
from .dual_fs import DualFS
from .local_fs import LocalFS
from .local_viking_fs import LocalVikingFS
from .storage import WikiStore
from .viking_fs import VikingClient, VikingFS

logger = logging.getLogger(__name__)

__all__ = [
    "DualFS",
    "FSBackend",
    "LocalFS",
    "LocalVikingFS",
    "VikingFS",
    "WikiStore",
    "create_raw_reader",
    "create_wiki_store",
]

_VIKING_DEFAULT_URL = "http://viking.ai.rootcloud.info/"

_client_cache: VikingClient | None = None


def _viking_client() -> VikingClient:
    """Return a memoized OpenViking HTTP client (shared by wiki + raw backends)."""
    global _client_cache  # noqa: PLW0603 - process-wide memoized singleton
    if _client_cache is None:
        from openviking_sdk import SyncHTTPClient

        _client_cache = SyncHTTPClient(
            url=os.environ.get("VIKING_BASE_URL", _VIKING_DEFAULT_URL),
            api_key=os.environ.get("VIKING_API_KEY"),
        )
        _client_cache.initialize()
    return _client_cache


def viking_read(uri: str, *, propagate_unavailable: bool = False) -> str | None:
    """Read an arbitrary ``viking://resources/<ns>/...`` URI via the shared client.

    Cross-namespace direct read: the caller passes the full URI (e.g. the
    ``730`` raw case library) rather than a namespace-relative key.  Returns
    ``None`` when the node does not exist.
    """
    try:
        content = _viking_client().read(uri)
    except OpenVikingError as exc:
        if isinstance(exc, NotFoundError) or str(getattr(exc, "code", "")).upper() == "NOT_FOUND":
            return None
        logger.warning("viking read failed: %s (%s)", uri, type(exc).__name__)
        if propagate_unavailable:
            raise
        return None
    return _strip_control_chars(content) if content is not None else None


def viking_list_children(uri: str, *, recursive: bool = False) -> dict[str, object]:
    """Enumerate direct children of a ``viking://resources/<ns>/...`` URI.

    Uses the SDK's ``ls`` (non-recursive) or ``tree`` (recursive) so the
    conductor can drill down into an arbitrary remote tree (raw manuals, cases,
    repairmenus) by URI instead of a fixed local ``chapters/`` layout.

    Returns:
        A browse payload with a ``children`` list of ``{"name", "uri",
        "is_dir"}`` entries, or an error payload on unknown URIs.
    """
    client = _viking_client()
    target = uri.rstrip("/")
    try:
        nodes = (
            client.tree(target) if recursive else client.ls(target, recursive=False, simple=False)
        )
    except Exception as exc:
        if isinstance(exc, NotFoundError) or str(getattr(exc, "code", "")).upper() == "NOT_FOUND":
            return {"uri": uri, "type": "error", "error": "not found"}
        logger.warning("viking list failed: %s (%s)", uri, type(exc).__name__)
        return {"uri": uri, "type": "error", "error": str(exc)[:200]}

    children: list[dict[str, object]] = []
    seen: set[str] = set()
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        node_uri = str(node.get("uri") or node.get("path") or node.get("name") or "")
        if not node_uri.startswith(target + "/"):
            continue
        # For recursive tree() the immediate children are at the first path
        # segment after the target; skip deeper nodes.
        remainder = node_uri[len(target) + 1 :]
        segment = remainder.split("/", 1)[0]
        if segment in seen:
            continue
        seen.add(segment)
        # A node nested deeper than one level contributes its parent
        # directory as the immediate child; keep the collapsed dir URI.
        child_uri = target + "/" + segment if "/" in remainder else node_uri
        is_dir = (
            "/" in remainder
            or bool(node.get("isDir"))
            or str(
                node.get("type") or node.get("kind") or "",
            )
            in {"directory", "dir", "folder"}
        )
        children.append({"name": segment, "uri": child_uri, "is_dir": is_dir})
    return {
        "uri": uri,
        "type": "viking",
        "child_count": len(children),
        "children": children,
    }


def create_wiki_store(wiki_root: str | Path) -> WikiStore:
    """Build a :class:`WikiStore` over the backend selected by env config.

    ``WIKI_STORAGE_BACKEND`` selects the storage mode:

    - ``viking`` (default): store remotely under
      ``viking://resources/{VIKING_NAMESPACE}``.
    - ``local``: store under ``wiki_root`` on the local filesystem.
    - ``local_viking``: store locally with ``viking://`` URI identity, then
      upload to OpenViking at finalize time.
    - ``dual``: mirror every write to the local ``wiki_root`` **and** the
      remote OpenViking namespace immediately (local is authoritative for
      reads; remote mirror is best-effort, failures are logged not raised).
      ``VIKING_NAMESPACE`` / ``VIKING_BASE_URL`` / ``VIKING_API_KEY`` are
      then required to resolve the remote root.
    """
    from wolfharness.capabilities.wiki.quality import set_wiki_root_uri

    backend = os.environ.get("WIKI_STORAGE_BACKEND", "viking")
    if backend == "local_viking":
        namespace = wiki_namespace()
        local_root = Path(wiki_root)
        local_root.mkdir(parents=True, exist_ok=True)
        store = WikiStore(LocalVikingFS(namespace, local_root))
        logger.info("WikiStore over LocalVikingFS, namespace=%s, root=%s", namespace, local_root)
    elif backend == "dual":
        namespace = wiki_namespace()
        local_root = Path(wiki_root)
        local_root.mkdir(parents=True, exist_ok=True)
        store = WikiStore(
            DualFS(
                LocalFS(local_root),
                VikingFS(namespace, _viking_client()),
            ),
        )
        logger.info("WikiStore over DualFS (local %s + viking %s)", local_root, namespace)
    elif backend != "viking":
        store = WikiStore(Path(wiki_root))
    else:
        namespace = wiki_namespace()
        logger.info("WikiStore over OpenViking backend, namespace=%s", namespace)
        store = WikiStore(VikingFS(namespace, _viking_client()))
    set_wiki_root_uri(store.root_uri)
    return store


def create_raw_reader(local_root: str | Path) -> FSBackend:
    """Build an :class:`FSBackend` for reading raw manual chapter markdown.

    Only the pure ``viking`` backend reads chapters remotely from
    ``viking://resources/{VIKING_RAW_NAMESPACE}``; ``local`` and
    ``local_viking`` read from the local ``local_root`` directory so the
    build phase never blocks on remote OpenViking HTTP. An explicit
    ``viking://`` URI as ``local_root`` always opts back into the remote
    backend. All backends expose path-keys like
    ``sy215c/chapters/01_1 前言/chapter.md``.
    """
    configured_root = str(local_root).strip().rstrip("/")
    explicit_prefix = "viking://resources/"
    # An explicit viking:// URI forces the remote backend regardless of the
    # storage backend setting.
    if configured_root.startswith(explicit_prefix):
        namespace = configured_root.removeprefix(explicit_prefix).strip("/")
        if not namespace:
            raise ValueError("Raw Viking URI must include a resource namespace")
        logger.info("Raw reader over OpenViking backend, namespace=%s", namespace)
        return VikingFS(namespace, _viking_client())
    # local / local_viking builds read raw chapters from the local filesystem;
    # only the pure "viking" backend fetches them remotely.
    backend = os.environ.get("WIKI_STORAGE_BACKEND", "viking")
    if backend == "local_viking":
        namespace = raw_namespace()
        logger.info(
            "Raw reader over local filesystem with viking URI, namespace=%s, root=%s",
            namespace,
            local_root,
        )
        return LocalFS(Path(local_root), root_uri=f"viking://resources/{namespace}")
    return LocalFS(Path(local_root))
    namespace = raw_namespace()
    logger.info("Raw reader over OpenViking backend, namespace=%s", namespace)
    return VikingFS(namespace, _viking_client())
