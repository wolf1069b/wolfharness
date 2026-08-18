"""Example: Knowledge Base MCP server backed by real files.

This demo shows how to build a standalone MCP server that exposes files
(such as ``intro.md``, ``logo.png``, ``structure.png``) from the ``kb_data/``
directory through the MCP Resource protocol. It demonstrates:

1. **Real files on disk** — Markdown documents and PNG images live in ``kb_data/``.
2. **A custom URI scheme (``kb://``)** — Resources use a domain-specific scheme
   with a namespace host (``kb://docs/...``, ``kb://images/...``) and real file
   extensions. Custom schemes are allowed by the spec as long as they follow
   RFC 3986.
3. **Resource templates, not static resources** — ``kb://docs/{name*}`` and
   ``kb://images/{name*}`` serve documents and images by URI and pick up files
   added later; ``kb://search{?q}`` demonstrates an RFC 6570 query-parameter
   template.
4. **A search tool** — Clients can search the KB via a regular MCP tool.
5. **Dynamic static resources** — ``resources/list`` is re-scanned in the
   background so files added/removed in ``kb_data/`` appear without restarting
   (``--scan-interval`` seconds).

Run with stdio (default, for IDE integration)::

    uv run python examples/kb_mcp_server_example.py

Run with streamable-http (for web clients)::

    uv run python examples/kb_mcp_server_example.py --transport streamable-http --port 8002

Test with ``fastmcp`` CLI::

    uv run fastmcp dev examples/kb_mcp_server_example.py
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import UTC, datetime
import json
import logging
import mimetypes
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import FastMCP
from fastmcp.resources import FileResource
from fastmcp.resources.base import ResourceContent
from fastmcp.server.lifespan import Lifespan
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
import mcp.types as mcp_types
from pydantic import Field


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger("kb-mcp-demo")

# Root of the knowledge base on disk. Defaults to ``examples/kb_data/``; the
# ``--kb-dir`` CLI flag overrides it, which lets tests point the server at a
# temporary directory. Kept mutable (not ``Final``) so tests can also re-point
# it before registering resources.
KB_DIR: Path = Path(__file__).parent / "kb_data"


def _namespace_dirs(kb_dir: Path = KB_DIR) -> dict[str, Path]:
    """Map URI hosts (``kb://docs``, ``kb://images``) to filesystem dirs.

    The ``kb://`` scheme uses the URI host as a namespace: ``kb://docs/intro.md``
    reads ``<kb_dir>/docs/intro.md``.
    """
    return {
        "docs": kb_dir / "docs",
        "images": kb_dir / "images",
    }


_NAMESPACE_DIRS = _namespace_dirs()

# MIME types that ``mimetypes`` does not resolve from a bare filename.
_MIME_BY_SUFFIX = {".md": "text/markdown"}
_mimetypes = mimetypes.MimeTypes()
for _suffix, _mime in _MIME_BY_SUFFIX.items():
    _mimetypes.add_type(_mime, _suffix)


def _mime_for(path: Path) -> str:
    """Return the MIME type for a file, falling back to ``application/octet-stream``."""
    return _mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _resource_description(path: Path, uri: str) -> str:
    """Return a short human-readable description for a resource.

    Descriptions surface in MCP ``resources/list`` and become the subtitle
    shown next to a resource in ``@`` mention menus, so they should be
    succinct and file-type aware.
    """
    kind = {
        ".md": "Markdown document",
        ".txt": "Text document",
        ".markdown": "Markdown document",
        ".png": "PNG image",
        ".jpg": "JPEG image",
        ".jpeg": "JPEG image",
    }.get(path.suffix, "File")
    return f"{kind} at {uri}"


def _kbs_files() -> list[Path]:
    """All files under kb_data/, grouped by namespace directory."""
    files: list[Path] = []
    for _dir in _NAMESPACE_DIRS.values():
        if _dir.is_dir():
            files.extend(p for p in _dir.rglob("*") if p.is_file())
    return sorted(files)


def _uri_for(path: Path) -> str:
    """Map a file on disk to its kb:// resource URI.

    The namespace directory name becomes the URI host: ``kb_data/docs/intro.md``
    → ``kb://docs/intro.md``.
    """
    rel = path.relative_to(KB_DIR).as_posix()  # e.g. "docs/intro.md"
    ns, _, name = rel.partition("/")
    return f"kb://{ns}/{name}"


def _read_file(path: Path) -> str | bytes:
    if path.suffix in {".md", ".txt", ".markdown"}:
        return path.read_text(encoding="utf-8")
    return path.read_bytes()


def _resolve_kb_uri(ns: str, name: str) -> Path:
    """Resolve a kb:// namespace + name to a file, raising KeyError on issues."""
    base = _NAMESPACE_DIRS.get(ns)
    if base is None:
        raise KeyError(f"Unknown namespace: {ns!r}")
    target = (base / name).resolve()
    if not target.is_relative_to(base.resolve()):
        raise KeyError(f"Path escapes namespace {ns!r}: {name!r}")
    if not target.is_file():
        raise KeyError(f"Resource not found in kb://{ns}/{name}")
    return target


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class _KBServer(FastMCP):
    """FastMCP server that keeps ``resources/list`` in sync with ``kb_data/``.

    A background task re-scans the KB directory every ``scan_interval``
    seconds so newly added files appear in ``resources/list`` (hence in
    ``@`` mention) without a server restart. When the list changes, a
    ``notifications/resources/list_changed`` broadcast goes to every client
    session the server has seen (see ``_SessionRegistryMiddleware``).
    """

    def __init__(self, scan_interval: float = 5.0, **kwargs: Any) -> None:
        super().__init__(lifespan=Lifespan(_kb_lifespan(scan_interval)), **kwargs)
        # Auto-register every client session that makes a request so the
        # scan loop can broadcast list_changed without client cooperation.
        self.add_middleware(_SessionRegistryMiddleware())


class _SessionRegistryMiddleware(Middleware):
    """Register every client session with the live-session registry.

    FastMCP 3.4.4 has no broadcast API; notifications are per-session. This
    middleware captures each client's session on inbound requests, so the
    background scan loop can reach all connected clients without requiring
    them to call a registration tool first.
    """

    async def on_message(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
    ) -> Any:
        ctx = context.fastmcp_context
        if ctx is not None:
            try:
                session = ctx.session
            except RuntimeError:
                session = None
            if session is not None:
                _active_sessions.add(session)
        return await call_next(context)


def _kb_lifespan(scan_interval: float):
    async def _enter(server: FastMCP[Any]) -> AsyncIterator[dict[str, Any]]:
        sync_static_resources()
        task = asyncio.create_task(_scan_loop(scan_interval))
        try:
            yield {}
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return _enter


async def _scan_loop(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            if sync_static_resources():
                await _broadcast_resources_list_changed()
        except Exception:
            logger.warning("kb_data/ sync failed; will retry next scan", exc_info=True)


_registered_uris: set[str] = set()

# --- Dynamic change notification (best-practice demo) ------------------------
#
# FastMCP (3.4.4) has no FastMCP-level broadcast API: notifications are
# per-session, sent through ``Context.send_notification()``. Since the
# background scan loop created inside the lifespan only holds the ``FastMCP``
# server (not any session), we keep a registry of live client sessions here.
# ``_SessionRegistryMiddleware`` fills it automatically on every inbound
# request; the scan loop iterates it when ``resources/list`` changes and
# pushes ``notifications/resources/list_changed`` to each registered client.

_active_sessions: set[Any] = set()
_session_lock = asyncio.Lock()


def _remove_resource(uri: str) -> None:
    """Remove a static resource from the underlying provider.

    Ignores ``KeyError`` (resource already gone). FastMCP has no public
    ``remove_resource``; the provider layer does.
    """
    for provider in getattr(mcp, "providers", []):
        remove = getattr(provider, "remove_resource", None)
        if remove is None:
            continue
        with contextlib.suppress(KeyError):
            remove(uri)
        return


async def _broadcast_resources_list_changed() -> None:
    """Push ``notifications/resources/list_changed`` to registered clients.

    Iterates the registry of live sessions (filled automatically by
    ``_SessionRegistryMiddleware``) and sends the notification to each
    still-connected one, dropping dead sessions as they are found.
    """
    async with _session_lock:
        snapshot = list(_active_sessions)
    for session in snapshot:
        try:
            await session.send_notification(
                mcp_types.ServerNotification(mcp_types.ResourceListChangedNotification())
            )
        except (ConnectionError, OSError, ValueError):
            # Client disconnected mid-broadcast: drop it so the next scan
            # doesn't try again. This is deliberate best-effort broadcasting.
            async with _session_lock:
                _active_sessions.discard(session)


def sync_static_resources() -> bool:
    """Sync the static resource list with the KB directory.

    Returns ``True`` if the resource set changed (added or removed). Scans
    ``kb_data/`` and adds/removes ``FileResource`` entries so
    ``resources/list`` reflects the current directory contents without
    restarting the server. Deleted or renamed files are removed; new files
    are added.
    """
    changed = False
    current = {_uri_for(path) for path in _kbs_files()}
    for uri in _registered_uris - current:
        _remove_resource(uri)
        _registered_uris.discard(uri)
        changed = True
    for path in _kbs_files():
        uri = _uri_for(path)
        if uri in _registered_uris:
            continue
        is_binary = path.suffix in {".png", ".jpg", ".jpeg"}
        mcp.add_resource(
            FileResource(
                uri=uri,
                path=path,
                is_binary=is_binary,
                mime_type=_mime_for(path),
                title=f"{path.stem} ({path.parent.name})",
                description=_resource_description(path, uri),
                annotations=mcp_types.Annotations(
                    lastModified=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
                ),
            )
        )
        _registered_uris.add(uri)
        changed = True
    return changed


mcp = _KBServer(
    name="kb-server",
    instructions=(
        "Knowledge Base MCP Server.\n\n"
        "Serves files from the local `kb_data/` directory via the MCP Resource "
        "protocol using the custom `kb://` URI scheme.\n"
        "Use `list_resources` to discover available files, then `read_resource` "
        "to fetch their contents.\n\n"
        "Resource URIs:\n"
        "- kb://docs/<name> — Markdown documents (e.g. kb://docs/intro.md)\n"
        "- kb://images/<name> — PNG images (e.g. kb://images/logo.png)\n"
        "- kb://search{?q} — search the KB by query string (template)"
    ),
)


# --- Resource templates (dynamic URIs) ---------------------------------------


@mcp.resource(
    "kb://docs/{name*}.md",
    name="Document",
    description="A Markdown document in the docs namespace",
    mime_type="text/markdown",
)
def get_document(name: str) -> list[ResourceContent]:
    """Read a Markdown document from the knowledge base.

    Args:
        name: Document path under the docs namespace, without the ``.md`` suffix,
            e.g. "intro", "sub/readme".

    Raises:
        KeyError: If the document does not exist or resolves outside the docs
            namespace.
    """
    content = _read_file(_resolve_kb_uri("docs", f"{name}.md"))
    return [ResourceContent(content, mime_type="text/markdown")]


@mcp.resource(
    "kb://images/{name*}.png",
    name="Image",
    description="A PNG image in the images namespace",
    mime_type="image/png",
)
def get_image(name: str) -> list[ResourceContent]:
    """Read a PNG image from the knowledge base.

    Args:
        name: Image path under the images namespace, without the ``.png`` suffix,
            e.g. "logo", "structure".

    Raises:
        KeyError: If the image does not exist or resolves outside the images
            namespace.
    """
    content = _read_file(_resolve_kb_uri("images", f"{name}.png"))
    return [ResourceContent(content, mime_type="image/png")]


@mcp.resource(
    "kb://docs/",
    name="Docs index",
    description="JSON listing of all documents in the docs namespace",
    mime_type="application/json",
)
def list_docs() -> str:
    """Return a JSON listing of every document under ``kb://docs/``.

    Directory reads return a single JSON listing content (not one content per
    file), because the MCP spec intends each ``contents[].uri`` to identify a
    concrete resource. Clients then read each file through the
    ``kb://docs/{name*}.md`` template.
    """
    files = [
        {"uri": _uri_for(p), "name": p.name, "size": p.stat().st_size}
        for p in _kbs_files()
        if p.parent == _NAMESPACE_DIRS["docs"]
    ]
    return json.dumps({"namespace": "docs", "files": files}, indent=2)


@mcp.resource(
    "kb://search{?q}",
    name="Search",
    description="Search the knowledge base by query string",
)
def search_resource(q: str = "") -> str:
    """Search the knowledge base, returning matches as a JSON snippet list.

    Args:
        q: Search term to match against text documents.

    Returns:
        The first matching document's content, or an empty string.
    """
    if not q:
        return ""
    for path in _kbs_files():
        if path.suffix not in {".md", ".txt", ".markdown"}:
            continue
        content = path.read_text(encoding="utf-8")
        if q.lower() in content.lower():
            return content
    return ""


# --- Tools ------------------------------------------------------------------


@mcp.tool
def search_kb(query: Annotated[str, Field(description="Search query string")]) -> str:
    """Search the knowledge base for text files matching the query.

    Performs a simple case-insensitive substring search across all text
    documents (``.md``, ``.txt``). Returns matching file paths and relevant
    excerpts.

    Args:
        query: The search term to look for.

    Returns:
        JSON string with search results.
    """
    query_lower = query.lower()
    results: list[dict[str, Any]] = []
    for path in _kbs_files():
        if path.suffix not in {".md", ".txt", ".markdown"}:
            continue
        content = path.read_text(encoding="utf-8")
        if query_lower not in content.lower():
            continue
        idx = content.lower().find(query_lower)
        start = max(0, idx - 40)
        end = min(len(content), idx + len(query) + 40)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        results.append({
            "file": path.name,
            "uri": _uri_for(path),
            "snippet": f"{prefix}{content[start:end]}{suffix}",
        })
    return json.dumps(
        {"query": query, "match_count": len(results), "results": results},
        indent=2,
    )


@mcp.tool
def list_kb_files() -> str:
    """List all files in the knowledge base.

    Returns a JSON string listing every file with its resource URI, MIME type,
    and size in bytes.
    """
    files = [
        {
            "file": p.name,
            "uri": _uri_for(p),
            "mime_type": _mime_for(p),
            "size": p.stat().st_size,
        }
        for p in _kbs_files()
    ]
    return json.dumps(
        {"total_files": len(files), "files": files},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and start the MCP server."""
    parser = argparse.ArgumentParser(description="Knowledge Base MCP Server Demo")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport type (default: stdio)",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP transports")
    parser.add_argument("--port", type=int, default=8002, help="Port for HTTP transports")
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=5.0,
        help="Seconds between kb_data/ re-scans for dynamic resources/list (default: 5.0)",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge base directory (defaults to examples/kb_data)",
    )
    args = parser.parse_args()

    if args.kb_dir is not None:
        global KB_DIR, _NAMESPACE_DIRS  # noqa: PLW0603
        KB_DIR = args.kb_dir
        _NAMESPACE_DIRS = _namespace_dirs(KB_DIR)

    print(f"Starting KB MCP Server (transport={args.transport})")
    print(f"  KB directory: {KB_DIR}")
    for path in _kbs_files():
        print(f"    {_uri_for(path)}  ({_mime_for(path)})")
    print()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
