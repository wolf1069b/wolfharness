"""Tool functions for the Viking capability.

Each tool is an async closure that captures the ``VikingCapability``
instance and takes a ``RunContext`` as the first parameter. All tools
wrap SDK calls in try/except and return ``ToolReturn`` objects — they
never raise exceptions to the caller.
"""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.tools import RunContext  # noqa: TC002 - needed at runtime for get_type_hints()

from wolfharness.capabilities.viking.constants import IMAGE_EXTENSIONS, IMAGE_MIME_TYPES
from wolfharness.capabilities.viking.utils import (
    add_line_numbers,
    format_glob_results,
    format_grep_results,
    format_ls_entries,
    format_search_results,
    truncate_text,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from wolfharness.capabilities.viking import VikingCapability


def _get_session_id(ctx: RunContext[Any]) -> str | None:
    """Extract session_id from RunContext deps if available."""
    deps = ctx.deps
    if deps is not None and hasattr(deps, "session_id"):
        return str(deps.session_id)
    return None


def _is_image_resource(uri: str) -> bool:
    """Whether a URI points to an image resource by its file extension.

    Matches the openviking server's extension-based image detection
    (``IMAGE_EXTENSIONS``). SVG is deliberately excluded — it is a vector
    format most vision APIs reject, so it never enters the byte path.
    """
    return PurePosixPath(uri).suffix.lower() in IMAGE_EXTENSIONS


def _image_uri_hint(uri: str) -> str:
    """Text hint for an image URI when image bytes are not returned.

    Used when the model cannot consume image bytes (text-only) or bytes
    are forced off. Mentions the URI so the model can still reference it.
    """
    return (
        f"[Image resource: {uri}]\n"
        f"The file is an image and cannot be shown as text. The image is "
        f"stored at the URI above — reference it when discussing the content."
    )


def build_tools(cap: VikingCapability) -> list[Callable[..., Any]]:
    """Build the list of tool functions for the Viking capability.

    The returned functions are plain async callables suitable for use
    with pydantic-ai's ``FunctionToolset``. Tools are filtered based on
    ``cap.mode``:

    - ``"retrieve"``: 7 read-only tools
    - ``"write"``: 6 write tools
    - ``"graph"``: 2 graph tools
    - ``"all"``: all 15 tools

    Args:
        cap: The ``VikingCapability`` instance that owns these tools.

    Returns:
        A list of async tool functions.
    """
    tools: list[Callable[..., Any]] = []

    # ---- Retrieve tools (7) ----

    if cap.mode in ("retrieve", "all"):

        async def viking_search(
            ctx: RunContext[Any],
            query: str,
            limit: int = 10,
            min_score: float = 0.35,
            level: list[int] | None = None,
            target_uri: str | list[str] = "",
        ) -> ToolReturn:
            """Search the Viking knowledge graph semantically.

            Uses embedding-based search to find relevant content.
            Results include relevance scores and snippets.

            Args:
                query: Natural-language search query.
                limit: Maximum number of results to return.
                min_score: Minimum relevance score (0.0 to 1.0).
                level: Filter by content level (e.g. [0, 1, 2] for L0-L2).
                target_uri: Restrict search to specific URI subtrees — a
                    single ``viking://`` URI or a list of them.

            Returns:
                Formatted search results grouped by context type.
            """
            try:
                client = await cap._ensure_client()
                sid = _get_session_id(ctx)
                sdk_filter: dict[str, Any] | None = {"level": level} if level else None
                if cap.allowed_uri_prefixes:
                    if isinstance(target_uri, str):
                        if target_uri:
                            err = cap._check_uri_allowed(target_uri, tool_name="viking_search")
                            if err:
                                return ToolReturn(return_value=err)
                    else:
                        for u in target_uri:
                            err = cap._check_uri_allowed(u, tool_name="viking_search")
                            if err:
                                return ToolReturn(return_value=err)
                    if not target_uri:
                        # SDK target_uri accepts a list — the server searches
                        # every allowed prefix. The old default used only the
                        # first prefix, silently dropping the other allowed
                        # trees.
                        target_uri = cap.allowed_uri_prefixes
                result = await client.search(
                    query,
                    target_uri=target_uri,
                    session_id=sid,
                    limit=limit,
                    score_threshold=min_score,
                    filter=sdk_filter,
                )
                return ToolReturn(return_value=format_search_results(result))
            except Exception as e:
                return ToolReturn(return_value=f"viking_search error ({type(e).__name__}): {e}")

        async def viking_find(
            ctx: RunContext[Any],
            query: str,
            limit: int = 10,
            min_score: float = 0.35,
            level: list[int] | None = None,
            target_uri: str | list[str] = "",
        ) -> ToolReturn:
            """Find content in Viking, deduplicating results.

            Similar to ``viking_search`` but deduplicates near-identical
            hits, returning a more diverse result set.

            Args:
                query: Natural-language search query.
                limit: Maximum number of results to return.
                min_score: Minimum relevance score (0.0 to 1.0).
                level: Filter by content level (e.g. [0, 1, 2] for L0-L2).
                target_uri: Restrict search to specific URI subtrees — a
                    single ``viking://`` URI or a list of them.

            Returns:
                Formatted search results grouped by context type.
            """
            try:
                client = await cap._ensure_client()
                sdk_filter: dict[str, Any] | None = {"level": level} if level else None
                if cap.allowed_uri_prefixes:
                    if isinstance(target_uri, str):
                        if target_uri:
                            err = cap._check_uri_allowed(target_uri, tool_name="viking_find")
                            if err:
                                return ToolReturn(return_value=err)
                    else:
                        for u in target_uri:
                            err = cap._check_uri_allowed(u, tool_name="viking_find")
                            if err:
                                return ToolReturn(return_value=err)
                    if not target_uri:
                        # See viking_search: SDK target_uri accepts a list so we
                        # search every allowed prefix, not just the first.
                        target_uri = cap.allowed_uri_prefixes
                result = await client.find(
                    query,
                    target_uri=target_uri,
                    limit=limit,
                    score_threshold=min_score,
                    filter=sdk_filter,
                )
                return ToolReturn(return_value=format_search_results(result))
            except Exception as e:
                return ToolReturn(return_value=f"viking_find error ({type(e).__name__}): {e}")

        async def viking_recall(
            ctx: RunContext[Any],
            query: str,
            quotas: dict[str, int] | None = None,
            max_chars: int = 6500,
            min_score: float = 0.1,
            peer_scope: str = "all",
            other_peer_penalty: float | dict[str, float] | None = None,
        ) -> ToolReturn:
            """Recall memories from Viking across multiple context types.

            Performs multiple ``find`` calls with different context types
            and merges the results into a single formatted string.

            Valid context types are: ``memory``, ``resource``, ``skill``.
            These correspond to the three top-level namespaces in Viking:
            - ``memory``: personal memories and conversation history
            - ``resource``: ingested documents and resources
            - ``skill``: stored skill definitions

            Args:
                query: Natural-language query for memory retrieval.
                quotas: Per-context-type result limits. Valid context types
                    are ``memory``, ``resource``, ``skill``. Defaults to
                    ``{"memory": 5, "resource": 3, "skill": 2}``.
                max_chars: Maximum total characters in the output.
                min_score: Minimum relevance score (0.0 to 1.0).
                peer_scope: Scope of peers to search ("all" or "actor").
                other_peer_penalty: Penalty applied to other peers' memories.

            Returns:
                Formatted string with recalled memories grouped by context type.
            """
            try:
                client = await cap._ensure_client()
                if quotas is None:
                    quotas = {
                        "memory": 5,
                        "resource": 3,
                        "skill": 2,
                    }
                # Build filter with optional peer_scope and other_peer_penalty.
                # The SDK's find() does not natively support these parameters,
                # so we pass them through the filter dict for the server to
                # interpret if supported.
                base_filter: dict[str, Any] = {}
                if peer_scope != "all":
                    base_filter["peer_scope"] = peer_scope
                if other_peer_penalty is not None:
                    base_filter["other_peer_penalty"] = other_peer_penalty
                sdk_filter = base_filter if base_filter else None

                sections: list[str] = []
                for context_type, quota in quotas.items():
                    result = await client.find(
                        query=query,
                        context_type=context_type,
                        limit=quota,
                        score_threshold=min_score,
                        filter=sdk_filter,
                    )
                    formatted = format_search_results(result)
                    sections.append(f"=== {context_type} ===\n{formatted}")
                merged = "\n\n".join(sections)
                return ToolReturn(return_value=truncate_text(merged, max_chars))
            except Exception as e:
                return ToolReturn(return_value=f"viking_recall error ({type(e).__name__}): {e}")

        async def viking_grep(
            ctx: RunContext[Any],
            uri: str,
            pattern: str | list[str],
            case_insensitive: bool = False,
            node_limit: int = 256,
        ) -> ToolReturn:
            """Search for a regex pattern within a Viking document.

            Returns matching lines with their line numbers. Supports
            multiple patterns searched concurrently.

            Args:
                uri: Full viking:// URI of the document to search.
                pattern: Regular expression pattern (or list of patterns)
                    to match.
                case_insensitive: Whether to ignore case when matching.
                node_limit: Maximum number of nodes to scan per pattern.

            Returns:
                Matching lines grouped by URI, or "No matches found."
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_grep"):
                    return ToolReturn(return_value=err)
                patterns = [pattern] if isinstance(pattern, str) else pattern

                async def _grep_one(p: str) -> list[dict[str, Any]]:
                    try:
                        result = await client.grep(
                            uri,
                            p,
                            case_insensitive=case_insensitive,
                            node_limit=node_limit,
                        )
                        if isinstance(result, dict):
                            matches = result.get("matches", [])
                            return matches if isinstance(matches, list) else []
                        return []
                    except Exception:
                        return []

                results = await asyncio.gather(*[_grep_one(p) for p in patterns])
                all_matches: list[dict[str, Any]] = []
                for p, matches in zip(patterns, results, strict=True):
                    for m in matches:
                        if isinstance(m, dict):
                            m_copy = dict(m)
                            m_copy.setdefault("pattern", p)
                            m_copy.setdefault("uri", uri)
                            all_matches.append(m_copy)

                return ToolReturn(return_value=format_grep_results(all_matches, patterns))
            except Exception as e:
                return ToolReturn(return_value=f"viking_grep error ({type(e).__name__}): {e}")

        async def viking_glob(
            ctx: RunContext[Any],
            pattern: str,
            uri: str = "viking://",
            node_limit: int = 100,
        ) -> ToolReturn:
            """Find Viking URIs matching a glob pattern.

            Args:
                pattern: Glob pattern (e.g. ``**/*.md``).
                uri: Base URI to search from.
                node_limit: Maximum number of nodes to scan.

            Returns:
                Matching viking:// URIs, one per line.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_glob"):
                    return ToolReturn(return_value=err)
                result = await client.glob(
                    pattern,
                    uri=uri,
                    node_limit=node_limit,
                )
                if isinstance(result, dict):
                    uris = result.get("matches", [])
                else:
                    uris = result if isinstance(result, list) else []
                return ToolReturn(return_value=format_glob_results([str(u) for u in uris], pattern))
            except Exception as e:
                return ToolReturn(return_value=f"viking_glob error ({type(e).__name__}): {e}")

        async def viking_ls(
            ctx: RunContext[Any],
            uri: str = "viking://",
            recursive: bool = False,
            show_abstract: bool = False,
        ) -> ToolReturn:
            """List contents of a Viking directory.

            Args:
                uri: Full viking:// URI of the directory to list.
                recursive: Whether to list recursively into subdirectories.
                show_abstract: If True, fetch and display L0 abstract for each
                    directory. Costs extra API calls but helps judge directory
                    relevance.

            Returns:
                Entries with ``[dir]``/``[file]`` markers. When
                ``show_abstract=True``, directories include an L0 abstract
                after a dash separator.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_ls"):
                    return ToolReturn(return_value=err)
                entries = await client.ls(uri, simple=False, recursive=recursive)
                entry_list = entries if isinstance(entries, list) else []

                if show_abstract and entry_list:
                    # Fetch abstracts for directories only
                    async def _safe_abstract(entry_uri: str) -> str:
                        try:
                            return str(await client.abstract(entry_uri) or "")
                        except Exception:
                            return ""

                    abstract_uris: list[str] = []
                    abstract_tasks: list[Any] = []
                    for entry in entry_list:
                        if isinstance(entry, dict):
                            is_dir = entry.get("type") in (
                                "directory",
                                "dir",
                                "folder",
                            ) or entry.get("isDir")
                            if is_dir:
                                e_uri = str(entry.get("uri") or "")
                                if e_uri:
                                    abstract_uris.append(e_uri)
                                    abstract_tasks.append(_safe_abstract(e_uri))

                    if abstract_tasks:
                        abstracts = await asyncio.gather(*abstract_tasks)
                        abstract_map: dict[str, str] = {}
                        for e_uri, ab in zip(abstract_uris, abstracts, strict=False):
                            if isinstance(ab, str) and ab.strip():
                                abstract_map[e_uri] = ab.strip()

                        if abstract_map:
                            lines: list[str] = []
                            for entry in entry_list:
                                if isinstance(entry, dict):
                                    name = entry.get("name", entry.get("uri", "?"))
                                    entry_type = entry.get("type", "file")
                                    is_dir = entry_type in (
                                        "directory",
                                        "dir",
                                        "folder",
                                    ) or entry.get("isDir")
                                    marker = "[dir]" if is_dir else "[file]"
                                    e_uri = str(entry.get("uri") or "")
                                    ab = abstract_map.get(e_uri, "")
                                    if ab:
                                        lines.append(f"{marker} {name} \u2014 {ab}")
                                    else:
                                        lines.append(f"{marker} {name}")
                                else:
                                    lines.append(f"[file] {entry}")
                            return ToolReturn(return_value="\n".join(lines))

                return ToolReturn(return_value=format_ls_entries(entry_list))
            except Exception as e:
                return ToolReturn(return_value=f"viking_ls error ({type(e).__name__}): {e}")

        async def viking_read(
            ctx: RunContext[Any],
            uris: str | list[str],
            level: str = "read",
            line: int = 1,
            limit: int = -1,
        ) -> ToolReturn:
            """Read content from one or more Viking URIs with tiered loading.

            Args:
                uris: A single viking:// URI or a list of URIs to read.
                level: Content depth \u2014 "abstract" (L0, ~100 tokens summary),
                    "overview" (L1, ~2k tokens structure), or "read" (L2, full
                    content). Default "read" for full content. Use "abstract"
                    for quick relevance checks or "overview" for planning without
                    loading full content.
                line: Starting line number (1-indexed, only applies when
                    level="read").
                limit: Maximum number of lines to read (-1 for all, only
                    applies when level="read").

            Returns:
                File content with line number prefixes (for level="read").
                Multiple files are separated by ``=== {uri} ===`` headers.
            """
            try:
                client = await cap._ensure_client()
                uri_list = [uris] if isinstance(uris, str) else uris
                for u in uri_list:
                    if err := cap._check_uri_allowed(u, tool_name="viking_read"):
                        return ToolReturn(return_value=err)
                sections: list[str] = []
                image_parts: list[BinaryImage] = []
                for u in uri_list:
                    is_image = _is_image_resource(u)
                    suffix = PurePosixPath(u).suffix.lower()
                    # SVG is a vector format most vision APIs reject — it
                    # never enters the byte path, always degrades to a text
                    # hint, regardless of the support_vision / model caps.
                    if is_image and (not cap._should_return_image_bytes() or suffix == ".svg"):
                        # Image resource but the model can't consume image
                        # bytes (or forced text / vector SVG) — text URI hint.
                        if len(uri_list) > 1:
                            sections.append(f"=== {u} ===\n{_image_uri_hint(u)}")
                        else:
                            sections.append(_image_uri_hint(u))
                        continue

                    if is_image:
                        # Image resource and the model accepts image bytes.
                        data = await client.download_bytes(u)
                        media_type = IMAGE_MIME_TYPES.get(
                            PurePosixPath(u).suffix.lower(), "application/octet-stream"
                        )
                        image_idx = len(image_parts) + 1  # 1-based, matches content order
                        image_parts.append(BinaryImage(data=data, media_type=media_type))
                        if len(uri_list) > 1:
                            sections.append(f"=== {u} ===\n[Image #{image_idx}: {media_type}]")
                        else:
                            sections.append(f"[Image #{image_idx}: {media_type}]")
                        continue

                    if level == "abstract":
                        content = await client.abstract(u)
                    elif level == "overview":
                        content = await client.overview(u)
                    else:
                        offset = line - 1  # SDK offset is 0-indexed
                        content = await client.read(u, offset=offset, limit=limit)

                    if level == "read":
                        numbered = add_line_numbers(str(content), start_line=line)
                    else:
                        # For abstract/overview, return content without line numbers
                        numbered = str(content)

                    if len(uri_list) > 1:
                        sections.append(f"=== {u} ===\n{numbered}")
                    else:
                        sections.append(numbered)

                if image_parts:
                    # Mixed content: text sections describe each file; image
                    # bytes follow as BinaryImage parts the model can view.
                    tool_content: list[Any] = ["\n\n".join(sections)]
                    tool_content.extend(image_parts)
                    return ToolReturn(
                        return_value="\n\n".join(sections),
                        content=tool_content,
                    )
                return ToolReturn(return_value="\n\n".join(sections))
            except Exception as e:
                return ToolReturn(return_value=f"viking_read error ({type(e).__name__}): {e}")

        async def viking_expand(
            ctx: RunContext[Any],
            uri: str,
        ) -> ToolReturn:
            """Expand a previously archived conversation from Viking.

            Loads the full content of a conversation that was archived
            during context compaction. Use this when you need to recall
            details from an earlier part of the conversation that was
            summarized and archived.

            Args:
                uri: The viking:// URI of the archived conversation
                    (e.g. ``viking://user/alice/memories/compacted/abc.md``).

            Returns:
                The full archived conversation content as markdown.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_expand"):
                    return ToolReturn(return_value=err)
                content = await client.read(uri)
                return ToolReturn(
                    return_value=str(content) if content else "No content found at URI."
                )
            except Exception as e:
                return ToolReturn(return_value=f"viking_expand error ({type(e).__name__}): {e}")

        retrieve_tools: list[Callable[..., Awaitable[ToolReturn]]] = [
            viking_search,
            viking_find,
            viking_grep,
            viking_glob,
            viking_ls,
            viking_read,
        ]
        if cap.enable_memory:
            retrieve_tools.append(viking_recall)
        if cap.compaction_expand_tool:
            retrieve_tools.append(viking_expand)
        tools.extend(retrieve_tools)

    # ---- Write tools (6) ----

    if cap.mode in ("write", "all"):

        async def viking_remember(
            ctx: RunContext[Any],
            reason: str = "",
        ) -> ToolReturn:
            """Schedule the current conversation for capture into Viking memory.

            Capture is deferred: the real conversation (this turn's exchange
            plus later turns) is ingested at the end of the current model
            boundary and committed for memory extraction. No conversation
            content is passed by this tool — genuine user/assistant roles are
            taken from the session directly.

            Args:
                reason: Optional reason for remembering — recorded as an
                    intent marker so the memory extraction focuses on it.

            Returns:
                Confirmation that the capture was scheduled.
            """
            cap._remember_pending.append(reason)
            return ToolReturn(
                return_value=(
                    "Capture scheduled — the current conversation will be "
                    "ingested into Viking memory after this turn."
                )
            )

        async def viking_write(
            ctx: RunContext[Any],
            uri: str,
            content: str,
            mode: Literal["create", "replace", "append"] = "create",
        ) -> ToolReturn:
            """Write content to a Viking URI.

            URIs must be under ``memories/`` or ``resources/`` paths
            (e.g. ``viking://user/default/memories/notes.md`` or
            ``viking://resources/wiki/Device/SY215.md``). Other paths
            will be rejected by the backend.

            Args:
                uri: Full viking:// URI to write to. Must be under
                    ``memories/`` or ``resources/``.
                content: Content to write.
                mode: Write mode \u2014 ``"create"`` (default, fails if exists),
                    ``"replace"`` (overwrite), or ``"append"`` (add to end).

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_write"):
                    return ToolReturn(return_value=err)
                await client.write(uri, content, mode=mode)
                return ToolReturn(
                    return_value=f"Wrote {len(content)} chars to {uri} (mode={mode})."
                )
            except Exception as e:
                return ToolReturn(return_value=f"viking_write error ({type(e).__name__}): {e}")

        async def viking_edit(
            ctx: RunContext[Any],
            uri: str,
            old_string: str,
            new_string: str,
            replace_all: bool = False,
        ) -> ToolReturn:
            """Edit a Viking document by replacing a string.

            Uses a read-modify-write cycle: reads the current content,
            replaces ``old_string`` with ``new_string``, then writes back.
            The URI must be under ``memories/`` or ``resources/`` (same
            restriction as ``viking_write``).

            Args:
                uri: Full viking:// URI of the document to edit.
                old_string: The exact string to find and replace.
                new_string: The replacement string.
                replace_all: If ``True``, replace all occurrences. If
                    ``False``, fails if there are multiple matches.

            Returns:
                Confirmation string, or an error message if the string
                was not found or appeared multiple times.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_edit"):
                    return ToolReturn(return_value=err)
                current = await client.read(uri)
                count = current.count(old_string)
                if count == 0:
                    return ToolReturn(
                        return_value=f"viking_edit error: old_string not found in {uri}."
                    )
                if count > 1 and not replace_all:
                    return ToolReturn(
                        return_value=(
                            f"viking_edit error: old_string found {count} times in {uri}. "
                            "Use replace_all=True to replace all occurrences."
                        )
                    )
                if replace_all:
                    modified = current.replace(old_string, new_string)
                else:
                    modified = current.replace(old_string, new_string, 1)
                await client.write(uri, modified, mode="replace")
                return ToolReturn(return_value=f"Replaced {count} occurrence(s) in {uri}.")
            except Exception as e:
                return ToolReturn(return_value=f"viking_edit error ({type(e).__name__}): {e}")

        async def viking_mkdir(
            ctx: RunContext[Any],
            uri: str,
            description: str | None = None,
        ) -> ToolReturn:
            """Create a directory in the Viking knowledge graph.

            Args:
                uri: Full viking:// URI of the directory to create.
                description: Optional description for the directory.

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_mkdir"):
                    return ToolReturn(return_value=err)
                await client.mkdir(uri, description=description)
                return ToolReturn(return_value=f"Created directory {uri}.")
            except Exception as e:
                return ToolReturn(return_value=f"viking_mkdir error ({type(e).__name__}): {e}")

        async def viking_add_resource(
            ctx: RunContext[Any],
            path: str,
            to: str | None = None,
            parent: str | None = None,
            processing_mode: str | None = None,
            watch_interval: float = 0,
        ) -> ToolReturn:
            """Add an external resource to the Viking knowledge graph.

            Ingests a local file or directory into the graph, making it
            searchable and linkable. The ``to`` target must be under
            ``viking://resources/`` (e.g. ``viking://resources/wiki/``).

            Args:
                path: Local file or directory path to ingest.
                to: Target viking:// URI under ``resources/`` to store the resource.
                parent: Parent viking:// URI under ``resources/`` for nesting.
                processing_mode: Processing mode for the resource (unused by
                    current SDK \u2014 kept for API compatibility).
                watch_interval: Watch interval in seconds (0 = no watch).

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                for target, label in ((to, "to"), (parent, "parent")):
                    if target and (
                        err := cap._check_uri_allowed(target, tool_name="viking_add_resource")
                    ):
                        return ToolReturn(return_value=f"{err} ({label})")
                # SDK add_resource() does not accept processing_mode;
                # pass only supported kwargs.
                result = await client.add_resource(
                    path,
                    to=to,
                    parent=parent,
                    watch_interval=watch_interval,
                )
                return ToolReturn(return_value=f"Added resource {path} to Viking. Result: {result}")
            except Exception as e:
                return ToolReturn(
                    return_value=f"viking_add_resource error ({type(e).__name__}): {e}"
                )

        async def viking_forget(
            ctx: RunContext[Any],
            uri: str,
            recursive: bool = False,
        ) -> ToolReturn:
            """Remove a document or directory from Viking.

            Args:
                uri: Full viking:// URI to remove.
                recursive: If ``True``, remove directories recursively.

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_forget"):
                    return ToolReturn(return_value=err)
                await client.rm(uri, recursive=recursive)
                return ToolReturn(return_value=f"Removed {uri}.")
            except Exception as e:
                return ToolReturn(return_value=f"viking_forget error ({type(e).__name__}): {e}")

        write_tools: list[Callable[..., Awaitable[ToolReturn]]] = [
            viking_write,
            viking_edit,
            viking_mkdir,
            viking_add_resource,
        ]
        if cap.enable_forget:
            write_tools.append(viking_forget)
        if cap.enable_memory:
            write_tools.append(viking_remember)
        tools.extend(write_tools)

    # ---- Graph tools (2) ----

    if cap.mode in ("graph", "all"):

        async def viking_link(
            ctx: RunContext[Any],
            from_uri: str,
            to_uris: str | list[str],
            reason: str = "",
        ) -> ToolReturn:
            """Create a link between nodes in the Viking knowledge graph.

            Both ``from_uri`` and all ``to_uris`` must point to existing
            nodes. The backend rejects links to non-existent nodes.

            Args:
                from_uri: Source viking:// URI. Must exist.
                to_uris: Target viking:// URI or list of URIs. All must exist.
                reason: Optional reason/label for the link.

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(from_uri, tool_name="viking_link"):
                    return ToolReturn(return_value=f"{err} (from_uri)")
                targets = to_uris if isinstance(to_uris, list) else [to_uris]
                for t in targets:
                    if err := cap._check_uri_allowed(t, tool_name="viking_link"):
                        return ToolReturn(return_value=f"{err} (to_uris)")
                await client.link(from_uri, to_uris, reason=reason)
                return ToolReturn(
                    return_value=f"Linked {from_uri} -> {', '.join(targets)} (reason: {reason!r})."
                )
            except Exception as e:
                return ToolReturn(return_value=f"viking_link error ({type(e).__name__}): {e}")

        async def viking_set_tags(
            ctx: RunContext[Any],
            uri: str,
            tags: list[str],
            recursive: bool = False,
        ) -> ToolReturn:
            """Set tags on a Viking node.

            Args:
                uri: Full viking:// URI of the node to tag.
                tags: List of ``"key=value"`` tag strings.
                recursive: If ``True``, apply tags to all children recursively.

            Returns:
                Confirmation string.
            """
            try:
                client = await cap._ensure_client()
                if err := cap._check_uri_allowed(uri, tool_name="viking_set_tags"):
                    return ToolReturn(return_value=err)
                await client.set_tags(uri, tags, mode="replace", recursive=recursive)
                return ToolReturn(return_value=f"Set {len(tags)} tag(s) on {uri}.")
            except Exception as e:
                return ToolReturn(return_value=f"viking_set_tags error ({type(e).__name__}): {e}")

        graph_tools: list[Callable[..., Awaitable[ToolReturn]]] = [viking_set_tags]
        if cap.enable_link:
            graph_tools.append(viking_link)
        tools.extend(graph_tools)

    if cap.enabled_tools is not None:
        names = {getattr(fn, "__name__", "") for fn in tools}
        allowed = {n for n in cap.enabled_tools if n in names}
        tools = [fn for fn in tools if getattr(fn, "__name__", "") in allowed]
    elif cap.disabled_tools is not None:
        tools = [fn for fn in tools if getattr(fn, "__name__", "") not in set(cap.disabled_tools)]

    return tools
