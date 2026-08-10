"""Helper functions for formatting and validating Viking SDK results.

These utilities are used by the tool functions in ``tools.py`` to format
SDK responses into human-readable strings and to perform common text
operations like line-numbering and truncation.
"""

from __future__ import annotations

from typing import Any


def format_search_results(results: dict[str, Any] | list[Any]) -> str:
    r"""Format SDK search/find results matching official MCP output.

    Groups hits by context type (memory, resource, skill) and formats
    each as ``- [type score%] viking://uri\n    abstract``, matching
    the official OpenViking MCP endpoint layout.

    Args:
        results: A dict (with Viking's grouped keys ``memories``/
            ``resources``/``skills``, or flat ``hits``/``results``) or
            a list of hits.

    Returns:
        A formatted multi-line string with a header, grouped hits, and
        a footer prompting the user to use the read tool.
    """
    # Map SDK plural keys to singular display names
    _ctx_map = {
        "memories": "memory",
        "resources": "resource",
        "skills": "skill",
    }

    groups: list[tuple[str, list[Any]]] = []

    if isinstance(results, dict):
        for plural, singular in _ctx_map.items():
            hits = results.get(plural)
            if hits and isinstance(hits, list):
                groups.append((singular, hits))
        # Fallback: flat hits/results list
        if not groups:
            flat = results.get("hits") or results.get("results")
            if flat and isinstance(flat, list):
                groups.append(("result", flat))
    elif isinstance(results, list):
        if results:
            groups.append(("result", results))

    total = sum(len(hits) for _, hits in groups)
    if total == 0:
        return "No matching context found."

    lines = [f"Found {total} item(s):", ""]
    for ctx_type, hits in groups:
        for hit in hits:
            if isinstance(hit, dict):
                uri = hit.get("uri", hit.get("path", "?"))
                score = hit.get("score", hit.get("similarity", 0.0))
                abstract = hit.get("abstract", "") or hit.get("overview", "") or "(no abstract)"
                score_pct = f"{score * 100:.0f}%" if isinstance(score, (int, float)) else str(score)
                ab = str(abstract).strip()
                lines.append(f"- [{ctx_type} {score_pct}] {uri}")
                lines.append(f"    {ab}")
            else:
                lines.append(f"- [{ctx_type}] {hit}")
    lines.append("")
    lines.append("Use the read tool to expand a URI.")
    return "\n".join(lines)


def format_grep_results(
    matches: list[dict[str, Any]],
    patterns: list[str],
) -> str:
    """Format grep matches matching official MCP output.

    Groups matches by URI and formats each as:
    ``L{line} [{pattern}]: {content}``.

    Args:
        matches: List of match dicts with ``uri``, ``line``, ``content``,
            and ``pattern`` keys.
        patterns: The list of patterns that were searched.

    Returns:
        A formatted multi-line string with a header and grouped matches.
    """
    if not matches:
        return f"No matches found for pattern(s): {', '.join(patterns)}"

    merged: dict[str, list[tuple[Any, str, str]]] = {}
    total = 0
    for m in matches:
        m_uri = m.get("uri", "?")
        m_line = m.get("line", "?")
        m_content = m.get("content", m.get("text", ""))
        m_pattern = m.get("pattern", patterns[0] if patterns else "")
        merged.setdefault(m_uri, []).append((m_line, m_content, m_pattern))
        total += 1

    lines = [
        f"Found {total} match(es) across {len(patterns)} pattern(s):",
    ]
    for m_uri, hits in merged.items():
        hits.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
        lines.append(f"\n{m_uri}")
        for line_no, content, p in hits:
            lines.append(f"  L{line_no} [{p}]: {content}")
    return "\n".join(lines)


def format_glob_results(uris: list[str], pattern: str) -> str:
    """Format glob results matching official MCP output.

    Args:
        uris: List of matching URI strings.
        pattern: The glob pattern that was searched.

    Returns:
        A formatted multi-line string with a header and URI list.
    """
    if not uris:
        return f"No files found matching: {pattern}"

    lines = [f"Found {len(uris)} file(s):"]
    for u in uris:
        # Matches may be plain URI strings or dicts with "uri" key
        uri = u.get("uri", str(u)) if isinstance(u, dict) else str(u)
        lines.append(f"  {uri}")
    return "\n".join(lines)


def format_ls_entries(entries: list[Any]) -> str:
    """Format ls results with ``[dir]``/``[file]`` markers.

    Args:
        entries: A list of entry dicts (with ``name``, ``type``, ``uri`` keys)
            or plain strings.

    Returns:
        A formatted string with one entry per line, prefixed with
        ``[dir]`` or ``[file]``.
    """
    if not entries:
        return "(empty)"

    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("name", entry.get("uri", "?"))
            entry_type = entry.get("type", "file")
            marker = "[dir]" if entry_type in ("directory", "dir", "folder") else "[file]"
            lines.append(f"{marker} {name}")
        else:
            lines.append(f"[file] {entry}")
    return "\n".join(lines)


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """Add line number prefixes like ``  1│ content`` to each line.

    Args:
        content: The text content to number.
        start_line: The line number for the first line (1-indexed).

    Returns:
        The content with line number prefixes.
    """
    lines = content.splitlines()
    if not lines:
        return ""
    # Calculate width for alignment based on the largest line number
    width = len(str(start_line + len(lines) - 1))
    formatted: list[str] = []
    for i, line in enumerate(lines):
        num = start_line + i
        formatted.append(f"{num:>{width}}\u2502 {line}")
    return "\n".join(formatted)


def is_viking_uri(uri: str) -> bool:
    """Check if a URI starts with ``viking://``.

    Args:
        uri: The URI string to check.

    Returns:
        ``True`` if the URI starts with ``viking://``, ``False`` otherwise.
    """
    return uri.startswith("viking://")


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` with an ellipsis indicator.

    If the text is shorter than ``max_chars``, it is returned unchanged.
    Otherwise, it is truncated and a ``[... truncated N chars]`` suffix
    is appended.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep.

    Returns:
        The (possibly truncated) text.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    removed = len(text) - max_chars
    return f"{truncated}\n[... truncated {removed} chars]"
