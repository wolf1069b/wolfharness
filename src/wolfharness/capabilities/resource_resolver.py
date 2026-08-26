"""Shared resource resolution utility — single entry point for reading resource content by URI.

Both the OpenCode converter (``_resolve_resource()``) and ``ResourceCapability.read_resource()``
tool delegate to ``resolve_resource_content()`` to avoid logic duplication.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import logfire
from pydantic_ai import BinaryContent

from wolfharness.capabilities.resource_protocols import (
    BlobResourceContent,
    TextResourceContent,
    UriSchemeMismatchError,
)
from wolfharness.skills.uri_resolver import ResolvedSkillURI, _name_alternatives


if TYPE_CHECKING:
    from pydantic_ai.messages import UserContent

    from wolfharness.capabilities.resource_protocols import ResourceAccess, SkillResource
    from wolfharness.capabilities.uri_scheme_registry import UriSchemeRegistry


# Default maximum text characters per resource read before truncation.
_DEFAULT_MAX_TEXT_CHARS = 10_000


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` if needed, appending a truncation suffix.

    Args:
        text: The text to potentially truncate.
        max_chars: Maximum number of characters to keep.

    Returns:
        The original text if within limit, or a truncated version with suffix.
    """
    if len(text) <= max_chars:
        return text
    suffix = (
        f"\n\n... [truncated: {len(text)} chars total, showing first {max_chars}. "
        f"Use a narrower resource URI (e.g. a chapter or chunk URI) to read "
        f"a specific section, or a paginated read tool for full content.]"
    )
    return text[:max_chars] + suffix


def _extract_skill_name(uri: str) -> str:
    """Extract the skill name from a ``skill://`` URI.

    Takes the first path segment after ``skill://``.

    Args:
        uri: A ``skill://`` URI.

    Returns:
        The skill name (first path segment).
    """
    path = uri[len("skill://") :]
    return path.split("/")[0] if path else ""


async def _resolve_skill_reference(
    skill_caps: list[SkillResource],
    skill_name: str,
    reference_path: str,
) -> str | None:
    """Resolve a skill reference file by looking up the skill's filesystem path.

    Iterates ``SkillResource`` providers, finds the ``SkillEntry`` matching
    ``skill_name``, and reads the reference file from the skill's ``skill_path``
    directory on disk.

    Args:
        skill_caps: List of ``SkillResource`` providers to query.
        skill_name: The skill name to look up.
        reference_path: Relative path to the reference file within the skill directory.

    Returns:
        The reference file content as a string, or ``None`` if the skill or
        reference file cannot be found.
    """
    from upathtools import UPath

    from wolfharness.skills.exceptions import SecurityError

    # Defense-in-depth: ResolvedSkillURI.parse() already rejects ".." segments,
    # but we double-check here in case this helper is called directly.
    if ".." in reference_path.split("/"):
        raise SecurityError(f"Path traversal detected in reference path: {reference_path}")

    # Try exact name and underscore/hyphen alternatives
    candidate_names = [skill_name, *_name_alternatives(skill_name)]

    for skill_cap in skill_caps:
        try:
            entries = await skill_cap.list_skills()
        except Exception:  # noqa: BLE001
            continue

        for entry in entries:
            if entry.name not in candidate_names:
                continue

            # Only filesystem skills (UPath) can have reference files read from disk.
            # PurePosixPath or None means virtual/remote skill — cannot read files.
            skill_path = entry.skill_path
            if not isinstance(skill_path, UPath):
                continue

            ref_file = skill_path / reference_path

            # Resolve and verify the path is within the skill directory
            try:
                resolved_ref = ref_file.resolve()
                resolved_skill = skill_path.resolve()
                if not str(resolved_ref).startswith(str(resolved_skill)):
                    raise SecurityError(f"Reference path escapes skill directory: {reference_path}")
            except (OSError, ValueError):
                continue

            if not ref_file.exists():
                continue

            return ref_file.read_text(encoding="utf-8")

    return None


def _filter_by_client_name(
    resource_caps: list[ResourceAccess],
    client_name: str,
) -> list[ResourceAccess] | None:
    """Filter resource caps to only those matching ``client_name``.

    Returns a filtered list, or ``None`` if no caps matched.
    """
    identified_caps = [
        resource_cap
        for resource_cap in resource_caps
        if getattr(resource_cap, "server_name", None) is not None
    ]
    if not identified_caps:
        return resource_caps
    selected_caps = [
        resource_cap
        for resource_cap in identified_caps
        if getattr(resource_cap, "server_name", None) == client_name
    ]
    return selected_caps or None


async def _resolve_skill_uri(
    uri: str,
    skill_caps: list[SkillResource],
    max_text_chars: int,
) -> list[UserContent] | None:
    """Resolve a ``skill://`` URI and return its content.

    Handles both reference files (``skill://name/path/to/file.md``) and
    main skill content (``skill://name``, which reads SKILL.md).

    Args:
        uri: The ``skill://`` URI to resolve.
        skill_caps: List of ``SkillResource`` providers.
        max_text_chars: Maximum text characters before truncation.

    Returns:
        Content items if the skill was found, or ``None`` if not.
    """
    if not uri.startswith("skill://"):
        return None
    resolved = ResolvedSkillURI.parse(uri)
    skill_name = resolved.skill_name

    # If the URI contains a reference path, read the reference file.
    # Exception: "SKILL.md" (case-insensitive) is the skill's main file —
    # use read_skill() for backward compatibility and virtual skill support.
    if resolved.reference_path is not None and resolved.reference_path.upper() != "SKILL.MD":
        try:
            ref_content = await _resolve_skill_reference(
                skill_caps, skill_name, resolved.reference_path
            )
        except Exception:  # noqa: BLE001
            logfire.exception(
                "Failed to read skill reference '{skill_name}/{ref}'",
                skill_name=skill_name,
                ref=resolved.reference_path,
            )
            return None
        if ref_content is not None:
            truncated = _truncate_text(ref_content, max_text_chars)
            return [f'<resource uri="{uri}">\n{truncated}\n</resource>']
        return None

    # No reference path — read SKILL.md content
    for skill_cap in skill_caps:
        try:
            content = await skill_cap.read_skill(skill_name)
        except Exception:  # noqa: BLE001
            logfire.exception(
                "Failed to read skill '{skill_name}' from {cap}",
                skill_name=skill_name,
                cap=type(skill_cap).__name__,
            )
            continue
        if content is None:
            continue
        truncated = _truncate_text(content, max_text_chars)
        return [f'<resource uri="{uri}">\n{truncated}\n</resource>']
    return None


@logfire.instrument("capability.resource_resolver.resolve")
async def resolve_resource_content(
    uri: str,
    resource_caps: list[ResourceAccess],
    skill_caps: list[SkillResource],
    *,
    max_text_chars: int = _DEFAULT_MAX_TEXT_CHARS,
    client_name: str | None = None,
    scheme_registry: UriSchemeRegistry | None = None,
) -> list[UserContent] | None:
    """Resolve a resource URI and return its content as ``UserContent`` items.

    Routes by URI scheme:
        - ``skill://skill-name`` → ``SkillResource.read_skill()`` (SKILL.md content)
        - ``skill://skill-name/references/file.md`` → reads the reference file
          from the skill's filesystem directory
        - Other URIs → ``ResourceAccess`` providers (``read_resource()``)

    When ``scheme_registry`` is provided, URIs with a registered scheme are
    routed directly to the authoritative provider (deterministic, O(1)).
    Unregistered schemes fall back to opaque providers (empty ``owned_schemes``).
    When ``scheme_registry`` is ``None``, falls back to the legacy iteration
    over all providers.

    Args:
        uri: The resource URI to resolve.
        resource_caps: List of ``ResourceAccess`` providers to query for non-skill URIs.
        skill_caps: List of ``SkillResource`` providers to query for ``skill://`` URIs.
        max_text_chars: Maximum text characters before truncation.
        client_name: Optional exact MCP server identifier for host-injected resources.
        scheme_registry: Optional ``UriSchemeRegistry`` for scheme-based routing.

    Returns:
        A list of ``UserContent`` items (strings and/or ``BinaryContent``) if the
        resource was found, or ``None`` if no provider could resolve the URI.
    """
    # ---- skill:// routing ----
    skill_result = await _resolve_skill_uri(uri, skill_caps, max_text_chars)
    if skill_result is not None:
        return skill_result
    if uri.startswith("skill://"):
        return None

    # ---- Base provider filtering ----
    # A connected MCP capability can still provide tools when its initialize
    # handshake explicitly omitted ``resources``.  Keep that server out of
    # Host ResourceSource routing while leaving generic legacy providers
    # (which have no negotiated state) untouched.
    selected_caps = [
        resource_cap
        for resource_cap in resource_caps
        if getattr(resource_cap, "resources_supported", None) is not False
    ]

    # ---- client_name filtering ----
    if client_name is not None:
        filtered = _filter_by_client_name(selected_caps, client_name)
        if filtered is None:
            return None
        selected_caps = filtered

    # ---- Scheme-based routing (via UriSchemeRegistry) ----
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    scheme = parsed.scheme

    if scheme and scheme_registry is not None:
        provider = scheme_registry.lookup(scheme)
        if provider is not None and (client_name is None or provider in selected_caps):
            try:
                contents = await provider.read_resource(uri)
            except UriSchemeMismatchError:
                logfire.warning(
                    "Provider '{name}' rejected URI '{uri}' (scheme mismatch)",
                    name=getattr(provider, "server_name", type(provider).__name__),
                    uri=uri,
                )
                return None
            except Exception:  # noqa: BLE001
                logfire.exception(
                    "Failed to read resource '{uri}' from {cap}",
                    uri=uri,
                    cap=type(provider).__name__,
                )
                return None
            if contents:
                return _convert_resource_parts(uri, contents, max_text_chars)
            return None

    # ---- Fallback for unregistered schemes ----
    # If the URI has a scheme but no registered owner, try only opaque
    # providers (those with empty owned_schemes).  If the URI has no scheme
    # at all, try all providers (backward-compatible behavior).
    if scheme:
        selected_caps = [cap for cap in selected_caps if not cap.owned_schemes]

    # ---- Legacy iteration over remaining providers ----
    for resource_cap in selected_caps:
        try:
            contents = await resource_cap.read_resource(uri)
        except Exception:  # noqa: BLE001
            logfire.exception(
                "Failed to read resource '{uri}' from {cap}",
                uri=uri,
                cap=type(resource_cap).__name__,
            )
            continue
        if contents is None:
            continue
        if not contents:
            continue
        parts = _convert_resource_parts(uri, contents, max_text_chars)
        if parts:
            return parts

    return None


def _convert_resource_parts(
    uri: str,
    contents: list[TextResourceContent | BlobResourceContent],
    max_text_chars: int,
) -> list[UserContent] | None:
    """Convert resource content blocks to ``UserContent`` items.

    Args:
        uri: The resource URI (for attribution in text wrappers).
        contents: List of resource content blocks.
        max_text_chars: Maximum text characters before truncation.

    Returns:
        A list of ``UserContent`` items, or ``None`` if empty.
    """
    parts: list[UserContent] = []
    for c in contents:
        if isinstance(c, TextResourceContent):
            truncated = _truncate_text(c.text, max_text_chars)
            parts.append(f'<resource uri="{uri}">\n{truncated}\n</resource>')
        elif isinstance(c, BlobResourceContent):
            decoded = base64.b64decode(c.blob)
            media_type = c.mime_type or "application/octet-stream"
            parts.append(f'<resource uri="{uri}">\n')
            parts.append(BinaryContent(data=decoded, media_type=media_type))
            parts.append("\n</resource>")
    return parts or None
