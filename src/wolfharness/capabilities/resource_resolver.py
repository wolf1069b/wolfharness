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
)
from wolfharness.skills.uri_resolver import ResolvedSkillURI, _name_alternatives


if TYPE_CHECKING:
    from pydantic_ai.messages import UserContent

    from wolfharness.capabilities.resource_protocols import ResourceAccess, SkillResource


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
    suffix = f"\n\n... [truncated: {len(text)} chars total, showing first {max_chars}]"
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


@logfire.instrument("capability.resource_resolver.resolve")
async def resolve_resource_content(
    uri: str,
    resource_caps: list[ResourceAccess],
    skill_caps: list[SkillResource],
    *,
    max_text_chars: int = 10_000,
    client_name: str | None = None,
) -> list[UserContent] | None:
    """Resolve a resource URI and return its content as ``UserContent`` items.

    Routes by URI scheme:
        - ``skill://skill-name`` → ``SkillResource.read_skill()`` (SKILL.md content)
        - ``skill://skill-name/references/file.md`` → reads the reference file
          from the skill's filesystem directory
        - Other URIs → ``ResourceAccess`` providers (``read_resource()``)

    Args:
        uri: The resource URI to resolve.
        resource_caps: List of ``ResourceAccess`` providers to query for non-skill URIs.
        skill_caps: List of ``SkillResource`` providers to query for ``skill://`` URIs.
        max_text_chars: Maximum text characters before truncation.
        client_name: Optional exact MCP server identifier for host-injected resources.

    Returns:
        A list of ``UserContent`` items (strings and/or ``BinaryContent``) if the
        resource was found, or ``None`` if no provider could resolve the URI.
    """
    # ---- skill:// routing ----
    if uri.startswith("skill://"):
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

    # ---- Other URI schemes → ResourceAccess providers ----
    # A connected MCP capability can still provide tools when its initialize
    # handshake explicitly omitted ``resources``.  Keep that server out of
    # Host ResourceSource routing while leaving generic legacy providers
    # (which have no negotiated state) untouched.
    selected_caps = [
        resource_cap
        for resource_cap in resource_caps
        if getattr(resource_cap, "resources_supported", None) is not False
    ]
    if client_name is not None:
        identified_caps = [
            resource_cap
            for resource_cap in resource_caps
            if getattr(resource_cap, "server_name", None) is not None
        ]
        if identified_caps:
            selected_caps = [
                resource_cap
                for resource_cap in identified_caps
                if getattr(resource_cap, "server_name", None) == client_name
            ]
            if not selected_caps:
                return None

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
        if parts:
            return parts

    return None
