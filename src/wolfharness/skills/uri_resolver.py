"""URI resolver for the skill:// scheme.

Provides secure parsing and resolution of skill URIs with support for:
- Flat URIs: skill://skill-name or skill://skill-name/reference/path
- Bare skill names: skill-name

Security features:
- Path traversal detection (rejects ".." in paths)
- Null byte detection
- Skill name validation (follows Skill model rules)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
import unicodedata
from urllib.parse import unquote, urlparse

from wolfharness.skills.exceptions import SecurityError, SkillNotFoundError


if TYPE_CHECKING:
    from wolfharness.capabilities.resource_protocols import (
        SkillEntry,
        SkillResource,
    )

from wolfharness.skills.skill import Skill


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedSkillURI:
    """A resolved and validated skill:// URI.

    Supports flat URI format only (D9):
    - ``skill://my-skill`` — bare skill reference
    - ``skill://my-skill/path/to/reference.md`` — skill with reference path
    - ``my-skill`` — bare skill name (no scheme)

    Attributes:
        skill_name: The validated skill name
        reference_path: Optional path to reference file within skill

    Examples:
        >>> ResolvedSkillURI.parse("skill://python-expert")
        ResolvedSkillURI(skill_name='python-expert', reference_path=None)

        >>> ResolvedSkillURI.parse("skill://python-expert/references/guide.md")
        ResolvedSkillURI(skill_name='python-expert', reference_path='references/guide.md')

        >>> ResolvedSkillURI.parse("python-expert")
        ResolvedSkillURI(skill_name='python-expert', reference_path=None)
    """

    skill_name: str
    reference_path: str | None

    @classmethod
    def parse(cls, uri: str) -> ResolvedSkillURI:
        """Parse and validate a skill URI.

        Supports two formats:
        1. ``skill://skill-name`` — Flat URI (D9). Netloc is the skill name,
           optional path is the reference (``skill://my-skill/path/to/ref.md``).
        2. ``skill-name`` — Bare skill name (no scheme).

        Args:
            uri: The URI to parse

        Returns:
            ResolvedSkillURI with validated components

        Raises:
            SecurityError: If path traversal, null bytes, or invalid characters detected
            ValueError: If URI format is invalid

        Examples:
            >>> ResolvedSkillURI.parse("skill://my-skill")
            ResolvedSkillURI(skill_name='my-skill', reference_path=None)

            >>> ResolvedSkillURI.parse("my-skill")
            ResolvedSkillURI(skill_name='my-skill', reference_path=None)
        """
        # Check for null bytes first
        if "\x00" in uri:
            msg = "URI contains null bytes"
            raise SecurityError(msg)

        # Check if it's a bare skill name (no scheme)
        # Note: We check the original URI before decoding to avoid misinterpreting
        # URL-encoded characters as scheme separators
        if "://" not in uri:
            # URL decode the bare skill name for validation
            decoded_name = unquote(uri)
            if "\x00" in decoded_name:
                msg = "URI contains null bytes after decoding"
                raise SecurityError(msg)
            skill_name = _validate_skill_name(decoded_name)
            return cls(skill_name=skill_name, reference_path=None)

        # Parse the URI first (before decoding) to correctly extract components
        parsed = urlparse(uri)

        # Validate scheme
        if parsed.scheme != "skill":
            msg = f"Invalid URI scheme: {parsed.scheme!r}, expected 'skill'"
            raise ValueError(msg)

        # Flat URI (D9): skill://skill-name or skill://skill-name/reference/path
        # netloc is ALWAYS the skill name — provider segment was removed in D9.
        if not parsed.netloc:
            msg = "URI is empty"
            raise ValueError(msg)

        skill_name = _validate_skill_name(unquote(parsed.netloc))

        # Parse and decode the path component as an optional reference
        path = unquote(parsed.path)
        if path.startswith("/"):
            path = path[1:]  # Remove leading slash

        if not path:
            return cls(skill_name=skill_name, reference_path=None)

        # Path content is the reference — check for traversal
        for part in path.split("/"):
            if part == "..":
                msg = f"Path traversal detected in URI: {uri!r}"
                raise SecurityError(msg)

        return cls(
            skill_name=skill_name,
            reference_path=path,
        )


# ---- Skill Name Validation ----


def _validate_skill_name(name: str) -> str:
    """Validate a skill name following Agent Skills Spec.

    Skill names must:
    - Be lowercase
    - Be alphanumeric with hyphens only (kebab-case)
    - Not start or end with hyphen
    - Not contain consecutive hyphens
    - Be non-empty

    Underscores are automatically normalized to hyphens per spec.

    Args:
        name: Skill name to validate

    Returns:
        Normalized skill name (underscores replaced with hyphens)

    Raises:
        SecurityError: If skill name is invalid
    """
    # Normalize unicode
    normalized = unicodedata.normalize("NFKC", name.strip())

    # Check for null bytes BEFORE normalization (security-sensitive check)
    if "\x00" in normalized:
        msg = "Skill name contains null bytes"
        raise SecurityError(msg)

    # Normalize underscores to hyphens per Agent Skills Spec (kebab-case).
    # The spec mandates "lowercase letters, numbers, and hyphens only".
    normalized = normalized.replace("_", "-")

    if not normalized:
        msg = "Skill name must be non-empty"
        raise SecurityError(msg)

    if normalized != normalized.lower():
        msg = f"Skill name {normalized!r} must be lowercase"
        raise SecurityError(msg)

    if normalized.startswith("-") or normalized.endswith("-"):
        msg = "Skill name cannot start or end with a hyphen"
        raise SecurityError(msg)

    if "--" in normalized:
        msg = "Skill name cannot contain consecutive hyphens"
        raise SecurityError(msg)

    if not all(c.isalnum() or c == "-" for c in normalized):
        msg = (
            f"Skill name {normalized!r} contains invalid characters. "
            "Only lowercase letters, digits, and hyphens are allowed."
        )
        raise SecurityError(msg)

    return normalized


def _name_alternatives(name: str) -> list[str]:
    """Generate alternative skill names by swapping - and _.

    MCP servers (e.g., FastMCP) use directory names as-is for skill
    identifiers, which may contain underscores. Models calling load_skill
    often use kebab-case by convention. This function generates the
    alternative form so the resolver can find the skill regardless of
    which convention the caller uses.

    Args:
        name: The original skill name

    Returns:
        List of alternative names (empty if name has no - or _)
    """
    if "_" in name:
        return [name.replace("_", "-")]
    if "-" in name:
        return [name.replace("-", "_")]
    return []


# Type alias: providers implement SkillResource protocol.
if TYPE_CHECKING:
    ProviderLike = SkillResource
else:
    ProviderLike = Any


class SkillURIResolver:
    """Resolver for skill:// URIs using registered providers.

    Manages a registry of resource providers and resolves skill URIs
    to actual skill instances. Providers implement the ``SkillResource``
    protocol (with ``list_skills()``/``read_skill()``).

    When an ``ExtensionRegistry`` is provided, URI resolution is
    delegated to ``ExtensionRegistry.resolve_uri()`` instead of
    using the internal ``_providers`` dict.

    Example:
        >>> resolver = SkillURIResolver()
        >>> resolver.register_provider("local", local_provider)
        >>> skill = await resolver.resolve("skill://local/python-expert")
    """

    def __init__(
        self,
        extension_registry: Any | None = None,
    ) -> None:
        """Initialize the resolver with an empty provider registry.

        Args:
            extension_registry: Optional ``ExtensionRegistry`` for
                URI resolution. When set, ``resolve()`` delegates to
                ``extension_registry.resolve_uri()`` instead of using
                the internal ``_providers`` dict.
        """
        self._providers: dict[str, ProviderLike] = {}
        self._extension_registry = extension_registry

    def register_provider(self, name: str, provider: ProviderLike) -> None:
        """Register a resource provider.

        Args:
            name: Internal provider name (used as dict key)
            provider: The resource provider instance (SkillResource)
        """
        self._providers[name] = provider

    def unregister_provider(self, name: str) -> None:
        """Unregister a resource provider.

        Args:
            name: Provider name to unregister
        """
        self._providers.pop(name, None)

    async def _find_skill_in_providers(
        self, skill_name: str, ref_path: str | None = None
    ) -> Skill | None:
        """Search all providers for a skill by name.

        Handles ``SkillResource`` providers (with ``list_skills()``/``read_skill()``).
        For ``SkillResource`` providers, constructs a lightweight
        ``Skill`` object from the ``SkillEntry`` metadata and content.

        Args:
            skill_name: The skill name to search for.
            ref_path: Optional reference path to store on the skill if found.

        Returns:
            The matching Skill or None if not found.
        """
        from wolfharness.capabilities.resource_protocols import SkillResource

        for provider in self._providers.values():
            # Check for new SkillResource protocol first.
            if isinstance(provider, SkillResource):
                try:
                    entries = await provider.list_skills()
                except Exception:
                    logger.warning(
                        "Failed to list skills from provider",
                        exc_info=True,
                    )
                    continue
                for entry in entries:
                    if entry.name == skill_name:
                        return await self._build_skill_from_entry(provider, entry, ref_path)
                # Try name alternatives (underscore ↔ hyphen).
                for alt_name in _name_alternatives(skill_name):
                    for entry in entries:
                        if entry.name == alt_name:
                            return await self._build_skill_from_entry(provider, entry, ref_path)
                continue
        return None

    async def _build_skill_from_entry(
        self,
        provider: SkillResource,
        entry: SkillEntry,
        ref_path: str | None,
    ) -> Skill | None:
        """Construct a lightweight Skill from a SkillEntry.

        Reads skill content from the given provider's ``read_skill()``
        method and constructs a ``Skill`` object with the metadata and
        content.

        Args:
            provider: The SkillResource that owns this entry.
            entry: The SkillEntry with metadata.
            ref_path: Optional reference path to attach.

        Returns:
            A Skill object, or None if content could not be read.
        """
        try:
            content = await provider.read_skill(entry.name)
        except Exception:  # noqa: BLE001
            return None
        if content is None:
            return None

        skill = Skill(
            name=entry.name,
            description=entry.description or f"Skill {entry.name}",
            skill_path=PurePosixPath(entry.uri),
            instructions=content,
        )
        if ref_path is not None:
            skill.resolved_reference_path = ref_path
        return skill

    async def _find_skill_with_alternatives(self, skill_name: str) -> Skill | None:
        """Search all providers for a skill, trying name alternatives.

        Args:
            skill_name: The skill name to search for.

        Returns:
            The matching Skill or None if not found.
        """
        skill = await self._find_skill_in_providers(skill_name)
        if skill is not None:
            return skill
        for alt_name in _name_alternatives(skill_name):
            skill = await self._find_skill_in_providers(alt_name)
            if skill is not None:
                return skill
        return None

    async def resolve(self, uri: str) -> Skill:
        """Resolve a skill URI to a Skill instance.

        Tries ``ExtensionRegistry`` first when configured. If it fails,
        falls back to searching all registered providers for the skill name.

        Supports flat URIs (D9):
        - ``skill://my-skill`` — by name
        - ``skill://my-skill/reference/path`` — by name (reference attached)
        - ``my-skill`` — bare skill name

        Args:
            uri: The skill URI to resolve

        Returns:
            The resolved Skill instance

        Raises:
            SecurityError: If URI validation fails
            SkillNotFoundError: If skill not found in any provider
        """
        # Delegate to ExtensionRegistry when available.
        if self._extension_registry is not None:
            from wolfharness.capabilities.extension_registry import Scope, ScopeLevel

            result = await self._extension_registry.resolve_uri(uri, Scope(level=ScopeLevel.POOL))
            if result is not None:
                if isinstance(result, Skill):
                    return result
                # str | bytes content — construct Skill with virtual path.
                return Skill(
                    name=ResolvedSkillURI.parse(uri).skill_name,
                    description=f"Skill {uri}",
                    skill_path=PurePosixPath(uri),
                    instructions=result if isinstance(result, str) else None,
                )
            msg = f"Skill {uri!r} not found via ExtensionRegistry"
            raise SkillNotFoundError(msg)

        resolved = ResolvedSkillURI.parse(uri)
        skill = await self._find_skill_with_alternatives(resolved.skill_name)
        if skill is not None:
            return skill
        msg = f"Skill {resolved.skill_name!r} not found in any provider"
        raise SkillNotFoundError(msg)

    def get_provider(self, name: str) -> ProviderLike | None:
        """Get a registered provider by name.

        Args:
            name: Provider name

        Returns:
            The provider instance or None if not found
        """
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())
