"""Tests for URI resolver and skill URI parsing.

This module provides comprehensive tests for:
- ResolvedSkillURI.parse() with various URI formats
- Path traversal detection and security checks
- Skill name validation
- URL decoding
- SkillURIResolver provider registration and resolution
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from wolfharness.capabilities.resource_protocols import SkillEntry, SkillResource
from wolfharness.skills.exceptions import SecurityError, SkillNotFoundError
from wolfharness.skills.uri_resolver import (
    ResolvedSkillURI,
    SkillURIResolver,
    _validate_skill_name,
)


pytestmark = pytest.mark.unit


# =============================================================================
# ResolvedSkillURI.parse() - Basic URI Parsing
# =============================================================================


def test_parse_basic_flat_uri() -> None:
    """Test parsing basic skill://skill-name URI."""
    uri = "skill://python-expert"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "python-expert"
    assert result.reference_path is None


def test_parse_uri_with_reference_path() -> None:
    """Test parsing URI with reference path."""
    uri = "skill://python-expert/references/guide.md"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "python-expert"
    assert result.reference_path == "references/guide.md"


def test_parse_uri_with_deep_reference_path() -> None:
    """Test parsing URI with deeply nested reference path."""
    uri = "skill://my-skill/a/b/c/d/file.md"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-skill"
    assert result.reference_path == "a/b/c/d/file.md"


def test_parse_bare_skill_name() -> None:
    """Test parsing bare skill name without scheme."""
    uri = "my-skill"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-skill"
    assert result.reference_path is None


def test_parse_bare_skill_name_with_hyphens() -> None:
    """Test parsing bare skill name with multiple hyphens."""
    uri = "my-test-skill-name"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-test-skill-name"
    assert result.reference_path is None


# =============================================================================
# ResolvedSkillURI.parse() - URL Decoding
# =============================================================================


def test_parse_uri_with_encoded_characters() -> None:
    """Test parsing URI with URL-encoded characters (hyphen decoded)."""
    uri = "skill://my%2Dskill"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-skill"


def test_parse_uri_with_encoded_reference_path() -> None:
    """Test parsing URI with URL-encoded reference path."""
    uri = "skill://my-skill/references/%66ile.md"  # %66 = 'f'
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-skill"
    assert result.reference_path == "references/file.md"


# =============================================================================
# ResolvedSkillURI.parse() - Trailing Slash / Empty Forms
# =============================================================================


def test_parse_flat_uri_with_trailing_slash() -> None:
    """Test that flat URI with trailing slash parses correctly."""
    uri = "skill://local/"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "local"
    assert result.reference_path is None


def test_parse_uri_with_only_scheme() -> None:
    """Test that URI with only scheme raises ValueError."""
    uri = "skill://"

    with pytest.raises(ValueError, match="URI is empty"):
        ResolvedSkillURI.parse(uri)


# =============================================================================
# ResolvedSkillURI.parse() - Path Traversal Detection
# =============================================================================


def test_parse_uri_with_path_traversal_in_reference() -> None:
    """Test that path traversal in URI reference raises SecurityError."""
    uri = "skill://my-skill/../etc/passwd"

    with pytest.raises(SecurityError, match="Path traversal"):
        ResolvedSkillURI.parse(uri)


def test_parse_uri_with_path_traversal_in_deep_reference() -> None:
    """Test that path traversal in deep reference path raises SecurityError."""
    uri = "skill://my-skill/../../../etc/passwd"

    with pytest.raises(SecurityError, match="Path traversal"):
        ResolvedSkillURI.parse(uri)


def test_parse_uri_with_encoded_path_traversal() -> None:
    """Test that encoded path traversal raises SecurityError."""
    uri = "skill://my-skill/%2e%2e/%2e%2e/secret"

    with pytest.raises(SecurityError, match="Path traversal"):
        ResolvedSkillURI.parse(uri)


def test_parse_uri_with_single_dot_is_allowed() -> None:
    """Test that single dot in path is allowed."""
    uri = "skill://my-skill/./file.md"
    result = ResolvedSkillURI.parse(uri)

    assert result.skill_name == "my-skill"
    assert result.reference_path == "./file.md"


# =============================================================================
# ResolvedSkillURI.parse() - Null Byte Detection
# =============================================================================


def test_parse_uri_with_null_byte() -> None:
    """Test that null byte in URI raises SecurityError."""
    uri = "skill://my-skill\x00"

    with pytest.raises(SecurityError, match="null bytes"):
        ResolvedSkillURI.parse(uri)


def test_parse_uri_with_encoded_null_byte() -> None:
    """Test that encoded null byte raises SecurityError."""
    uri = "skill://my-skill%00"

    with pytest.raises(SecurityError, match="null bytes"):
        ResolvedSkillURI.parse(uri)


# =============================================================================
# ResolvedSkillURI.parse() - Invalid URI Format
# =============================================================================


def test_parse_uri_with_invalid_scheme() -> None:
    """Test that invalid scheme raises ValueError."""
    uri = "http://my-skill"

    with pytest.raises(ValueError, match="Invalid URI scheme"):
        ResolvedSkillURI.parse(uri)


# =============================================================================
# Skill Name Validation
# =============================================================================


def test_validate_skill_name_with_valid_names() -> None:
    """Test valid skill names are accepted."""
    valid_names = [
        "my-skill",
        "skill123",
        "a",
        "abc-def-ghi",
        "python-expert",
    ]

    for name in valid_names:
        result = _validate_skill_name(name)
        assert result == name, f"{name!r} should be valid"


def test_validate_skill_name_converts_to_lowercase() -> None:
    """Test that skill name is normalized to lowercase."""
    # Note: The validator expects already-lowercase input
    # This tests that mixed case is rejected
    with pytest.raises(SecurityError, match="must be lowercase"):
        _validate_skill_name("My-Skill")


def test_validate_skill_name_rejects_uppercase() -> None:
    """Test that uppercase letters are rejected."""
    with pytest.raises(SecurityError, match="must be lowercase"):
        _validate_skill_name("Python-Expert")


def test_validate_skill_name_rejects_starting_hyphen() -> None:
    """Test that skill name starting with hyphen is rejected."""
    with pytest.raises(SecurityError, match="cannot start or end with a hyphen"):
        _validate_skill_name("-my-skill")


def test_validate_skill_name_rejects_ending_hyphen() -> None:
    """Test that skill name ending with hyphen is rejected."""
    with pytest.raises(SecurityError, match="cannot start or end with a hyphen"):
        _validate_skill_name("my-skill-")


def test_validate_skill_name_rejects_consecutive_hyphens() -> None:
    """Test that consecutive hyphens are rejected."""
    with pytest.raises(SecurityError, match="consecutive hyphens"):
        _validate_skill_name("my--skill")


def test_validate_skill_name_rejects_invalid_characters() -> None:
    """Test that invalid characters are rejected (after underscore normalization)."""
    invalid_names = [
        "my.skill",  # Dot
        "my/skill",  # Slash
        "my skill",  # Space
    ]

    for name in invalid_names:
        with pytest.raises(SecurityError, match="invalid characters"):
            _validate_skill_name(name)


def test_validate_skill_name_normalizes_underscores() -> None:
    """Test that underscores are normalized to hyphens per Agent Skills Spec."""
    result = _validate_skill_name("my_skill")
    assert result == "my-skill"

    result = _validate_skill_name("systematic_troubleshooting")
    assert result == "systematic-troubleshooting"

    result = _validate_skill_name("multi_word_skill_name")
    assert result == "multi-word-skill-name"


def test_validate_skill_name_rejects_empty() -> None:
    """Test that empty skill name is rejected."""
    with pytest.raises(SecurityError, match="non-empty"):
        _validate_skill_name("")


def test_validate_skill_name_rejects_whitespace_only() -> None:
    """Test that whitespace-only skill name is rejected."""
    with pytest.raises(SecurityError, match="non-empty"):
        _validate_skill_name("   ")


def test_validate_skill_name_strips_whitespace() -> None:
    """Test that skill name is stripped of whitespace."""
    result = _validate_skill_name("  my-skill  ")
    assert result == "my-skill"


# =============================================================================
# SkillURIResolver - Provider Registration
# =============================================================================


def _make_skill_resource(
    entries: list[SkillEntry] | None = None,
    content: str = "skill content",
) -> MagicMock:
    """Create a mock SkillResource provider for testing."""
    provider = MagicMock(spec=SkillResource)
    provider.list_skills = AsyncMock(return_value=entries or [])
    provider.read_skill = AsyncMock(return_value=content)
    provider.skill_exists = AsyncMock(return_value=len(entries or []) > 0)
    return provider


def test_resolver_register_provider() -> None:
    """Test registering a provider."""
    resolver = SkillURIResolver()
    provider = _make_skill_resource()

    resolver.register_provider("local", provider)

    assert resolver.get_provider("local") is provider


def test_resolver_register_multiple_providers() -> None:
    """Test registering multiple providers."""
    resolver = SkillURIResolver()
    provider1 = _make_skill_resource()
    provider2 = _make_skill_resource()

    resolver.register_provider("local", provider1)
    resolver.register_provider("remote", provider2)

    assert resolver.get_provider("local") is provider1
    assert resolver.get_provider("remote") is provider2


def test_resolver_unregister_provider() -> None:
    """Test unregistering a provider."""
    resolver = SkillURIResolver()
    provider = _make_skill_resource()

    resolver.register_provider("local", provider)
    assert resolver.get_provider("local") is provider

    resolver.unregister_provider("local")
    assert resolver.get_provider("local") is None


def test_resolver_unregister_nonexistent_provider() -> None:
    """Test unregistering a nonexistent provider does not raise."""
    resolver = SkillURIResolver()

    # Should not raise
    resolver.unregister_provider("nonexistent")


def test_resolver_list_providers() -> None:
    """Test listing registered providers."""
    resolver = SkillURIResolver()
    provider = _make_skill_resource()

    resolver.register_provider("local", provider)
    resolver.register_provider("remote", provider)

    providers = resolver.list_providers()
    assert "local" in providers
    assert "remote" in providers
    assert len(providers) == 2


# =============================================================================
# SkillURIResolver - Skill Resolution
# =============================================================================


@pytest.mark.asyncio
async def test_resolver_resolve_by_bare_name() -> None:
    """Test resolving skill by bare name across all providers."""
    resolver = SkillURIResolver()
    entry = SkillEntry(name="my-skill", description="desc", uri="skill://my-skill")
    provider = _make_skill_resource(entries=[entry], content="instructions")

    resolver.register_provider("local", provider)
    result = await resolver.resolve("my-skill")

    assert result.name == "my-skill"
    assert result.description == "desc"


@pytest.mark.asyncio
async def test_resolver_resolve_by_flat_uri() -> None:
    """Test resolving skill by flat skill:// URI."""
    resolver = SkillURIResolver()
    entry = SkillEntry(name="my-skill", description="desc", uri="skill://my-skill")
    provider = _make_skill_resource(entries=[entry], content="instructions")

    resolver.register_provider("local", provider)
    result = await resolver.resolve("skill://my-skill")

    assert result.name == "my-skill"


@pytest.mark.asyncio
async def test_resolver_resolve_not_found_any_provider() -> None:
    """Test that SkillNotFoundError is raised when skill not in any provider."""
    resolver = SkillURIResolver()
    provider = _make_skill_resource(entries=[])

    resolver.register_provider("local", provider)

    with pytest.raises(SkillNotFoundError, match="not found"):
        await resolver.resolve("missing-skill")


@pytest.mark.asyncio
async def test_resolver_resolve_searches_multiple_providers() -> None:
    """Test that resolver searches all providers for bare skill name."""
    resolver = SkillURIResolver()

    entry2 = SkillEntry(name="skill-2", description="desc", uri="skill://skill-2")
    provider1 = _make_skill_resource(entries=[])
    provider2 = _make_skill_resource(entries=[entry2], content="instructions")

    resolver.register_provider("provider1", provider1)
    resolver.register_provider("provider2", provider2)

    result = await resolver.resolve("skill-2")

    assert result.name == "skill-2"


@pytest.mark.asyncio
async def test_resolver_resolve_first_match_wins() -> None:
    """Test that first matching skill is returned when duplicates exist."""
    resolver = SkillURIResolver()

    entry1 = SkillEntry(name="my-skill", description="desc1", uri="skill://skill1")
    entry2 = SkillEntry(name="my-skill", description="desc2", uri="skill://skill2")
    provider1 = _make_skill_resource(entries=[entry1], content="instructions1")
    provider2 = _make_skill_resource(entries=[entry2], content="instructions2")

    resolver.register_provider("provider1", provider1)
    resolver.register_provider("provider2", provider2)

    result = await resolver.resolve("my-skill")

    # First provider's skill should be returned
    assert result.name == "my-skill"
    assert result.description == "desc1"


# =============================================================================
# SkillURIResolver.unregister_provider() Tests
# =============================================================================


def test_unregister_provider_removes_from_list() -> None:
    """Test that unregister_provider() removes provider from the registry."""
    resolver = SkillURIResolver()

    provider = _make_skill_resource()
    resolver.register_provider("test_provider", provider)

    assert "test_provider" in resolver.list_providers()

    resolver.unregister_provider("test_provider")

    assert "test_provider" not in resolver.list_providers()


def test_unregister_provider_nonexistent_is_noop() -> None:
    """Test that unregister_provider() with nonexistent name does not raise."""
    resolver = SkillURIResolver()

    # Should not raise
    resolver.unregister_provider("nonexistent")

    assert len(resolver.list_providers()) == 0


@pytest.mark.asyncio
async def test_unregister_provider_prevents_resolution() -> None:
    """Test that skills from unregistered provider are no longer resolved."""
    resolver = SkillURIResolver()

    entry = SkillEntry(name="my-skill", description="desc", uri="skill://my-skill")
    provider = _make_skill_resource(entries=[entry], content="instructions")

    resolver.register_provider("test_provider", provider)

    # Should resolve before unregister
    result = await resolver.resolve("my-skill")
    assert result.name == "my-skill"

    resolver.unregister_provider("test_provider")

    # Should fail after unregister
    with pytest.raises(SkillNotFoundError):
        await resolver.resolve("my-skill")
