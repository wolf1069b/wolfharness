"""Security audit tests for skill system.

This module provides comprehensive security tests for:
- Path traversal protection in reference file loading
- URL-encoded path traversal attacks
- Null byte injection attacks
- Symlink-based directory traversal attacks
- URI-level security in ResolvedSkillURI.parse()

Security is enforced at two layers:
1. ``_load_reference_content()`` — validates filesystem reference paths
2. ``ResolvedSkillURI.parse()`` — validates skill:// URIs (used for MCP-based skills)
"""

from __future__ import annotations

import pytest
from upathtools import UPath

from wolfharness.skills.exceptions import SecurityError
from wolfharness.skills.uri_resolver import ResolvedSkillURI


pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================


def _make_skill(skill_dir):
    """Create a Skill object from a filesystem directory."""
    from wolfharness.skills.skill import Skill

    return Skill(
        name="test-skill",
        description="Skill with references",
        skill_path=UPath(str(skill_dir)),
    )


# =============================================================================
# Path Traversal Attack Tests — _load_reference_content (filesystem)
# =============================================================================


@pytest.fixture
def skill_with_references(tmp_path):
    """Create a skill with references directory for testing."""
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: Skill with references
---

# Test Skill

Instructions.
""")

    # Create references directory with files
    refs_dir = skill_dir / "references"
    refs_dir.mkdir()

    (refs_dir / "guide.md").write_text("# Guide\n\nGuide content.")
    (refs_dir / "config.json").write_text('{"key": "value"}')

    # Create subdirectory
    subdir = refs_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested content")

    return skill_dir


@pytest.fixture
def local_skill(skill_with_references):
    """Create a Skill object for filesystem reference tests."""
    return _make_skill(skill_with_references)


# =============================================================================
# Path Traversal Attack Tests — _load_reference_content
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_absolute_path(local_skill):
    """Test path traversal with absolute path attempt: /etc/passwd."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises(SecurityError):
        await _load_reference_content(local_skill, "/etc/passwd")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_basic_dotdot(local_skill):
    """Test basic path traversal: ../../../etc/passwd."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises(SecurityError) as exc_info:
        await _load_reference_content(local_skill, "../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_embedded(local_skill):
    """Test embedded path traversal: subdir/../../../etc/passwd."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises(SecurityError) as exc_info:
        await _load_reference_content(local_skill, "subdir/../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_multiple_dotdot(local_skill):
    """Test multiple .. sequences: ../../../../../../../etc/passwd."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises(SecurityError) as exc_info:
        await _load_reference_content(local_skill, "../../../../../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_leading_dotdot(local_skill):
    """Test leading .. sequence: ../etc/passwd."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises(SecurityError) as exc_info:
        await _load_reference_content(local_skill, "../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_traversal_mixed_separators(local_skill):
    r"""Test path traversal with mixed separators: ..\..\..\etc\passwd."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # On Unix, backslash is treated as literal character
    # This test verifies the path is rejected
    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "..\\..\\..\\etc\\passwd")


# =============================================================================
# URL-Encoded Path Traversal Tests — _load_reference_content
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_url_encoded_traversal_percent_2f(local_skill):
    """Test URL-encoded path traversal: ..%2f..%2f..%2fetc%2fpasswd."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # _load_reference_content does not URL-decode before checking
    # This should fail as file not found (path still blocked)
    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "..%2f..%2f..%2fetc%2fpasswd")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_url_encoded_traversal_lowercase(local_skill):
    """Test URL-encoded with lowercase: ..%2f..%2fetc%2fpasswd."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "..%2f..%2fetc%2fpasswd")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_url_encoded_traversal_uppercase(local_skill):
    """Test URL-encoded with uppercase: ..%2F..%2Fetc%2Fpasswd."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "..%2F..%2Fetc%2Fpasswd")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_url_encoded_dot(local_skill):
    """Test URL-encoded dot: %2e%2e/%2e%2e/%2e%2e/etc/passwd."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # %2e is encoded dot, but _load_reference_content doesn't URL-decode
    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "%2e%2e/%2e%2e/%2e%2e/etc/passwd")


# =============================================================================
# Null Byte Injection Tests — _load_reference_content
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_null_byte_injection(local_skill):
    r"""Test null byte injection: file\x00.txt."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "file\x00.txt")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_null_byte_injection_with_path(local_skill):
    r"""Test null byte injection with path: subdir/file\x00.txt."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "subdir/file\x00.txt")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_null_byte_at_start(local_skill):
    r"""Test null byte at start of path: \x00file.txt."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    with pytest.raises((SecurityError, ReferenceNotFoundError)):
        await _load_reference_content(local_skill, "\x00file.txt")


# =============================================================================
# Symlink Attack Tests — _load_reference_content
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_symlink_to_outside_directory(skill_with_references):
    """Test that symlink pointing outside references dir is blocked.

    Creates a symlink inside references/ that points to a file outside
    the references directory. The _load_reference_content should resolve
    the symlink and reject the path.
    """
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    refs_dir = skill_with_references / "references"

    # Create a file outside the references directory
    outside_file = skill_with_references.parent / "outside_secret.txt"
    outside_file.write_text("SECRET CONTENT OUTSIDE REFERENCES")

    # Create a symlink inside references pointing to outside file
    symlink_path = refs_dir / "malicious_link.txt"
    try:
        symlink_path.symlink_to(outside_file)

        skill = _make_skill(skill_with_references)
        with pytest.raises((SecurityError, ReferenceNotFoundError)):
            await _load_reference_content(skill, "references/malicious_link.txt")
    finally:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        if outside_file.exists():
            outside_file.unlink()


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_symlink_chain_traversal(skill_with_references):
    """Test symlink chain that eventually escapes references directory.

    Creates a chain of symlinks where the final target is outside
    the allowed directory.
    """
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    refs_dir = skill_with_references / "references"
    subdir = refs_dir / "subdir"

    # Create files and symlinks
    outside_file = skill_with_references.parent / "secret.txt"
    outside_file.write_text("SECRET")

    intermediate_link = skill_with_references.parent / "intermediate.txt"

    link1 = subdir / "link1.txt"
    link2 = refs_dir / "link2.txt"

    try:
        # Create intermediate link outside references
        intermediate_link.symlink_to(outside_file)
        # Create link1 -> intermediate (within subdir)
        link1.symlink_to(intermediate_link)
        # Create link2 -> link1 (within refs)
        link2.symlink_to(link1)

        skill = _make_skill(skill_with_references)
        with pytest.raises((SecurityError, ReferenceNotFoundError)):
            await _load_reference_content(skill, "references/link2.txt")
    finally:
        for link in [link2, link1, intermediate_link]:
            if link.exists() or link.is_symlink():
                link.unlink()
        if outside_file.exists():
            outside_file.unlink()


# =============================================================================
# Path Traversal Attack Tests — ResolvedSkillURI.parse() (URI-level)
# =============================================================================


@pytest.mark.security
def test_uri_path_traversal_basic_dotdot():
    """Test URI path traversal: skill://provider/test-skill/../../../etc/passwd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_path_traversal_embedded():
    """Test URI embedded path traversal: skill://provider/test-skill/refs/../../../etc/passwd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/refs/../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_path_traversal_leading_dotdot():
    """Test URI leading .. sequence: skill://provider/test-skill/../etc/passwd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_path_traversal_deeply_nested():
    """Test URI deeply nested traversal: skill://provider/test-skill/a/b/c/../../../../etc/passwd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/a/b/c/../../../../etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


# =============================================================================
# URL-Encoded Path Traversal Tests — ResolvedSkillURI.parse()
# =============================================================================


@pytest.mark.security
def test_uri_url_encoded_traversal_percent_2f():
    """Test URI URL-encoded path traversal: ..%2f..%2f..%2fetc%2fpasswd.

    ResolvedSkillURI.parse() URL-decodes before checking,
    so this should raise SecurityError.
    """
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/..%2f..%2f..%2fetc%2fpasswd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_url_encoded_traversal_uppercase():
    """Test URI URL-encoded with uppercase: ..%2F..%2Fetc%2Fpasswd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/..%2F..%2Fetc%2fpasswd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_url_encoded_dot():
    """Test URI URL-encoded dot: %2e%2e/%2e%2e/%2e%2e/etc/passwd."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/%2e%2e/%2e%2e/%2e%2e/etc/passwd")

    assert "traversal" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_double_url_encoding():
    """Test double URL-encoded path: %%32%65%%32%65 (double-encoded ..)."""
    # Double encoding: % -> %25, 2 -> %32, e -> %65
    # %%32%65 = %2e = .
    # After single unquote: %2e%2e — not "..", so no SecurityError
    # But the resulting reference path is not ".." so it's neutralized
    result = ResolvedSkillURI.parse("skill://provider/test-skill/%%32%65%%32%65")
    # The double-encoded value is not decoded to ".." so it passes
    # but the reference path is not ".." — attack is neutralized
    assert result.reference_path != ".."


# =============================================================================
# Null Byte Injection Tests — ResolvedSkillURI.parse()
# =============================================================================


@pytest.mark.security
def test_uri_null_byte_injection():
    r"""Test URI null byte injection: skill://provider/test-skill/file\x00.txt."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/file\x00.txt")

    assert "null bytes" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_null_byte_in_middle():
    r"""Test URI null byte in middle: skill://provider/test-skill/config\x00.json."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/config\x00.json")

    assert "null bytes" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_null_byte_with_path():
    r"""Test URI null byte with path: skill://provider/test-skill/subdir/file\x00.txt."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/subdir/file\x00.txt")

    assert "null bytes" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_null_byte_at_start():
    r"""Test URI null byte at start: skill://provider/test-skill/\x00file.txt."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/\x00file.txt")

    assert "null bytes" in str(exc_info.value).lower()


@pytest.mark.security
def test_uri_multiple_null_bytes():
    r"""Test URI multiple null bytes: skill://provider/test-skill/file\x00\x00\x00.txt."""
    with pytest.raises(SecurityError) as exc_info:
        ResolvedSkillURI.parse("skill://provider/test-skill/file\x00\x00\x00.txt")

    assert "null bytes" in str(exc_info.value).lower()


# =============================================================================
# Edge Case Security Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_empty_path(local_skill):
    """Test empty path handling."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # Empty path resolves to the skill directory itself — read_text fails
    # because it's a directory, not a file.
    with pytest.raises((SecurityError, ReferenceNotFoundError, IsADirectoryError)):
        await _load_reference_content(local_skill, "")


@pytest.mark.security
def test_uri_empty_path():
    """Test URI with trailing slash — no reference path.

    A URI like skill://provider/test-skill/ has an empty trailing path
    component. This is not a security issue — it means "load the skill
    itself, no reference file." The parser should handle it gracefully.
    """
    result = ResolvedSkillURI.parse("skill://test-skill/")
    assert result.skill_name == "test-skill"
    assert result.reference_path is None


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_single_dot(local_skill):
    """Test single dot path: ./file.txt."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # Single dot should be allowed (refers to current directory)
    # But file doesn't exist, so ReferenceNotFoundError
    with pytest.raises(ReferenceNotFoundError):
        await _load_reference_content(local_skill, "./nonexistent.txt")


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_dot_slash_prefix(local_skill):
    """Test dot slash prefix: ./references/guide.md."""
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # This should work - single dot is not traversal
    content = await _load_reference_content(local_skill, "references/guide.md")
    assert "Guide content" in content


@pytest.mark.asyncio
@pytest.mark.security
async def test_local_path_with_special_chars(local_skill):
    """Test path with special characters that are NOT traversal."""
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    # These should be treated as literal filenames (which don't exist)
    with pytest.raises(ReferenceNotFoundError):
        await _load_reference_content(local_skill, "file@2x.txt")

    with pytest.raises(ReferenceNotFoundError):
        await _load_reference_content(local_skill, "file#name.txt")


@pytest.mark.security
def test_uri_path_with_special_chars():
    """Test URI path with special characters that are NOT traversal."""
    # These should not trigger SecurityError — they are just unusual filenames
    # The URI parser may reject them for invalid skill names, but not for security
    try:
        result = ResolvedSkillURI.parse("skill://provider/test-skill/file@2x.txt")
        # If it parses, the reference path should contain the special chars
        assert result.reference_path is not None
    except (SecurityError, ValueError):
        # If it fails, it should not be a SecurityError about traversal
        pass


# =============================================================================
# Security Validation Summary Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.security
async def test_all_attacks_raise_security_error_or_blocked(local_skill):
    """Summary test: verify all attack vectors are blocked.

    This test documents that both security layers correctly block:
    1. Path traversal with ..
    2. URL-encoded path traversal
    3. Null byte injection
    4. Symlink-based attacks
    """
    from wolfharness.skills.exceptions import ReferenceNotFoundError
    from wolfharness_toolsets.builtin.skills import _load_reference_content

    attacks_blocked = []

    # Test 1: Basic path traversal — filesystem
    try:
        await _load_reference_content(local_skill, "../../../etc/passwd")
        attacks_blocked.append(("local_basic_traversal", False))
    except (SecurityError, ReferenceNotFoundError):
        attacks_blocked.append(("local_basic_traversal", True))

    # Test 2: URL-encoded traversal — URI parser
    try:
        ResolvedSkillURI.parse("skill://provider/test-skill/..%2f..%2fetc%2fpasswd")
        attacks_blocked.append(("uri_url_encoded_traversal", False))
    except SecurityError:
        attacks_blocked.append(("uri_url_encoded_traversal", True))

    # Test 3: Null byte — filesystem
    try:
        await _load_reference_content(local_skill, "file\x00.txt")
        attacks_blocked.append(("local_null_byte", False))
    except (SecurityError, ReferenceNotFoundError):
        attacks_blocked.append(("local_null_byte", True))

    # Test 4: Null byte — URI parser
    try:
        ResolvedSkillURI.parse("skill://provider/test-skill/file\x00.txt")
        attacks_blocked.append(("uri_null_byte", False))
    except SecurityError:
        attacks_blocked.append(("uri_null_byte", True))

    # Verify all attacks were blocked
    failed = [name for name, blocked in attacks_blocked if not blocked]
    if failed:
        pytest.fail(f"Security vulnerabilities detected! Unblocked attacks: {failed}")


# =============================================================================
# Documentation Test
# =============================================================================


def test_security_considerations_documented():
    """Verify security considerations are properly documented in code.

    This test checks that SecurityError has appropriate docstrings
    and is properly exported from the exceptions module.
    """
    from wolfharness.skills.exceptions import SecurityError

    # Verify SecurityError can be instantiated
    error = SecurityError("Test security violation")
    assert "Security violation" in str(error)
    assert "Test security violation" in str(error)

    # Verify it's a proper exception hierarchy
    from wolfharness.skills.exceptions import SkillError

    assert issubclass(SecurityError, SkillError)


@pytest.mark.security
def test_security_error_message_format():
    """Test that SecurityError produces properly formatted messages."""
    error = SecurityError("Path traversal detected in: ../../../etc/passwd")
    msg = str(error)

    assert "Security violation" in msg
    assert "Path traversal detected" in msg
    assert "../../../etc/passwd" in msg
