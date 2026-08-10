"""Tests for skill-related exception classes.

This module provides comprehensive tests for:
- SkillError inheritance and base behavior
- SkillNotFoundError with/without available_skills
- ReferenceNotFoundError
- SecurityError
- ProviderError
"""

from __future__ import annotations

import pytest

from wolfharness.skills.exceptions import (
    ProviderError,
    ReferenceNotFoundError,
    SecurityError,
    SkillError,
    SkillNotFoundError,
)
from wolfharness.utils.baseregistry import AgentPoolError


pytestmark = pytest.mark.unit


# =============================================================================
# SkillError Tests
# =============================================================================


def test_skill_error_inherits_from_agent_pool_error() -> None:
    """Test that SkillError inherits from AgentPoolError."""
    assert issubclass(SkillError, AgentPoolError)


def test_skill_error_can_be_caught_as_agent_pool_error() -> None:
    """Test that SkillError can be caught as AgentPoolError."""
    with pytest.raises(AgentPoolError):
        raise SkillError("test error")


def test_skill_error_message() -> None:
    """Test that SkillError preserves message."""
    msg = "Custom skill error message"
    exc = SkillError(msg)

    assert str(exc) == msg


# =============================================================================
# SkillNotFoundError Tests
# =============================================================================


def test_skill_not_found_error_inherits_from_skill_error() -> None:
    """Test that SkillNotFoundError inherits from SkillError."""
    assert issubclass(SkillNotFoundError, SkillError)


def test_skill_not_found_error_without_available_skills() -> None:
    """Test SkillNotFoundError without available skills list."""
    exc = SkillNotFoundError("my-skill")

    assert "my-skill" in str(exc)
    assert "not found" in str(exc).lower()
    assert "available skills" not in str(exc).lower()


def test_skill_not_found_error_with_available_skills() -> None:
    """Test SkillNotFoundError with available skills list."""
    available = ["python-expert", "go-developer", "rust-guru"]
    exc = SkillNotFoundError("javascript-dev", available_skills=available)

    assert "javascript-dev" in str(exc)
    assert "not found" in str(exc).lower()
    assert "available skills" in str(exc).lower()
    for skill in available:
        assert skill in str(exc)


def test_skill_not_found_error_with_empty_available_skills() -> None:
    """Test SkillNotFoundError with empty available skills list."""
    exc = SkillNotFoundError("my-skill", available_skills=[])

    assert "my-skill" in str(exc)
    # Empty list should not add "Available skills" section
    assert "available skills" not in str(exc).lower()


def test_skill_not_found_error_with_single_available_skill() -> None:
    """Test SkillNotFoundError with single available skill."""
    exc = SkillNotFoundError("unknown-skill", available_skills=["only-skill"])

    assert "unknown-skill" in str(exc)
    assert "available skills: only-skill" in str(exc).lower()


# =============================================================================
# ReferenceNotFoundError Tests
# =============================================================================


def test_reference_not_found_error_inherits_from_skill_error() -> None:
    """Test that ReferenceNotFoundError inherits from SkillError."""
    assert issubclass(ReferenceNotFoundError, SkillError)


def test_reference_not_found_error_message() -> None:
    """Test ReferenceNotFoundError message includes path."""
    path = "skills/python-expert/references/advanced.md"
    exc = ReferenceNotFoundError(path)

    assert path in str(exc)
    assert "reference file not found" in str(exc).lower()


def test_reference_not_found_error_with_simple_path() -> None:
    """Test ReferenceNotFoundError with simple filename."""
    path = "guide.md"
    exc = ReferenceNotFoundError(path)

    assert path in str(exc)


def test_reference_not_found_error_with_absolute_path() -> None:
    """Test ReferenceNotFoundError with absolute path."""
    path = "/absolute/path/to/reference.md"
    exc = ReferenceNotFoundError(path)

    assert path in str(exc)


# =============================================================================
# SecurityError Tests
# =============================================================================


def test_security_error_inherits_from_skill_error() -> None:
    """Test that SecurityError inherits from SkillError."""
    assert issubclass(SecurityError, SkillError)


def test_security_error_message() -> None:
    """Test SecurityError message includes violation description."""
    message = "Path traversal detected"
    exc = SecurityError(message)

    assert message in str(exc)
    assert "security violation" in str(exc).lower()


def test_security_error_with_path_traversal() -> None:
    """Test SecurityError with path traversal message."""
    message = "Path traversal detected in URI: 'skill://local/skill/../../../etc/passwd'"
    exc = SecurityError(message)

    assert message in str(exc)
    assert "security violation" in str(exc).lower()


def test_security_error_with_null_bytes() -> None:
    """Test SecurityError with null bytes message."""
    message = "URI contains null bytes"
    exc = SecurityError(message)

    assert message in str(exc)


# =============================================================================
# ProviderError Tests
# =============================================================================


def test_provider_error_inherits_from_skill_error() -> None:
    """Test that ProviderError inherits from SkillError."""
    assert issubclass(ProviderError, SkillError)


def test_provider_error_message() -> None:
    """Test ProviderError message includes error description."""
    message = "Failed to connect to provider"
    exc = ProviderError(message)

    assert message in str(exc)
    assert "provider error" in str(exc).lower()


def test_provider_error_with_details() -> None:
    """Test ProviderError with detailed error message."""
    message = "Provider 'mcp-server' returned status 500: Internal Server Error"
    exc = ProviderError(message)

    assert message in str(exc)
    assert "provider error" in str(exc).lower()


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


def test_all_skill_errors_can_be_caught_as_skill_error() -> None:
    """Test that all skill exceptions can be caught as SkillError."""
    exceptions = [
        SkillNotFoundError("test"),
        ReferenceNotFoundError("test"),
        SecurityError("test"),
        ProviderError("test"),
    ]

    for exc in exceptions:
        with pytest.raises(SkillError):
            raise exc


def test_all_skill_errors_can_be_caught_as_agent_pool_error() -> None:
    """Test that all skill exceptions can be caught as AgentPoolError."""
    exceptions = [
        SkillNotFoundError("test"),
        ReferenceNotFoundError("test"),
        SecurityError("test"),
        ProviderError("test"),
    ]

    for exc in exceptions:
        with pytest.raises(AgentPoolError):
            raise exc


def test_exception_str_representation() -> None:
    """Test that all exceptions have proper string representation."""
    test_cases = [
        (SkillError("base error"), "base error"),
        (SkillNotFoundError("missing"), "missing"),
        (ReferenceNotFoundError("ref.md"), "ref.md"),
        (SecurityError("violation"), "violation"),
        (ProviderError("failure"), "failure"),
    ]

    for exc, expected_substring in test_cases:
        assert expected_substring in str(exc)
