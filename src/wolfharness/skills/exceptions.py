"""Skill-related exceptions."""

from __future__ import annotations

from wolfharness.utils.baseregistry import AgentPoolError


class SkillError(AgentPoolError):
    """Base exception for all skill-related errors."""


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found.

    Includes optional list of available skills to help users find
    the correct skill name.
    """

    def __init__(self, skill_name: str, available_skills: list[str] | None = None) -> None:
        """Initialize the exception with skill name and optional available skills.

        Args:
            skill_name: The name of the skill that was not found.
            available_skills: Optional list of available skill names to suggest.
        """
        msg = f"Skill not found: {skill_name}"
        if available_skills:
            msg += f". Available skills: {', '.join(available_skills)}"
        super().__init__(msg)


class ReferenceNotFoundError(SkillError):
    """Raised when a skill reference file cannot be found."""

    def __init__(self, reference_path: str) -> None:
        """Initialize the exception with the reference path.

        Args:
            reference_path: The path to the reference file that was not found.
        """
        msg = f"Reference file not found: {reference_path}"
        super().__init__(msg)


class SecurityError(SkillError):
    """Raised on path traversal or other security violations."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a security violation message.

        Args:
            message: Description of the security violation.
        """
        super().__init__(f"Security violation: {message}")


class ProviderError(SkillError):
    """Raised when a provider operation fails."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a provider error message.

        Args:
            message: Description of the provider error.
        """
        super().__init__(f"Provider error: {message}")
