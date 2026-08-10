"""Skill command dataclass for protocol-agnostic command representation.

This is a lightweight dataclass used by protocol server skill bridges
to wrap skills as slash commands. Command discovery is handled by
``ExtensionRegistry.get_command_resources()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from wolfharness.skills.skill import Skill


@dataclass(frozen=True)
class SkillCommand:
    """A skill exposed as a slash command.

    Attributes:
        name: Command name (typically the skill name without prefix).
        description: Human-readable description of what the command does.
        skill: The underlying Skill instance containing full skill metadata.
        input_hint: Hint text shown to users about command arguments.
        category: Command category for grouping (default "skill").
        skill_uri: Optional skill:// URI for the skill.
    """

    name: str
    description: str
    skill: Skill
    input_hint: str = "Arguments for skill"
    category: str = "skill"
    skill_uri: str | None = None

    @property
    def resolved_skill_uri(self) -> str:
        """Get the skill URI, generating from name if not explicitly set."""
        return self.skill_uri or f"skill://{self.name}"

    def is_valid_input(self, input_text: str) -> tuple[bool, str | None]:
        """Validate input text for this command.

        Args:
            input_text: The input to validate.

        Returns:
            A tuple containing:
                - Boolean indicating if input is valid
                - Error message string if invalid, None if valid
        """
        if not input_text.strip():
            return False, "Input cannot be empty"
        return True, None
