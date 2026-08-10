"""Configuration for skill slash commands."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SkillSlashConfig(BaseModel):
    """Per-skill configuration for slash command exposure.

    This config controls how a skill is exposed as a slash command,
    including whether it requires confirmation and which agents can use it.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    """Whether this skill is exposed as a slash command."""

    require_confirmation: bool = Field(default=False)
    """Whether to require user confirmation before executing."""

    allowed_agents: list[str] = Field(default_factory=list)
    """List of agent names that can use this skill (empty = all)."""

    aliases: list[str] = Field(default_factory=list)
    """Alternative names for this command."""


class SkillCommandConfig(BaseModel):
    """Global configuration for skill slash commands."""

    model_config = ConfigDict(extra="forbid")

    default_config: SkillSlashConfig = Field(default_factory=SkillSlashConfig)
    """Default config for all skills."""

    per_skill_config: dict[str, SkillSlashConfig] = Field(default_factory=dict)
    """Per-skill overrides keyed by skill name."""

    prefix: str = Field(default="/skill:")
    """Command prefix used for skill commands."""

    def get_skill_config(self, skill_name: str) -> SkillSlashConfig:
        """Get config for a specific skill.

        Returns per-skill config if exists, otherwise default.

        Args:
            skill_name: The name of the skill to get config for.

        Returns:
            The SkillSlashConfig for the skill.
        """
        return self.per_skill_config.get(skill_name, self.default_config)
