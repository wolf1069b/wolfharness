"""Test skill command configuration models."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from wolfharness_config.skill_commands import SkillCommandConfig, SkillSlashConfig


pytestmark = pytest.mark.unit


class TestSkillSlashConfig:
    """Test SkillSlashConfig model."""

    def test_default_values(self):
        """Test that SkillSlashConfig has correct default values."""
        config = SkillSlashConfig()

        assert config.enabled is True
        assert config.require_confirmation is False
        assert config.allowed_agents == []
        assert config.aliases == []

    def test_custom_values(self):
        """Test that SkillSlashConfig accepts custom values."""
        config = SkillSlashConfig(
            enabled=False,
            require_confirmation=True,
            allowed_agents=["agent1", "agent2"],
            aliases=["alias1", "alias2"],
        )

        assert config.enabled is False
        assert config.require_confirmation is True
        assert config.allowed_agents == ["agent1", "agent2"]
        assert config.aliases == ["alias1", "alias2"]

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden (ConfigDict extra="forbid")."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SkillSlashConfig(
                enabled=True,
                unknown_field="value",
            )

    def test_partial_customization(self):
        """Test that partial customization preserves defaults."""
        config = SkillSlashConfig(
            require_confirmation=True,
        )

        assert config.enabled is True  # default
        assert config.require_confirmation is True  # custom
        assert config.allowed_agents == []  # default
        assert config.aliases == []  # default


class TestSkillCommandConfig:
    """Test SkillCommandConfig model."""

    def test_default_values(self):
        """Test that SkillCommandConfig has correct default values."""
        config = SkillCommandConfig()

        assert config.default_config == SkillSlashConfig()
        assert config.per_skill_config == {}
        assert config.prefix == "/skill:"

    def test_custom_prefix(self):
        """Test custom prefix configuration."""
        config = SkillCommandConfig(prefix="/cmd:")

        assert config.prefix == "/cmd:"
        assert config.default_config.enabled is True
        assert config.per_skill_config == {}

    def test_custom_default_config(self):
        """Test custom default config for all skills."""
        custom_default = SkillSlashConfig(
            enabled=True,
            require_confirmation=True,
        )
        config = SkillCommandConfig(default_config=custom_default)

        assert config.default_config.require_confirmation is True
        assert config.per_skill_config == {}

    def test_per_skill_config(self):
        """Test per-skill override configuration."""
        skill_override = SkillSlashConfig(
            enabled=False,
            allowed_agents=["admin"],
            aliases=["quick-test"],
        )
        config = SkillCommandConfig(per_skill_config={"test-skill": skill_override})

        assert "test-skill" in config.per_skill_config
        assert config.per_skill_config["test-skill"].enabled is False
        assert config.per_skill_config["test-skill"].allowed_agents == ["admin"]
        assert config.per_skill_config["test-skill"].aliases == ["quick-test"]

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden (ConfigDict extra="forbid")."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SkillCommandConfig(
                prefix="/cmd:",
                unknown_field="value",
            )

    def test_get_skill_config_uses_default(self):
        """Test get_skill_config returns default when no per-skill config."""
        config = SkillCommandConfig()

        skill_config = config.get_skill_config("unknown-skill")

        assert skill_config == config.default_config
        assert skill_config.enabled is True

    def test_get_skill_config_uses_override(self):
        """Test get_skill_config returns per-skill config when set."""
        skill_override = SkillSlashConfig(
            enabled=False,
            require_confirmation=True,
            allowed_agents=["agent1"],
        )
        config = SkillCommandConfig(per_skill_config={"my-skill": skill_override})

        skill_config = config.get_skill_config("my-skill")

        assert skill_config.enabled is False
        assert skill_config.require_confirmation is True
        assert skill_config.allowed_agents == ["agent1"]

    def test_get_skill_config_isolation(self):
        """Test that per-skill configs are isolated from each other."""
        config = SkillCommandConfig(
            per_skill_config={
                "skill1": SkillSlashConfig(enabled=False),
                "skill2": SkillSlashConfig(require_confirmation=True),
            }
        )

        skill1_config = config.get_skill_config("skill1")
        skill2_config = config.get_skill_config("skill2")
        default_config = config.get_skill_config("skill3")

        assert skill1_config.enabled is False
        assert skill1_config.require_confirmation is False

        assert skill2_config.enabled is True
        assert skill2_config.require_confirmation is True

        assert default_config.enabled is True
        assert default_config.require_confirmation is False


class TestYamlConfigLoading:
    """Test YAML config loading patterns."""

    def test_skill_config_from_dict(self):
        """Test creating SkillSlashConfig from dict (simulating YAML loading)."""
        data = {
            "enabled": True,
            "require_confirmation": True,
            "allowed_agents": ["admin", "coder"],
            "aliases": ["test", "t"],
        }
        config = SkillSlashConfig.model_validate(data)

        assert config.enabled is True
        assert config.require_confirmation is True
        assert config.allowed_agents == ["admin", "coder"]
        assert config.aliases == ["test", "t"]

    def test_skill_command_config_from_dict(self):
        """Test creating SkillCommandConfig from dict (simulating YAML loading)."""
        data = {
            "prefix": "/sk:",
            "default_config": {
                "enabled": False,
                "require_confirmation": True,
            },
            "per_skill_config": {
                "special-skill": {
                    "enabled": True,
                    "allowed_agents": ["admin"],
                }
            },
        }
        config = SkillCommandConfig.model_validate(data)

        assert config.prefix == "/sk:"
        assert config.default_config.enabled is False
        assert config.default_config.require_confirmation is True
        assert config.per_skill_config["special-skill"].enabled is True
        assert config.per_skill_config["special-skill"].allowed_agents == ["admin"]

    def test_empty_dict_defaults(self):
        """Test that empty dict produces default values."""
        config = SkillSlashConfig.model_validate({})

        assert config.enabled is True
        assert config.require_confirmation is False
        assert config.allowed_agents == []
        assert config.aliases == []

        command_config = SkillCommandConfig.model_validate({})

        assert command_config.default_config.enabled is True
        assert command_config.per_skill_config == {}
        assert command_config.prefix == "/skill:"
