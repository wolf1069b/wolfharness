"""Tests for SkillsConfig model."""

from __future__ import annotations

import pytest
from upathtools import UPath

from wolfharness_config.skills import DEFAULT_SKILLS_PATHS, SkillsConfig


pytestmark = pytest.mark.unit


def test_skills_config_default_values():
    """Test SkillsConfig with default values."""
    config = SkillsConfig()

    assert config.paths == []
    assert config.include_default is True


def test_skills_config_with_custom_paths():
    """Test SkillsConfig with custom paths."""
    config = SkillsConfig(paths=[UPath("./my-skills"), UPath("/absolute/path")])

    assert len(config.paths) == 2
    assert config.paths[0] == UPath("./my-skills")
    assert config.paths[1] == UPath("/absolute/path")
    assert config.include_default is True


def test_skills_config_include_default_false():
    """Test SkillsConfig with include_default set to False."""
    config = SkillsConfig(include_default=False)

    assert config.paths == []
    assert config.include_default is False


def test_get_effective_paths_custom_only():
    """Test get_effective_paths with custom paths only (no defaults)."""
    config = SkillsConfig(
        paths=[UPath("./skills"), UPath("/absolute/skills")],
        include_default=False,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths()

    assert len(result) == 2
    # ConfigPath resolves paths during validation; without config context,
    # relative paths stay relative (get_effective_paths no longer resolves them)
    assert str(result[0]).endswith("skills")
    assert result[1] == UPath("/absolute/skills")


def test_get_effective_paths_with_defaults():
    """Test get_effective_paths includes default paths when enabled."""
    config = SkillsConfig(
        paths=[UPath("./custom-skills")],
        include_default=True,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths()

    assert len(result) == 3
    # Custom path (may or may not be absolute depending on ConfigPath context)
    assert str(result[0]).endswith("custom-skills")
    # Default paths appended after custom paths
    assert result[1] == DEFAULT_SKILLS_PATHS[0]  # ~/.claude/skills/
    assert result[2] == DEFAULT_SKILLS_PATHS[1]  # .claude/skills/


def test_get_effective_paths_with_config_file_path():
    """Test that config_file_path parameter is ignored (deprecated).

    ConfigPath now handles path resolution automatically during validation.
    The config_file_path parameter is kept for backward compatibility but
    has no effect.
    """
    config_file = UPath("/home/user/project/config.yml")

    config = SkillsConfig(
        paths=[UPath("../shared-skills"), UPath("./local-skills")],
        include_default=False,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths(config_file_path=config_file)

    assert len(result) == 2
    # ConfigPath resolves paths during validation; without a config context,
    # relative paths are NOT resolved against config_file_path anymore
    assert str(result[0]).endswith("shared-skills")
    assert str(result[1]).endswith("local-skills")


def test_get_effective_paths_absolute_paths_unaffected():
    """Test that absolute paths are not modified by config_file_path."""
    config_file = UPath("/some/other/path/config.yml")

    config = SkillsConfig(
        paths=[UPath("/custom/absolute/skills")],
        include_default=False,
    )

    result = config.get_effective_paths(config_file_path=config_file)

    assert len(result) == 1
    assert result[0] == UPath("/custom/absolute/skills")


def test_get_effective_paths_no_config_file_uses_cwd():
    """Test that without config context, relative paths stay relative.

    ConfigPath resolves paths during validation; without a config dir
    context, relative paths are not resolved to absolute.
    """
    config = SkillsConfig(
        paths=[UPath("./test-skills")],
        include_default=False,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths(config_file_path=None)

    assert len(result) == 1
    assert str(result[0]).endswith("test-skills")


def test_get_effective_paths_remote_paths():
    """Test that remote paths are preserved as-is."""
    config = SkillsConfig(
        paths=[UPath("s3://bucket/skills"), UPath("github://org/repo/skills")],
        include_default=False,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths()

    assert len(result) == 2
    assert result[0] == UPath("s3://bucket/skills")
    assert result[1] == UPath("github://org/repo/skills")


def test_get_effective_paths_first_path_wins():
    """Test 'first path wins' priority - custom paths before defaults."""
    config = SkillsConfig(
        paths=[UPath("./my-skills")],
        include_default=True,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths()

    # Custom paths come first
    assert str(result[0]).endswith("my-skills")
    # Default paths come after
    assert result[1] == DEFAULT_SKILLS_PATHS[0]
    assert result[2] == DEFAULT_SKILLS_PATHS[1]


def test_pydantic_validation():
    """Test that SkillsConfig validates properly with Pydantic."""
    # Valid config
    config = SkillsConfig(paths=[UPath("/path")], include_default=True)
    assert config.paths == [UPath("/path")]
    assert config.include_default is True

    # Invalid types should raise ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SkillsConfig(paths=["not", "a", "list"], include_default="not a bool")


def test_empty_config_no_defaults():
    """Test empty config with defaults disabled returns empty list."""
    config = SkillsConfig(paths=[], include_default=False)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = config.get_effective_paths()

    assert result == []


def test_config_yaml_roundtrip():
    """Test that SkillsConfig can be created from string paths (YAML-style)."""
    # Simulate what YAML parsing produces: string paths, not UPath objects
    config = SkillsConfig(
        paths=["./skills", "/absolute/skills"],
        include_default=True,
    )

    assert len(config.paths) == 2
    assert str(config.paths[0]).endswith("skills")
    assert config.paths[1] == UPath("/absolute/skills")
    assert config.include_default is True

    # ConfigPath resolves relative paths during validation;
    # without a config dir context, relative paths stay relative
    assert config.paths[1].is_absolute()
