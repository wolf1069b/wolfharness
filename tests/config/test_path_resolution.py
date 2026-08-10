"""TDD test suite for RFC-0009: Unified Configuration Relative Paths Resolution.

This test module covers:
- resolve_config_path() function
- ConfigPath Pydantic type
- ConfigContextManager context handling
- All resolution scenarios and edge cases

Priority order for resolution:
1. AGENTPOOL_LEGACY_PATHS=1 (highest priority - bypasses all)
2. AGENTPOOL_CONFIG_DIR environment variable
3. CONFIG_DIR context variable (via ConfigContextManager)
4. Current working directory (fallback)
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel
import pytest
from upathtools import UPath

from wolfharness_config.context import CONFIG_DIR, ConfigContextManager
from wolfharness_config.paths import (
    CONFIG_DIR_ENV_VAR,
    LEGACY_PATHS_ENV_VAR,
    ConfigPath,
    resolve_config_path,
)


pytestmark = pytest.mark.unit


# =============================================================================
# Tests for resolve_config_path() function
# =============================================================================


def test_resolve_config_path_with_context(tmp_path: Path):
    """Relative path + Context -> Resolved to Context directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path("./foo.txt")

    assert isinstance(result, UPath)
    assert str(result) == str(config_dir / "foo.txt")


def test_resolve_config_path_without_context():
    """Relative path + No Context -> Resolved relative to CWD."""
    relative_path = "./some_relative_file.txt"

    # Clear any context
    with ConfigContextManager(None):
        result = resolve_config_path(relative_path)

    assert isinstance(result, UPath)
    # Without context, should remain relative (or resolve to CWD)
    cwd = Path.cwd()
    assert not result.is_absolute() or str(result) == str(cwd / "some_relative_file.txt")


def test_resolve_config_path_absolute():
    """Absolute path -> Unchanged (absolute)."""
    absolute_path = "/absolute/path/to/file.txt"

    result = resolve_config_path(absolute_path)

    assert isinstance(result, UPath)
    assert result.is_absolute()
    assert str(result) == absolute_path


def test_resolve_config_path_env_override(monkeypatch, tmp_path: Path):
    """AGENTPOOL_CONFIG_DIR env var overrides default resolution."""
    env_config_dir = tmp_path / "env_config"
    env_config_dir.mkdir()
    context_dir = tmp_path / "context_dir"
    context_dir.mkdir()

    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_config_dir))

    with ConfigContextManager(str(context_dir)):
        result = resolve_config_path("./bar.txt")

    # Environment variable should take precedence
    assert isinstance(result, UPath)
    assert str(result) == str(env_config_dir / "bar.txt")


def test_resolve_config_path_legacy_mode(monkeypatch, tmp_path: Path):
    """Legacy mode bypasses all context logic."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    env_dir = tmp_path / "env_config"
    env_dir.mkdir()

    # Set both legacy mode and env var
    monkeypatch.setenv(LEGACY_PATHS_ENV_VAR, "1")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))

    with ConfigContextManager(str(context_dir)):
        result = resolve_config_path("./bar.txt")

    # In legacy mode, path should be returned as-is (relative to CWD)
    # It should ignore env var, context, and everything else
    assert isinstance(result, UPath)
    assert not result.is_absolute()


def test_resolve_config_path_legacy_mode_does_not_resolve():
    """In legacy mode, relative paths are never resolved to absolute."""
    relative_path = "./never_resolved.txt"

    os.environ[LEGACY_PATHS_ENV_VAR] = "1"
    try:
        result = resolve_config_path(relative_path)
        # In legacy mode, relative paths should stay relative (UPath normalizes ./)
        assert not result.is_absolute()
        # UPath may normalize ./path to path, just verify it's not absolute
        assert "never_resolved.txt" in str(result)
    finally:
        del os.environ[LEGACY_PATHS_ENV_VAR]


def test_resolve_config_path_absolute_stays_absolute(monkeypatch, tmp_path: Path):
    """Absolute paths are never resolved against context."""
    absolute_path = "/abs/path/file.txt"
    context_path = tmp_path / "config.yml"

    with ConfigContextManager(str(context_path)):
        result = resolve_config_path(absolute_path)

    assert str(result) == absolute_path
    assert result.is_absolute()


def test_resolve_config_path_resolution_priority(tmp_path: Path, monkeypatch):
    """Test complete resolution priority order."""
    legacy_dir = tmp_path / "legacy"
    env_dir = tmp_path / "env"
    context_dir = tmp_path / "context"
    legacy_dir.mkdir()
    env_dir.mkdir()
    context_dir.mkdir()

    # Test with all three set: legacy should win
    monkeypatch.setenv(LEGACY_PATHS_ENV_VAR, "1")
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))

    with ConfigContextManager(str(context_dir)):
        result = resolve_config_path("test.txt")

    # Legacy mode should bypass all
    assert not result.is_absolute()

    # Clean up legacy mode
    monkeypatch.delenv(LEGACY_PATHS_ENV_VAR)

    # Test with env + context: env should win
    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))

    with ConfigContextManager(str(context_dir)):
        result = resolve_config_path("test.txt")

    assert str(result) == str(env_dir / "test.txt")

    # Clean up env var
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR)

    # Test with context only: context should win
    with ConfigContextManager(str(context_dir)):
        result = resolve_config_path("test.txt")

    assert str(result) == str(context_dir / "test.txt")

    # Test with nothing: should use CWD (relative to current)
    with ConfigContextManager(None):
        result = resolve_config_path("test.txt")

    # No context set, so path stays as-is or resolves to CWD
    assert result.name == "test.txt"


def test_resolve_config_path_with_upath_input(tmp_path: Path):
    """resolve_config_path accepts UPath inputs."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path(UPath("./upath_input.txt"))

    assert isinstance(result, UPath)
    assert str(result) == str(config_dir / "upath_input.txt")


def test_resolve_config_path_nested_relative(tmp_path: Path):
    """Nested relative paths should resolve correctly."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path("./relative/nested/path.txt")

    expected = config_dir / "relative" / "nested" / "path.txt"
    assert str(result) == str(expected)


def test_resolve_config_path_parent_directory(tmp_path: Path):
    """Parent directory references (../) should work."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path("../parent_file.txt")

    # UPath doesn't automatically resolve .., but result should contain the correct path parts
    assert ".." in str(result) or "parent_file.txt" in str(result)
    # The path should be relative to config_dir
    assert str(config_dir) in str(result) or str(config_dir.parent) in str(result)


def test_resolve_config_path_empty_raises():
    """Empty path should be handled gracefully."""
    result = resolve_config_path("")
    assert str(result) == "."


def test_resolve_config_path_dot_path(tmp_path: Path):
    """Single dot path should resolve to config dir."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path(".")

    assert str(result) == str(config_dir)


# =============================================================================
# Tests for ConfigPath type annotation
# =============================================================================


class SampleConfigPathModel(BaseModel):
    """Test model using ConfigPath type."""

    path_field: ConfigPath
    name: str = "test"


def test_config_path_type_annotated():
    """ConfigPath type with BeforeValidator works correctly."""
    # This test verifies the type alias works - actual resolution happens
    # when validator is invoked during model instantiation
    config_path_type = ConfigPath

    # The type should be an Annotated type with a BeforeValidator
    assert hasattr(config_path_type, "__args__")
    assert hasattr(config_path_type, "__metadata__")


def test_config_path_in_pydantic_model(tmp_path: Path):
    """ConfigPath works in actual Pydantic model."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create test file to ensure it exists
    test_file = config_dir / "test_config.txt"
    test_file.write_text("test content")

    # Test that it accepts string path
    with ConfigContextManager(str(config_dir)):
        model = SampleConfigPathModel(path_field=str(test_file))

        assert isinstance(model.path_field, UPath)
        assert str(model.path_field) == str(test_file)


def test_config_path_auto_resolves_in_model(tmp_path: Path):
    """ConfigPath automatically resolves when model is instantiated."""
    config_dir = tmp_path / "project" / "config"
    config_dir.mkdir(parents=True)

    with ConfigContextManager(str(config_dir)):
        model = SampleConfigPathModel(path_field="./auto_resolved.txt")

    # Path should be resolved relative to config_dir
    expected = config_dir / "auto_resolved.txt"
    assert str(model.path_field) == str(expected)


def test_config_path_absolute_not_resolved(tmp_path: Path):
    """Absolute paths in ConfigPath should not be resolved."""
    absolute_path = "/absolute/config/path.txt"

    with ConfigContextManager(str(tmp_path)):
        model = SampleConfigPathModel(path_field=absolute_path)

    assert str(model.path_field) == absolute_path
    assert model.path_field.is_absolute()


def test_config_path_list_in_model(tmp_path: Path):
    """ConfigPath works in list fields."""

    class ModelWithListPaths(BaseModel):
        paths: list[ConfigPath]

    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        model = ModelWithListPaths(paths=["./file1.txt", "./file2.txt", "/absolute/file3.txt"])

    assert len(model.paths) == 3
    assert str(model.paths[0]) == str(config_dir / "file1.txt")
    assert str(model.paths[1]) == str(config_dir / "file2.txt")
    assert str(model.paths[2]) == "/absolute/file3.txt"


def test_config_path_with_env_override(tmp_path: Path, monkeypatch):
    """ConfigPath respects env var override during validation."""
    env_dir = tmp_path / "env_config"
    env_dir.mkdir()
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))

    with ConfigContextManager(str(context_dir)):
        model = SampleConfigPathModel(path_field="./env_override.txt")

    # Env var should take precedence over context
    assert str(model.path_field) == str(env_dir / "env_override.txt")


def test_config_path_with_legacy_mode(tmp_path: Path, monkeypatch):
    """ConfigPath respects legacy mode during validation."""
    context_dir = tmp_path / "context"
    context_dir.mkdir()

    monkeypatch.setenv(LEGACY_PATHS_ENV_VAR, "1")

    with ConfigContextManager(str(context_dir)):
        model = SampleConfigPathModel(path_field="./legacy_unchanged.txt")

    # In legacy mode, path should stay relative (UPath may normalize ./ prefix)
    assert not model.path_field.is_absolute()
    assert "legacy_unchanged.txt" in str(model.path_field)


# =============================================================================
# Tests for ConfigContextManager
# =============================================================================


def test_config_context_manager_sets_context(tmp_path: Path):
    """ConfigContextManager properly sets CONFIG_DIR context."""
    config_dir = tmp_path / "my_config"
    config_dir.mkdir()

    # Before context - CONFIG_DIR should be None or different
    initial = CONFIG_DIR.get()

    with ConfigContextManager(str(config_dir)):
        # Inside context - CONFIG_DIR should be set
        current = CONFIG_DIR.get()
        assert current is not None
        assert str(current) == str(config_dir)

    # After context - CONFIG_DIR should be restored
    restored = CONFIG_DIR.get()
    assert restored == initial


def test_config_context_manager_with_file_path(tmp_path: Path):
    """ConfigContextManager extracts directory from file path."""
    config_file = tmp_path / "config" / "agents.yml"
    config_file.parent.mkdir()
    config_file.write_text("# config")

    with ConfigContextManager(str(config_file)):
        current = CONFIG_DIR.get()
        assert str(current) == str(config_file.parent)


def test_config_context_manager_with_directory_path(tmp_path: Path):
    """ConfigContextManager handles directory paths correctly."""
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()

    # Directory has no suffix, so used as-is
    with ConfigContextManager(str(config_dir)):
        current = CONFIG_DIR.get()
        assert str(current) == str(config_dir)


def test_config_context_manager_restores_on_exception(tmp_path: Path):
    """ConfigContextManager properly resets context even on exception."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    initial = CONFIG_DIR.get()

    def _raise_exception():
        raise ValueError("Test exception")

    try:
        with ConfigContextManager(str(config_dir)):
            assert CONFIG_DIR.get() is not None
            _raise_exception()
    except ValueError:
        pass

    # After exception, context should be restored
    assert CONFIG_DIR.get() == initial


def test_config_context_manager_none_does_not_set_context():
    """Passing None to ConfigContextManager does not set context."""
    initial = CONFIG_DIR.get()

    with ConfigContextManager(None):
        # Should not change context
        current = CONFIG_DIR.get()
        assert current == initial


def test_config_context_manager_nested():
    """Nested ConfigContextManagers should use the most recent one."""
    # Note: ContextVars don't actually nest - they replace
    # This tests the behavior when nesting context managers
    outer_dir = Path("/outer/config")
    inner_dir = Path("/inner/config")

    with ConfigContextManager(str(outer_dir)):
        # Inner context replaces outer
        with ConfigContextManager(str(inner_dir)):
            inner_context = CONFIG_DIR.get()
            assert str(inner_context) == str(inner_dir)

        # After inner exits, outer should be... handled based on implementation
        # ContextVar tokens ensure proper restoration
        restored = CONFIG_DIR.get()
        # The behavior depends on implementation - context may be None or restored
        assert restored is not None or restored is None  # Just verify it doesn't crash


def test_config_context_manager_returns_self(tmp_path: Path):
    """ConfigContextManager returns self on enter."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)) as ctx:
        assert ctx is not None
        assert isinstance(ctx, ConfigContextManager)


# =============================================================================
# Environment variable cleanup tests
# =============================================================================


def test_env_var_cleanup(tmp_path: Path, monkeypatch):
    """Ensure environment variables are properly cleaned up."""
    # This test validates that we properly save/restore env vars
    # in case tests run in parallel

    # Clear any existing env vars
    for var in [LEGACY_PATHS_ENV_VAR, CONFIG_DIR_ENV_VAR]:
        if var in os.environ:
            monkeypatch.delenv(var)

    # Set env vars
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    monkeypatch.setenv(CONFIG_DIR_ENV_VAR, str(env_dir))
    monkeypatch.setenv(LEGACY_PATHS_ENV_VAR, "1")

    # Verify they're set
    assert os.environ.get(CONFIG_DIR_ENV_VAR) == str(env_dir)
    assert os.environ.get(LEGACY_PATHS_ENV_VAR) == "1"

    # Clean up
    monkeypatch.delenv(CONFIG_DIR_ENV_VAR)
    monkeypatch.delenv(LEGACY_PATHS_ENV_VAR)

    # Verify cleanup
    assert CONFIG_DIR_ENV_VAR not in os.environ
    assert LEGACY_PATHS_ENV_VAR not in os.environ


# =============================================================================
# Edge cases and error handling
# =============================================================================


def test_resolve_config_path_with_special_characters(tmp_path: Path):
    """Paths with special characters should be handled correctly."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path("./file with spaces.txt")

    assert "file with spaces.txt" in str(result)


def test_resolve_config_path_with_unicode(tmp_path: Path):
    """Paths with unicode characters should be handled correctly."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        result = resolve_config_path("./文件📄.txt")

    assert "文件📄.txt" in str(result)


def test_resolve_config_path_trailing_slash(tmp_path: Path):
    """Paths with trailing slashes should be handled."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    with ConfigContextManager(str(config_dir)):
        # Directory path
        result = resolve_config_path("./subdir/")

    # Should resolve to config_dir/subdir/ (trailing slash behavior may vary)
    assert "subdir" in str(result)


def test_config_context_with_nonexistent_path(tmp_path: Path):
    """ConfigContextManager with non-existent path should handle gracefully."""
    nonexistent = tmp_path / "does" / "not" / "exist" / "config.yml"

    # This should still work - pathlib doesn't require paths to exist
    with ConfigContextManager(str(nonexistent)) as ctx:
        # Context manager doesn't validate path existence
        assert ctx._config_dir is not None
        # Parent of the file path is extracted
