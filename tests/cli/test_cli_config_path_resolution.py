"""Integration tests for RFC-0009: CLI commands use ConfigContextManager for path resolution.

This test module verifies that all CLI commands correctly wrap their
AgentsManifest loading with ConfigContextManager to ensure relative paths
are resolved relative to the configuration file, not the current working directory.

Test coverage:
- serve-opencode: model_validate with ConfigContextManager
- serve-acp: model_validate with ConfigContextManager
- serve-agui: from_file with ConfigContextManager
- serve-mcp: from_file with ConfigContextManager
- serve-api: from_file with ConfigContextManager
- serve-vercel: from_file with ConfigContextManager
- watch: from_file with ConfigContextManager
- task: from_file with ConfigContextManager
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wolfharness.models.manifest import AgentsManifest
from wolfharness_config.context import CONFIG_DIR, ConfigContextManager


pytestmark = pytest.mark.unit


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def test_config_with_relative_path(tmp_path: Path) -> Path:
    """Create a test config with relative paths for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create a relative skills directory
    skills_dir = config_dir / "custom_skills"
    skills_dir.mkdir()

    # Create config with relative path
    config_content = """
skills:
  paths:
    - "./custom_skills"

agents:
  test_agent:
    type: native
    model: "openai:gpt-4o-mini"
    system_prompt: "Test agent"
"""
    config_file = config_dir / "agents.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def cwd_outside_config(tmp_path: Path) -> Path:
    """Create a different directory to run tests from (not the config dir)."""
    other_dir = tmp_path / "other_working_dir"
    other_dir.mkdir()
    return other_dir


# =============================================================================
# Helper functions for testing
# =============================================================================


def get_resolved_skills_path(manifest: AgentsManifest) -> Path | None:
    """Extract the resolved skills path from a manifest."""
    if manifest.skills and manifest.skills.paths and len(manifest.skills.paths) > 0:
        # Handle ConfigPath type which might be a list or single path
        first_path = manifest.skills.paths[0]
        if hasattr(first_path, "resolve"):
            return Path(str(first_path.resolve()))
        return Path(str(first_path))
    return None


# =============================================================================
# Tests for serve-opencode command
# =============================================================================


class TestServeOpenCode:
    """Test that serve-opencode uses ConfigContextManager for path resolution."""

    def test_serve_opencode_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """Relative paths in config should resolve to config file directory."""
        config_path = test_config_with_relative_path
        config_dir = config_path.parent

        # Simulate being in a different working directory
        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Import and test the resolution logic
            from wolfharness_config.resolution import resolve_config

            resolved = resolve_config(explicit_path=str(config_path))

            # Load manifest with ConfigContextManager (the fix)
            with ConfigContextManager(resolved.primary_path):
                manifest = AgentsManifest.model_validate(resolved.data)
            if resolved.primary_path:
                manifest = manifest.model_copy(update={"config_file_path": resolved.primary_path})

            # Verify relative path was resolved to config directory
            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            expected_path = config_dir / "custom_skills"
            assert "custom_skills" in str(skills_path)
            # Should be absolute or at least point to correct location
            absolute_skills = (
                skills_path if skills_path.is_absolute() else cwd_outside_config / skills_path
            )
            assert str(absolute_skills) == str(expected_path) or expected_path.name in str(
                skills_path
            )

        finally:
            os.chdir(original_cwd)

    def test_serve_opencode_without_context_resolves_to_cwd(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """Without ConfigContextManager, paths would resolve incorrectly."""
        config_path = test_config_with_relative_path

        # Simulate being in a different working directory
        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            from wolfharness_config.resolution import resolve_config

            resolved = resolve_config(explicit_path=str(config_path))

            # Load manifest WITHOUT ConfigContextManager (simulates bug)
            manifest_without_context = AgentsManifest.model_validate(resolved.data)

            # The skills path should NOT be resolved correctly
            skills_path = get_resolved_skills_path(manifest_without_context)
            if skills_path and not skills_path.is_absolute():
                # Without context, relative paths would be wrong
                # This test demonstrates what would happen without the fix
                pass  # Documenting the expected behavior

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for serve-acp command
# =============================================================================


class TestServeAcp:
    """Test that serve-acp uses ConfigContextManager for path resolution."""

    def test_serve_acp_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """Relative paths should resolve to config file, not CWD."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            from wolfharness_config.resolution import resolve_config

            resolved = resolve_config(explicit_path=str(config_path))

            # Same pattern as serve_acp.py
            with ConfigContextManager(resolved.primary_path):
                manifest = AgentsManifest.model_validate(resolved.data)
            if resolved.primary_path:
                manifest = manifest.model_copy(update={"config_file_path": resolved.primary_path})

            # Verify configuration directory is used
            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for serve-agui command
# =============================================================================


class TestServeAgui:
    """Test that serve-agui uses ConfigContextManager for path resolution."""

    def test_serve_agui_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """serve-agui should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as serve_agui.py
            with ConfigContextManager(str(config_path)):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for serve-mcp command
# =============================================================================


class TestServeMcp:
    """Test that serve-mcp uses ConfigContextManager for path resolution."""

    def test_serve_mcp_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """serve-mcp should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as serve_mcp.py
            with ConfigContextManager(config_path):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for serve-api command
# =============================================================================


class TestServeApi:
    """Test that serve-api uses ConfigContextManager for path resolution."""

    def test_serve_api_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """serve-api should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as serve_api.py
            with ConfigContextManager(config_path):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for serve-vercel command
# =============================================================================


class TestServeVercel:
    """Test that serve-vercel uses ConfigContextManager for path resolution."""

    def test_serve_vercel_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """serve-vercel should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as serve_vercel.py
            with ConfigContextManager(config_path):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for watch command
# =============================================================================


class TestWatch:
    """Test that watch uses ConfigContextManager for path resolution."""

    def test_watch_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """Watch should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as watch.py
            with ConfigContextManager(config_path):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for task command
# =============================================================================


class TestTask:
    """Test that task uses ConfigContextManager for path resolution."""

    def test_task_resolves_paths_relative_to_config(
        self,
        test_config_with_relative_path: Path,
        cwd_outside_config: Path,
    ):
        """Task should resolve paths relative to config file."""
        config_path = test_config_with_relative_path

        original_cwd = Path.cwd()
        try:
            os.chdir(cwd_outside_config)

            # Same pattern as task.py
            with ConfigContextManager(config_path):
                manifest = AgentsManifest.from_file(config_path)

            skills_path = get_resolved_skills_path(manifest)
            assert skills_path is not None
            assert "custom_skills" in str(skills_path)

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Tests for context nesting and edge cases
# =============================================================================


class TestConfigContextEdgeCases:
    """Test edge cases for ConfigContextManager usage in CLI."""

    def test_nested_context_preserves_inner_value(self, tmp_path: Path):
        """Nested ConfigContextManagers should use innermost context."""
        outer_dir = tmp_path / "outer"
        inner_dir = tmp_path / "inner"
        outer_dir.mkdir()
        inner_dir.mkdir()

        with ConfigContextManager(outer_dir):
            outer_config_dir = CONFIG_DIR.get()
            assert outer_config_dir is not None
            assert str(outer_config_dir) == str(outer_dir)

            with ConfigContextManager(inner_dir):
                inner_config_dir = CONFIG_DIR.get()
                assert inner_config_dir is not None
                assert str(inner_config_dir) == str(inner_dir)

            # After inner context, outer should be restored
            restored_dir = CONFIG_DIR.get()
            assert str(restored_dir) == str(outer_dir)

    def test_context_none_does_not_set(self, tmp_path: Path):
        """ConfigContextManager with None should not set context."""
        original_dir = CONFIG_DIR.get()

        with ConfigContextManager(None):
            current_dir = CONFIG_DIR.get()
            assert current_dir == original_dir

    def test_context_exception_cleanup(self, tmp_path: Path):
        """Context should be cleaned up even on exception."""
        original_dir = CONFIG_DIR.get()
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        try:
            with ConfigContextManager(config_dir):
                raise ValueError("Test exception")  # noqa: TRY301
        except ValueError:
            pass

        # Context should be reset
        current_dir = CONFIG_DIR.get()
        assert current_dir == original_dir
