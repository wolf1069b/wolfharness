"""Tests for deferred tool config and checkpoint config wiring.

Ensures DeferredToolConfig fields are specifiable in tool YAML config,
CheckpointConfig is specifiable in SessionPoolConfig, and the wiring
from config to Tool() instance works correctly.
"""

from __future__ import annotations

from datetime import timedelta

import pytest


pytestmark = pytest.mark.unit


def tool_conf(**kwargs: object) -> str:
    """Get Tool config instance.

    The tool() lambda is defined in this module so its __module__ and
    __qualname__ resolve correctly for the import_path validation.
    """
    return kwargs.get("import_path", "")


class TestDeferredToolConfigOnBaseConfig:
    """Tests that BaseToolConfig has deferred fields with correct defaults."""

    def test_base_tool_config_has_deferred_fields(self):
        """BaseToolConfig should have deferred fields with correct defaults.

        Checks deferred, deferred_kind, deferred_strategy, deferred_placeholder,
        deferred_timeout fields.
        """
        from wolfharness_config.tools import BaseToolConfig

        # Check fields exist and have correct defaults
        fields = BaseToolConfig.model_fields

        assert "deferred" in fields
        assert fields["deferred"].default is False

        assert "deferred_kind" in fields
        assert fields["deferred_kind"].default == "external"

        assert "deferred_strategy" in fields
        assert fields["deferred_strategy"].default == "block"

        assert "deferred_placeholder" in fields
        assert (
            fields["deferred_placeholder"].default == "This tool is processing in the background."
        )

        assert "deferred_timeout" in fields
        assert fields["deferred_timeout"].default is None

    def test_deferred_fields_have_json_schema_annotations(self):
        """Deferred fields should have Field titles for YAML auto-completion."""
        from wolfharness_config.tools import BaseToolConfig

        schema = BaseToolConfig.model_json_schema()
        props = schema["properties"]

        assert props["deferred"]["title"] == "Deferred execution"
        assert props["deferred_kind"]["title"] == "Deferred kind"
        assert props["deferred_strategy"]["title"] == "Deferred strategy"
        assert props["deferred_placeholder"]["title"] == "Deferred placeholder"
        assert props["deferred_timeout"]["title"] == "Deferred timeout"


class TestDeferredToolWiring:
    """Tests that YAML deferred fields map to Tool.deferred=True at load time."""

    def test_import_tool_default_not_deferred(self):
        """ImportToolConfig with default settings produces non-deferred tool."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(import_path=f"{__name__}:tool_conf")
        tool = config.get_tool()

        assert tool.deferred is False

    def test_import_tool_deferred_true(self):
        """ImportToolConfig with deferred=True produces deferred tool."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
        )
        tool = config.get_tool()

        assert tool.deferred is True

    def test_import_tool_deferred_kind_unapproved(self):
        """deferred_kind='unapproved' forces strategy='block'."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_kind="unapproved",
        )
        tool = config.get_tool()

        assert tool.deferred_kind == "unapproved"
        # unapproved + block is valid combination
        assert tool.deferred_strategy == "block"

    def test_import_tool_deferred_strategy_continue(self):
        """deferred_strategy='continue' with deferred_kind='external'."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_kind="external",
            deferred_strategy="continue",
        )
        tool = config.get_tool()

        assert tool.deferred_kind == "external"
        assert tool.deferred_strategy == "continue"

    def test_import_tool_deferred_placeholder_custom(self):
        """Custom deferred_placeholder is passed through."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_placeholder="Please wait, processing...",
        )
        tool = config.get_tool()

        assert tool.deferred_placeholder == "Please wait, processing..."

    def test_import_tool_deferred_timeout_string(self):
        """deferred_timeout as a string '30s' is parsed to timedelta(seconds=30)."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_timeout="30s",
        )
        tool = config.get_tool()

        assert tool.deferred_timeout == timedelta(seconds=30)

    def test_import_tool_deferred_timeout_timedelta(self):
        """deferred_timeout as timedelta is passed through directly."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_timeout=timedelta(minutes=5),
        )
        tool = config.get_tool()

        assert tool.deferred_timeout == timedelta(minutes=5)

    def test_import_tool_deferred_timeout_none(self):
        """deferred_timeout=None means no timeout."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_timeout=None,
        )
        tool = config.get_tool()

        assert tool.deferred_timeout is None

    def test_import_tool_deferred_fields_are_all_kwargs_passthrough(self):
        """All deferred fields should pass through to Tool() via kwargs."""
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_kind="external",
            deferred_strategy="block",
            deferred_placeholder="Processing...",
            deferred_timeout="1m",
        )
        tool = config.get_tool()

        # Verify ALL deferred-related fields are wired through
        assert tool.deferred is True
        assert tool.deferred_kind == "external"
        assert tool.deferred_strategy == "block"
        assert tool.deferred_placeholder == "Processing..."
        assert tool.deferred_timeout == timedelta(seconds=60)

    def test_invalid_combination_raises_tool_error(self):
        """deferred_kind='unapproved' with strategy='continue' should raise ToolError."""
        from wolfharness.tools.exceptions import ToolError
        from wolfharness_config.tools import ImportToolConfig

        config = ImportToolConfig(
            import_path=f"{__name__}:tool_conf",
            deferred=True,
            deferred_kind="unapproved",
            deferred_strategy="continue",
        )

        with pytest.raises(ToolError, match="must block"):
            config.get_tool()


class TestCheckpointConfigWiring:
    """Tests that CheckpointConfig is specifiable in SessionPoolConfig."""

    def test_session_pool_config_has_checkpoint_field(self):
        """SessionPoolConfig should have a checkpoint field accepting CheckpointConfig."""
        from wolfharness_config.session_pool import SessionPoolConfig

        fields = SessionPoolConfig.model_fields
        assert "checkpoint" in fields
        assert fields["checkpoint"].default is None

    def test_session_pool_config_default_checkpoint_is_none(self):
        """Default SessionPoolConfig should have checkpoint=None."""
        from wolfharness_config.session_pool import SessionPoolConfig

        config = SessionPoolConfig()
        assert config.checkpoint is None

    def test_session_pool_config_with_checkpoint_enabled(self):
        """SessionPoolConfig with checkpoint enabled."""
        from wolfharness_config.durable import CheckpointConfig
        from wolfharness_config.session_pool import SessionPoolConfig

        config = SessionPoolConfig(
            checkpoint=CheckpointConfig(enabled=True),
        )
        assert config.checkpoint is not None
        assert config.checkpoint.enabled is True

    def test_session_pool_config_with_checkpoint_disabled(self):
        """SessionPoolConfig with checkpoint disabled."""
        from wolfharness_config.durable import CheckpointConfig
        from wolfharness_config.session_pool import SessionPoolConfig

        config = SessionPoolConfig(
            checkpoint=CheckpointConfig(enabled=False),
        )
        assert config.checkpoint is not None
        assert config.checkpoint.enabled is False

    def test_checkpoint_field_json_schema(self):
        """Checkpoint field should have JSON schema annotations."""
        from wolfharness_config.session_pool import SessionPoolConfig

        schema = SessionPoolConfig.model_json_schema()
        assert "checkpoint" in schema["properties"]


class TestDeferredToolConfigExists:
    """Verify DeferredToolConfig is importable."""

    def test_deferred_tool_config_importable(self):
        """DeferredToolConfig can be imported from wolfharness_config.durable."""
        from wolfharness_config.durable import DeferredToolConfig

        assert DeferredToolConfig is not None

    def test_deferred_tool_config_defaults(self):
        """DeferredToolConfig has correct defaults."""
        from wolfharness_config.durable import DeferredToolConfig

        config = DeferredToolConfig()
        assert config.enabled is True
        assert config.default_strategy == "block"
        assert config.default_timeout is None


class TestYamlSchema:
    """Tests that JSON schema properly exposes deferred fields for YAML auto-completion."""

    def test_import_tool_config_schema_has_deferred_fields(self):
        """ImportToolConfig JSON schema should include deferred fields."""
        from wolfharness_config.tools import ImportToolConfig

        schema = ImportToolConfig.model_json_schema()
        props = schema["properties"]

        assert "deferred" in props
        assert "deferred_kind" in props
        assert "deferred_strategy" in props
        assert "deferred_placeholder" in props

    def test_session_pool_config_schema_has_checkpoint_field(self):
        """SessionPoolConfig JSON schema should include checkpoint field."""
        from wolfharness_config.session_pool import SessionPoolConfig

        schema = SessionPoolConfig.model_json_schema()
        assert "checkpoint" in schema["properties"]
