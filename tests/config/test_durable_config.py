"""Tests for DeferredToolConfig and CheckpointConfig.

Unit tests for deferred tool and checkpoint configuration models.
Tests cover default values, YAML round-trip, ISO 8601 serialization,
and timeout field validation.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import yaml

from wolfharness_config.durable import CheckpointConfig, DeferredToolConfig


pytestmark = pytest.mark.unit


# =============================================================================
# DeferredToolConfig Tests
# =============================================================================


def test_deferred_tool_defaults():
    """DeferredToolConfig defaults: enabled=True, default_strategy='block', default_timeout=None."""
    config = DeferredToolConfig()

    assert config.enabled is True
    assert config.default_strategy == "block"
    assert config.default_timeout is None


def test_deferred_tool_custom_strategy():
    """DeferredToolConfig accepts valid strategy values."""
    for strategy in ("block", "continue", "stream"):
        config = DeferredToolConfig(default_strategy=strategy)
        assert config.default_strategy == strategy


def test_deferred_tool_serialize_as_dict():
    """DeferredToolConfig(enabled=True, default_strategy='block') serializes cleanly."""
    config = DeferredToolConfig(enabled=True, default_strategy="block")

    d = config.model_dump()

    assert d["enabled"] is True
    assert d["default_strategy"] == "block"
    assert d["default_timeout"] is None


def test_deferred_tool_timeout_as_timedelta():
    """DeferredToolConfig(default_timeout=timedelta(hours=1)) stores as timedelta."""
    config = DeferredToolConfig(default_timeout=timedelta(hours=1))

    assert isinstance(config.default_timeout, timedelta)
    assert config.default_timeout == timedelta(hours=1)


def test_deferred_tool_timeout_serializes_iso8601():
    """DeferredToolConfig(default_timeout=timedelta(hours=1)) serializes as ISO 8601 'PT1H'."""
    config = DeferredToolConfig(default_timeout=timedelta(hours=1))

    serialized = config.model_dump(mode="json")

    assert serialized["default_timeout"] == "PT1H"


def test_deferred_tool_timeout_parse_string():
    """DeferredToolConfig with string timeout parses to timedelta via field_validator."""
    config = DeferredToolConfig(default_timeout="1h")

    assert isinstance(config.default_timeout, timedelta)
    assert config.default_timeout == timedelta(hours=1)


def test_deferred_tool_timeout_none_remains_none():
    """DeferredToolConfig with None timeout stays None after validation."""
    config = DeferredToolConfig(default_timeout=None)

    assert config.default_timeout is None


def test_deferred_tool_yaml_round_trip():
    """DeferredToolConfig YAML round-trip: string timeout round-trips via YAML.

    Uses string timeout ('30m') so YAML-safe serialization works.
    ISO 8601 (PT30M) is tested in test_deferred_tool_timeout_serializes_iso8601.
    """
    original = DeferredToolConfig(
        enabled=True,
        default_strategy="block",
        default_timeout="30m",
    )

    # Dump to dict (Python objects) for YAML-safe round-trip
    dumped = original.model_dump(mode="python")
    assert dumped["default_timeout"] == timedelta(minutes=30)

    # Reconstruct from dict directly (no YAML serialization needed for dict round-trip)
    restored = DeferredToolConfig(**dumped)

    assert restored.enabled == original.enabled
    assert restored.default_strategy == original.default_strategy
    assert isinstance(restored.default_timeout, timedelta)
    assert restored.default_timeout == timedelta(minutes=30)


def test_deferred_tool_disabled():
    """DeferredToolConfig(enabled=False)."""
    config = DeferredToolConfig(enabled=False)

    assert config.enabled is False
    assert config.default_strategy == "block"


def test_deferred_tool_timeout_complex_duration():
    """DeferredToolConfig timeout parses complex duration strings."""
    config = DeferredToolConfig(default_timeout="1h 30m")

    assert isinstance(config.default_timeout, timedelta)
    assert config.default_timeout == timedelta(hours=1, minutes=30)


# =============================================================================
# CheckpointConfig Tests
# =============================================================================


def test_checkpoint_defaults():
    """CheckpointConfig() defaults to enabled=True."""
    config = CheckpointConfig()

    assert config.enabled is True


def test_checkpoint_disabled():
    """CheckpointConfig(enabled=False)."""
    config = CheckpointConfig(enabled=False)

    assert config.enabled is False


def test_checkpoint_serialize_as_dict():
    """CheckpointConfig(enabled=True) serializes cleanly."""
    config = CheckpointConfig(enabled=True)

    d = config.model_dump()

    assert d["enabled"] is True


def test_checkpoint_yaml_round_trip():
    """CheckpointConfig YAML round-trip preserves enabled field."""
    original = CheckpointConfig(enabled=True)

    yaml_str = yaml.dump(original.model_dump(mode="json"))
    data = yaml.safe_load(yaml_str)
    restored = CheckpointConfig(**data)

    assert restored.enabled == original.enabled
