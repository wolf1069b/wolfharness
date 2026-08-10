"""Unit tests for _coerce_subagent_display_mode().

Tests all known values (legacy, zed, inline, tool_box) and unknown values,
verifying return values and log messages.
"""

from __future__ import annotations

import logging

import pytest

from wolfharness_server.acp_server.server import _coerce_subagent_display_mode


# ---------------------------------------------------------------------------
# Known values: pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coerce_legacy():
    """'legacy' passes through unchanged."""
    assert _coerce_subagent_display_mode("legacy") == "legacy"


@pytest.mark.unit
def test_coerce_zed():
    """'zed' passes through unchanged."""
    assert _coerce_subagent_display_mode("zed") == "zed"


# ---------------------------------------------------------------------------
# Deprecated values: map to legacy with warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coerce_inline_deprecated(caplog: pytest.LogCaptureFixture):
    """'inline' maps to 'legacy' with a deprecation warning."""
    caplog.set_level(logging.WARNING)
    result = _coerce_subagent_display_mode("inline")
    assert result == "legacy"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "deprecated" in record.getMessage()
    assert "inline" in record.getMessage()


@pytest.mark.unit
def test_coerce_tool_box_deprecated(caplog: pytest.LogCaptureFixture):
    """'tool_box' maps to 'legacy' with a deprecation warning."""
    caplog.set_level(logging.WARNING)
    result = _coerce_subagent_display_mode("tool_box")
    assert result == "legacy"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "deprecated" in record.getMessage()
    assert "tool_box" in record.getMessage()


# ---------------------------------------------------------------------------
# Unknown values: fall back to legacy with warning
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coerce_unknown_fallback(caplog: pytest.LogCaptureFixture):
    """'unknown' maps to 'legacy' with a warning about unknown mode."""
    caplog.set_level(logging.WARNING)
    result = _coerce_subagent_display_mode("unknown")
    assert result == "legacy"
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "WARNING"
    assert "Unknown" in record.getMessage()
    assert "unknown" in record.getMessage()
    assert "falling back" in record.getMessage()


# ---------------------------------------------------------------------------
# New known values: pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coerce_qwen(caplog: pytest.LogCaptureFixture):
    """'qwen' passes through unchanged with no warning."""
    caplog.set_level(logging.WARNING)
    result = _coerce_subagent_display_mode("qwen")
    assert result == "qwen"
    assert len(caplog.records) == 0, (
        f"Expected no warnings, got: {[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.unit
def test_pool_server_config_accepts_qwen():
    """ACPPoolServerConfig accepts 'qwen' without ValidationError."""
    from wolfharness_config.pool_server import ACPPoolServerConfig

    config = ACPPoolServerConfig(subagent_display_mode="qwen")
    assert config.subagent_display_mode == "qwen"
