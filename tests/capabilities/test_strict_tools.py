"""Unit tests for StrictToolsCapability.

Covers the strict flag baking logic and YAML config registration.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai.tools import RunContext, ToolDefinition
import pytest

from wolfharness.capabilities.strict_tools import StrictToolsCapability
from wolfharness_config.capabilities import (
    StrictToolsCapabilityConfig,
    build_capability,
    is_known_capability_type,
)

pytestmark = pytest.mark.unit


def _tool_def(name: str = "tool_a", strict: bool | None = None) -> ToolDefinition:
    """Build a minimal tool definition with an explicit strict value."""
    return ToolDefinition(name=name, strict=strict)


def _ctx() -> RunContext[Any]:
    """A run context placeholder — never touched by the capability."""
    return cast(RunContext[Any], None)


async def test_prepare_tools_forces_strict_when_none() -> None:
    """Definitions with ``strict=None`` are upgraded to ``strict=True``."""
    cap = StrictToolsCapability()
    result = await cap.prepare_tools(_ctx(), [_tool_def(strict=None)])

    assert len(result) == 1
    assert result[0].strict is True


async def test_prepare_tools_keeps_explicit_strict() -> None:
    """Definitions that already carry an explicit strict value are untouched."""
    cap = StrictToolsCapability()
    result = await cap.prepare_tools(
        _ctx(),
        [_tool_def(strict=False), _tool_def(strict=True)],
    )

    assert [td.strict for td in result] == [False, True]


async def test_prepare_tools_disabled_passthrough() -> None:
    """When ``enabled=False``, definitions pass through unchanged."""
    cap = StrictToolsCapability(enabled=False)
    result = await cap.prepare_tools(_ctx(), [_tool_def(strict=None)])

    assert result[0].strict is None


async def test_prepare_output_tools_off_by_default() -> None:
    """Output tools are untouched unless ``apply_to_output_tools`` is set."""
    cap = StrictToolsCapability()
    result = await cap.prepare_output_tools(_ctx(), [_tool_def(strict=None)])

    assert result[0].strict is None


async def test_prepare_output_tools_when_enabled() -> None:
    """``apply_to_output_tools=True`` upgrades output-tool definitions."""
    cap = StrictToolsCapability(apply_to_output_tools=True)
    result = await cap.prepare_output_tools(_ctx(), [_tool_def(strict=None)])

    assert result[0].strict is True


def test_yaml_short_name_registered() -> None:
    """``strict_tools`` is a known capability type."""
    assert is_known_capability_type("strict_tools")


def test_build_capability_from_config() -> None:
    """The typed config builds a StrictToolsCapability instance."""
    cap = build_capability(StrictToolsCapabilityConfig())

    assert isinstance(cap, StrictToolsCapability)
    assert cap.enabled is True
    assert cap.apply_to_output_tools is False